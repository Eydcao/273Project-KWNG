#%%
import tensorflow as tf


def pow_10(x, dtype=tf.float32, device='cpu'):
    return tf.pow(tf.constant([10], dtype=dtype), x)


class Gaussian(object):
    def __init__(self, D, log_sigma, dtype=tf.float32, device='cpu'):
        self.D = D
        self.params = log_sigma
        self.dtype = dtype
        self.device = device
        self.adaptive = False
        self.params_0 = log_sigma

    def get_exp_params(self):
        return pow_10(self.params, dtype=self.dtype, device=self.device)

    def update_params(self, log_sigma):
        self.params = log_sigma

    def square_dist(self, X, Y):
        # Squared distance matrix of pariwise elements in X and basis
        # Inputs:
        # X     : N by d matrix of data points
        # basis : M by d matrix of basis points
        # output: N by M matrix
        return self._square_dist(X, Y)

    def kernel(self, X, Y):
        # Gramm matrix between vectors X and basis
        # Inputs:
        # X     : N by d matrix of data points
        # basis : M by d matrix of basis points
        # output: N by M matrix
        return self._kernel(self.params, X, Y)

    def dkdxdy(self, X, Y, mask=None):
        return self._dkdxdy(self.params, X, Y, mask=mask)

    def _square_dist(self, X, Y):
        n_x, d = X.shape
        n_y, d = Y.shape
        dist = -2 * tf.einsum('mr,nr->mn', X, Y) + tf.transpose(
            tf.reshape(tf.tile(tf.reduce_sum(X**2, 1),
                               tf.constant([n_y])), [n_y, n_x])) + tf.reshape(
                                   tf.tile(tf.reduce_sum(Y**2, 1),
                                           tf.constant([n_x])), [n_x, n_y])
        return dist

    def _kernel(self, log_sigma, X, Y):
        N, d = X.shape
        sigma = pow_10(log_sigma, dtype=self.dtype, device=self.device)
        tmp = self._square_dist(X, Y)
        dist = tf.math.maximum(tmp, tf.zeros_like(tmp))
        if self.adaptive:
            ss = tf.stop_gradient(tf.reduce_mean(dist))
            dist = dist / (ss + 1e-5)
        return tf.exp(-0.5 * dist / sigma)

    def _dkdxdy(self, log_sigma, X, Y, mask=None):
        # X : [M,T]
        # Y : [N,R]
        # dkdxdy ,   dkdxdy2  = [M,N,T,R]
        # dkdx = [M,N,T]
        N, d = X.shape
        sigma = pow_10(log_sigma, dtype=self.dtype, device=self.device)
        gram = self._kernel(log_sigma, X, Y)

        D = (tf.expand_dims(X, 1) - tf.expand_dims(Y, 0)) / sigma

        I = tf.ones(D.shape[-1], dtype=self.dtype) / sigma

        dkdy = tf.einsum('mn,mnr->mnr', gram, D)
        dkdx = -dkdy

        if mask is None:
            D2 = tf.einsum('mnt,mnr->mntr', D, D)
            I = tf.eye(D.shape[-1], dtype=self.dtype) / sigma
            dkdxdy = I - D2
            dkdxdy = tf.einsum('mn, mntr->mntr', gram, dkdxdy)
        else:
            D_masked = tf.einsum('mnt,mt->mn', D, mask)
            D2 = tf.einsum('mn,mnr->mnr', D_masked, D)

            dkdxdy = tf.einsum('mn,mr->mnr', gram, mask) / sigma - tf.einsum(
                'mn, mnr->mnr', gram, D2)
            dkdx = tf.einsum('mnt,mt->mn', dkdx, mask)

        return dkdxdy, dkdx, gram


#%%
if __name__ == "__main__":
    g = Gaussian(0, 1)
    print("get_exp_params ", g.get_exp_params())
    x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    y = tf.constant([[2, 3, 1], [5, 6, 4], [7, 8, 9]], dtype=tf.float32)
    ds = g.square_dist(x, y)
    print("sqr dist x y ", ds)

    # %%
    k = g.kernel(x, y)
    print("kernel x y ", k)
    # %%
    grad = g.dkdxdy(x, y)
    print("grad x y ", grad)

# %%
