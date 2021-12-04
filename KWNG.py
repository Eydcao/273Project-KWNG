#%%
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import GuaranteeConst
from gaussian import Gaussian


class KWNG(object):
    def __init__(self, num_basis=10, eps=1e-5, with_diag_mat=True):
        super(KWNG, self).__init__()
        self.kernel = Gaussian(0, -5)
        self.eps = eps
        self.thresh = 0.
        self.num_basis = num_basis
        self.with_diag_mat = with_diag_mat
        self.K = None
        self.T = None

    def compute_cond_matrix(self, outputs, net_pars):
        L = 0
        d = 0
        if len(outputs.shape) == 1:
            L = outputs.shape[0]
        else:
            L, d = outputs.shape

        # stop gradient for the basis
        if d == 0:
            outputs = tf.expand_dims(outputs, -1)
            d = 1
        basis = tf.stop_gradient(tf.random.shuffle(outputs)[0:self.num_basis])

        mask_int = tf.random.uniform([self.num_basis],
                                     minval=0,
                                     maxval=d,
                                     dtype=tf.int32)
        mask = tf.one_hot(mask_int, d, dtype=outputs.dtype)
        # mask = tf.ones([self.num_basis, d], dtype=outputs.dtype)
        # print(mask)

        sigma = tf.stop_gradient(
            tf.math.log(tf.reduce_mean(self.kernel.square_dist(basis,
                                                               outputs))))
        print(" sigma:   " + str(tf.exp(sigma).numpy()))
        sigma /= np.log(10.)

        if hasattr(self.kernel, 'params_0'):
            self.kernel.params = self.kernel.params_0 + sigma

        dkdxdy, dkdx, _ = self.kernel.dkdxdy(basis, outputs, mask=mask)
        self.K = (1. / L) * tf.einsum('mni,kni->mk', dkdxdy, dkdxdy)
        aux_loss = tf.reduce_mean(dkdx, 1)
        # get Jaccobian by tensorflow aux_loss->net_parameters
        # then set T to the Jaccobian

        return aux_loss

    def compute_natural_gradient(self, g):
        ss, uu, vv = tf.linalg.svd(tf.cast(self.K, tf.float64))
        ss_inv, mask = self.pseudo_inverse(ss)
        ss_inv = tf.sqrt(ss_inv)
        vv = tf.cast(tf.einsum('i,ji->ij', ss_inv, vv), tf.float32)
        self.T = tf.einsum('ij,jk->ik', vv, self.T)
        cond_g, G, D = self.make_system(g, mask)

        # print(G)

        try:
            U = tf.linalg.cholesky(G)
            cond_g = tf.squeeze(
                tf.linalg.cholesky_solve(U, tf.expand_dims(cond_g, -1)), -1)
        except:
            try:
                cond_g = tf.squeeze(
                    tf.linalg.solve(G, tf.expand_dims(cond_g, -1)), -1)
            except:
                pinv = tf.linalg.pinv(G)
                cond_g = tf.einsum('mk,k', pinv, cond_g)

        cond_g = tf.einsum('md,m->d', self.T, cond_g)
        cond_g = (g - cond_g) / self.eps
        cond_g = D * cond_g
        return cond_g

    def make_system(self, g, mask):
        if self.with_diag_mat == 1:
            D = tf.sqrt(tf.reduce_sum(self.T * self.T, 0))
            D = 1. / (D + 1e-8)
        elif self.with_diag_mat == 0:
            D = tf.ones(self.T.shape[1], dtype=self.T.dtype)

        cond_g = D * g
        cond_g = tf.einsum('md,d->m', self.T, cond_g)
        P = tf.cast(mask, cond_g.dtype)
        G = tf.einsum('md,d,kd->mk', self.T, D,
                      self.T) + self.eps * tf.linalg.diag(P)
        return cond_g, G, D

    def pseudo_inverse(self, S):
        SS = 1. / S
        mask = (S > self.thresh)
        SS = SS * tf.cast(mask, SS.dtype)
        mask = (S > self.thresh)
        return SS, mask


#%%
if __name__ == "__main__":
    KG = KWNG(Gaussian)
    KG.T = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
    cond_g, G, D = KG.make_system(tf.constant([1, 2, 3], dtype=tf.float32),
                                  tf.constant([True, True, True]))
    print(cond_g)
    print(G)
    print(D)

    KG.compute_cond_matrix(
        tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=tf.float32), None)
# %%
