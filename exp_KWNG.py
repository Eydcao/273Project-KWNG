import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
from KWNG import KWNG

import matplotlib.pyplot as plt
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_validate = x_test[:100]


class Autoencoder(object):
    def __init__(
            self,
            input_shape=(28, 28),
            use_kwng=False,
            learning_rate=1e-2,
    ):
        self.model = tf.keras.Sequential([
            layers.InputLayer(input_shape),
            layers.Flatten(name="x0"),
            layers.Dense(512, use_bias=False, name="pre_x1"),
            layers.Activation("relu", name="x1"),
            layers.Dense(256, use_bias=False, name="pre_x2"),
            layers.Activation("relu", name="x2"),
            layers.Dense(128, use_bias=False, name="pre_x3"),
            layers.Activation("relu", name="x3"),
            layers.Dense(64, use_bias=False, name="pre_x4"),
            layers.Activation("relu", name="x4"),
            layers.Dense(32, use_bias=False, name="pre_x5"),
            layers.Activation("relu", name="x5"),  # latent representation
            layers.Dense(64, use_bias=False, name="pre_x6"),
            layers.Activation("relu", name="x6"),
            layers.Dense(128, use_bias=False, name="pre_x7"),
            layers.Activation("relu", name="x7"),
            layers.Dense(256, use_bias=False, name="pre_x8"),
            layers.Activation("relu", name="x8"),
            layers.Dense(512, use_bias=False, name="pre_x9"),
            layers.Activation("relu", name="x9"),
            layers.Dense(784, use_bias=False, name="pre_x10"),
            layers.Activation("relu", name="x10"),
            layers.Reshape((28, 28))
        ])
        outputs = []
        pre_outputs = []
        for layer in self.model.layers:
            if layer.name[0] == 'x':
                outputs.append(layer.output)
            elif layer.name[0:3] == "pre":
                pre_outputs.append(layer.output)
        self.extend_model = tf.keras.Model(
            self.model.input, [self.model.output, outputs, pre_outputs])
        self.dtype = tf.float32
        # self.ngd = ngd
        self.sigma = 1e-6
        self.training_error = []
        # KWNG class
        self.kwng = KWNG()
        self.use_kwng = use_kwng
        self.dot_prod = 0
        self.save_path = 'kwng_model'
        if not self.use_kwng:
            self.save_path = 'sgd'

        self.old_loss = -1.
        self.reduction_coeff = 0.85
        self.dumping_freq = 5
        self.min_red = 0.25
        self.max_red = 0.75
        self.eps_min = 1e-10
        self.eps_max = 1e5
        self.dumping_counter = 0
        self.reduction_factor = 0.

        self.lr = learning_rate

    def __loss(self, x_input):
        outputs = self.extend_model(x_input)
        loss = losses.MeanSquaredError()(x_input, outputs[0])
        self.x = outputs[1]
        self.pre_x = outputs[2]
        return loss, outputs[0]

    def __flat_grad(self, grad):
        grad_flat = []
        for g in grad:
            grad_flat.append(tf.reshape(g, [-1]))
        grad_flat = tf.concat(grad_flat, 0)
        return grad_flat

    def __struct_grad(self, grad, flat_grad):
        grad_st = []
        id = 0
        for g in grad:
            x, y = g.shape
            size = x * y
            grad_st.append(tf.reshape(flat_grad[id:id + size], g.shape))
            id += size
        return grad_st

    def __flat_jacobian(self, Jacobian):
        flat_J = []
        for J in Jacobian:
            flat_J.append(tf.reshape(J, [J.shape[0], -1]))
        flat_J = tf.concat(flat_J, 1)
        return flat_J

    def __grad(self, x_input):
        with tf.GradientTape() as tape:
            loss_value, _ = self.__loss(x_input)

        grad = tape.gradient(loss_value, self.__wrap_training_variables())
        flat_grad = self.__flat_grad(grad)

        # compute conditional matrix
        with tf.GradientTape() as tape2:
            _, outputs = self.__loss(x_input)
            if self.use_kwng:
                # Adjust epsilon
                if self.old_loss > -1:
                    # Compute the reduction ratio
                    red = 2. * (self.old_loss -
                                loss_value) / (self.lr * self.dot_prod + 1e-6)
                    if red > self.reduction_factor:
                        self.reduction_factor = red.numpy()
                self.dumping_counter = self.dumping_counter + 1
                if self.old_loss > -1 and np.mod(self.dumping_counter,
                                                 self.dumping_freq) == 0:
                    if self.reduction_factor < self.min_red and self.kwng.eps < self.eps_max:
                        self.kwng.eps /= self.reduction_coeff
                    if self.reduction_factor > self.max_red and self.kwng.eps > self.eps_min:
                        self.kwng.eps *= self.reduction_coeff
                    print("New epsilon: " + str(self.kwng.eps) +
                          ", Reduction_factor: " + str(self.reduction_factor))
                self.reduction_factor = 0.
                # End adjust epsilon
                L, M, N = outputs.shape
                aux_loss = self.kwng.compute_cond_matrix(
                    tf.reshape(outputs, [L, M * N]),
                    self.__wrap_training_variables())
        if self.use_kwng:
            self.kwng.T = self.__flat_jacobian(
                tape2.jacobian(aux_loss, self.__wrap_training_variables()))

        # calculate KWNG estimator here
        if self.use_kwng:
            natural_grad = self.kwng.compute_natural_gradient(flat_grad)
            # If the dot product is negative, just use the euclidean gradient
            self.dot_prod = tf.reduce_sum(flat_grad * natural_grad)
            if self.dot_prod <= 0:
                print('bad ng direction, using normal gradient')
                natural_grad = flat_grad
            # graient clipping
            norm_grad = tf.norm(natural_grad)
            clip_coef = 1. / (norm_grad + 1e-6)
            if clip_coef < 1.:
                self.dot_prod = self.dot_prod / norm_grad
                natural_grad = natural_grad / norm_grad
            # restore gradient structure to list of tensors
            grad = self.__struct_grad(grad, natural_grad)
            # Saving the current value of the loss
            self.old_loss = loss_value.numpy()
        # end calculate KWNG estimator here
        return loss_value, grad

    def __wrap_training_variables(self):
        var = self.model.trainable_variables
        return var

    def summary(self):
        self.model.summary()

    # The training function
    def fit(self,
            x_train,
            x_validate,
            tf_optimizer,
            tf_epochs=5000,
            batch_size=1024,
            shuffle_buffer_size=10 * 512):
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(
            batch_size)
        print(len(train_dataset))
        validate_data = tf.convert_to_tensor(x_validate, dtype='float32')
        for epoch in range(tf_epochs):
            # Optimization step
            for data in train_dataset:
                loss_value, grads = self.__grad(data)
                tf_optimizer.apply_gradients(
                    zip(grads, self.__wrap_training_variables()))
            # if (epoch % 1 == 0):
            if (epoch % 5 == 0):
                if (self.save_path != None):
                    self.model.save('outputs/' + self.save_path + '/epoch_' +
                                    str(epoch) + '.h5')
            print(f"epoch: {epoch}, loss_value: {self.__loss(x_validate)[0]}")
            self.training_error.append(loss_value)


if __name__ == "__main__":
    import numpy as np
    lr = 0.01
    use_kwng = False
    net1 = Autoencoder(use_kwng=True, learning_rate=lr)
    net1.summary()
    tf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                            beta_1=0.99,
                                            epsilon=1e-1)
    net1.fit(x_train, x_validate, tf_optimizer, 1000, batch_size=1024)
    net2 = Autoencoder(use_kwng=False, learning_rate=lr)
    net2.summary()
    net2.fit(x_train, x_validate, tf_optimizer, 1000, batch_size=1024)
    # save error information
    np.save('kwng_error.npy', np.log(net1.training_error))
    np.save('sgd.npy', np.log(net2.training_error))
