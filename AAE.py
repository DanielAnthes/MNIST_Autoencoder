import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class AAE(tf.keras.Model):

    def __init__(self):

        super(AAE, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=3, activation='relu') 
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=3, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')
        self.encode_dense = tf.keras.layers.Dense(2, activation='relu')

        self.reconst_dense = tf.keras.layers.Dense(2, activation='relu')

        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=4, strides=1, activation='relu')
        self.deconv2 = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation='relu')
        self.deconv3 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=3, activation='sigmoid')

    def encode(self, x):
        batchsize = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.reshape(x, [-1, 32]) # flatten output of convolutions
        x = self.encode_dense(x)
        return x

    def decode(self, x):
        x = self.reconst_dense(x)
        x = tf.reshape(x, [-1, 1, 1, 2])
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

    def loss(self, x, r):
        ssqr = tf.math.square(x - r)
        reconstr_loss = tf.math.reduce_sum(ssqr)
        return reconstr_loss

    def call(self,x):
        z = self.encode(x)
        r = self.decode(z)
        return z, r

    def load(self, name="mnist_aae"):
      self.load_weights(name)


