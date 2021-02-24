import tensorflow as tf

class AAE(tf.keras.Model):

    def __init__(self):
        super(AAE, self).__init__()
        
        # encoder
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=4, strides=2, activation='relu')
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(2, activation='relu')

        # decoder
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.fc4 = tf.keras.layers.Dense(1352, activation='relu')
        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, activation='relu')

    def encode(self,X):
        nbatch = X.shape[0]
        X = self.conv1(X)
        X = tf.reshape(X, [nbatch, -1])
        X = self.fc1(X)
        X = self.fc2(X)
        return X


    def decode(self,X):
        nbatch = X.shape[0]
        X = self.fc3(X)
        X = self.fc4(X)
        X = tf.reshape(X, [nbatch, 13, 13, 8])
        X = self.deconv1(X)
        return X

    def call(self, X):
        Z = self.encode(X)
        R = self.decode(Z)

        return Z,R
    
    def loss(self, X, R):
        ssqr = tf.math.square(X - R)
        reconstruction_loss = tf.math.reduce_sum(ssqr)
        return reconstruction_loss

    def load(self, name="mnist_aae"):
      self.load_weights(name)


