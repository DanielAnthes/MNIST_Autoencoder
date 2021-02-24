import tensorflow as tf

class AAE(tf.keras.Model):

    def __init__(self):
        super(AAE, self).__init__()
        
        # encoder
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=4, strides=2, activation='relu')
        self.fc1 = tf.keras.layers.Dense(32, activation='relu')

        # decoder
        self.fc2 = tf.keras.layers.Dense(1352, activation='relu')
        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, activation='relu')

    def encode(self,X):
        nbatch = X.shape[0]
        print(X.shape)
        X = self.conv1(X)
        print(X.shape)
        X = tf.reshape(X, [nbatch, -1])
        print(X.shape)
        X = self.fc1(X)
        return X


    def decode(self,X):
        nbatch = X.shape[0]
        X = self.fc2(X)
        print(X.shape)
        X = tf.reshape(X, [nbatch, 13, 13, 8])
        X = self.deconv1(X)
        print(X.shape)
        return X

    def call(self, X):
        Z = self.encode(X)
        R = self.decode(Z)

        return Z,R
        
