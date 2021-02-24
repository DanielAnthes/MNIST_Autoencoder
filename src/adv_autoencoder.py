import tensorflow as tf
import numpy as np

class AAE(tf.keras.Model):

    def __init__(self, datashape=(28,28,1)):
        super(AAE, self).__init__()
        
        # encoder
        self.conv1 = tf.keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=4, strides=2, activation='relu', name='enc_conv')
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', name='enc_fc1')
        self.fc2 = tf.keras.layers.Dense(2, activation=None, name='enc_fc2')

        # decoder
        self.fc3 = tf.keras.layers.Dense(128, activation='relu', name='dec_fc1')
        self.fc4 = tf.keras.layers.Dense(1352, activation='relu', name='dec_fc2')
        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, activation='sigmoid', name='dec_deconv')  # outputs must be pixel intensities between 0 and 1

        # discriminator
        self.fc5 = tf.keras.layers.Dense(128, activation='relu', name='disc_fc1')
        self.fc6 = tf.keras.layers.Dense(2, activation='softmax', name='disc_fc2')
        
        data = np.random.normal(size=(1,*datashape))
        z,r = self.call(data)
        self.discriminate(z)

        trainable_vars = self.trainable_variables
        self.encoder_weights = [v for v in trainable_vars if 'enc_' in v.name]
        self.decoder_weights = [v for v in trainable_vars if 'dec_' in v.name]
        self.autoencoder_weights = self.encoder_weights + self.decoder_weights
        self.discriminator_weights = [v for v in trainable_vars if 'disc_' in v.name]

        
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

    def discriminate(self, sample):
        x = self.fc5(sample)
        prob = self.fc6(x)
        return prob


    def call(self, X):
        Z = self.encode(X)
        R = self.decode(Z)

        return Z,R
    
    def reconstruction_loss(self, X, R):
        '''
        X       -   original image
        R       -   reconstruction

        computes pixel wise L2 loss
        '''
        ssqr = tf.math.square(X - R)
        loss = tf.math.reduce_sum(ssqr)
        return loss

    def discriminator_loss(self, probs, labels):
        '''
        probs       -   probabilities for each class
        labels      -   one hot encoding of labels
        
        calculates negative log likelihood loss for discriminator
        '''
        prob_true = tf.math.log(probs) * labels
        nll = - tf.math.reduce_sum(prob_true)
        return nll

    def load(self, name="mnist_aae"):
      self.load_weights(name)

    def train_autoencoder(self, X):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for w in self.autoencoder_weights:
                tape.watch(w)
            z, r  = self.call(X)
            loss = self.reconstruction_loss(X, r)
        grad = tape.gradient(loss, self.autoencoder_weights)
        return loss, grad

    def train_GAN(self, X, n_fakes):
        # TODO shuffle?
        '''
        X           - true datapoints
        n_fakes     - number of samples to draw from prior distribution
        
        encodes true stimuli into latent space and builds a dataset of true encoded stimuli and generated samples from a 2D Gaussian prior
        '''
        dataset_size = X.shape[0] + n_fakes
        labels = np.zeros((dataset_size,2))
        labels[:X.shape[0],0] = 1
        labels[X.shape[0]:,1] = 1
        
        fakes = np.random.normal(size=(n_fakes,2))
                
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as enc_tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
                for w in self.discriminator_weights:
                    tape.watch(w)
                for w in self.encoder_weights:
                    enc_tape.watch(w)
                
                Z = self.encode(X)
                dataset = tf.concat((Z, fakes), axis=0)

                probs = self.discriminate(dataset)
                disc_loss = self.discriminator_loss(probs, labels)
                enc_loss = - disc_loss

        enc_grad = enc_tape.gradient(enc_loss, self.encoder_weights)
        disc_grad = tape.gradient(disc_loss, self.discriminator_weights)
        del tape  # delete persistent tapes
        del enc_tape
        return enc_loss, disc_loss, enc_grad, disc_grad
        

