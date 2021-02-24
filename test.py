from adv_autoencoder import AAE
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist

(X_train, label_train), _ = mnist.load_data()
X_train = tf.cast(X_train, tf.float32) / 255
n_data = X_train.shape[0]

print(n_data)
X = X_train[0][None,:,:,None]

model = AAE()
model(X)
