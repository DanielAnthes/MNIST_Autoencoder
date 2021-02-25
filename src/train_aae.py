import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.datasets import mnist
from math import ceil
from adv_autoencoder import AAE
from util import plot_reconst, plot_latent_space
import datetime

### LOGGING ###

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '../logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_summary_writer.set_as_default()


#### HYPERPARAMETERS ###

BATCHSIZE = 400 
DATASET_REPS = 12

### DATA ###

(X_train, label_train), _ = mnist.load_data()
X_train = tf.cast(X_train, tf.float32) / 255
n_data = X_train.shape[0]
dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(n_data) # TODO redo shuffling in proper location?
dataset = dataset.batch(BATCHSIZE)
dataset = dataset.repeat(DATASET_REPS)
n_batch = ceil((n_data/BATCHSIZE)*DATASET_REPS)
print(f"DATASET SIZE: {n_data}\nBATCHSIZE: {BATCHSIZE}\nDATASET REPS: {DATASET_REPS}")

### MODEL ###

model = AAE()
optimizer = tf.keras.optimizers.Adam()

### TRAINING ###
model.fit(dataset, BATCHSIZE, optimizer, train_summary_writer)
