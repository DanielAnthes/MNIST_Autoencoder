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

# training loop
i = 1
for X in dataset:
    X = X[:,:,:,None]
    print(f"BATCH: {i}/{n_batch}, NUM IMGS: {X.shape[0]}", end="\r")
    loss, grad = model.train_autoencoder(X)
    optimizer.apply_gradients(zip(grad, model.autoencoder_weights))
    i += 1
    with train_summary_writer.as_default():
        tf.summary.scalar('reconstruction loss', loss, step=i)
    
    if i % 10 == 0 and i > 0:
        batchsize = X.shape[0]
        enc_loss, disc_loss, enc_grad, disc_grad = model.train_GAN(X, n_fakes = batchsize)
        if tf.math.is_nan(enc_loss) or tf.math.is_nan(disc_loss):
            print(f"crash in iteration {i}")
            raise Exception
        optimizer.apply_gradients(zip(disc_grad, model.discriminator_weights))
        optimizer.apply_gradients(zip(enc_grad, model.encoder_weights))
        
        with train_summary_writer.as_default():
            tf.summary.scalar('encoder loss', enc_loss, step=i)
            tf.summary.scalar('discriminator loss', disc_loss, step=i)


model.save_weights("../models/mnist_aae" + current_time)
