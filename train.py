import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.datasets import mnist
from math import ceil
from AAE import AAE
from util import plot_reconst, plot_latent_space
import datetime

devices = tf.config.list_physical_devices('GPU') 
print(devices)


### LOGGING ###

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_summary_writer.set_as_default()


#### HYPERPARAMETERS ###

BATCHSIZE = 400 
DATASET_REPS = 100

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

def train_step(model, optimizer, X):
    with tf.GradientTape() as tape:
        z, r  = model(X)
        loss = model.loss(X, r)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# training loop
i = 1
for X in dataset:
    X = X[:,:,:,None]
    print(f"BATCH: {i}/{n_batch}, NUM IMGS: {X.shape[0]}", end="\r")
    loss = train_step(model, optimizer, X)
    i += 1
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=i)

Z, R = model(X_train[:,:,:,None])
fig = plot_latent_space(Z, label_train)
plt.savefig("latentspace.png")
fig2 = plot_reconst(X_train[0:5], R[0:5].numpy().squeeze())
plt.savefig("reconst.png")
plt.show()
