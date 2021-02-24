import matplotlib.pyplot as plt
import numpy as np


def plot_latent_space(X, labels):
    classes = np.unique(labels)
    fig = plt.figure(figsize=(12,8))
    for c in classes:
        samples = X[labels == c, :]
        plt.scatter(samples[:,0], samples[:,1], label=str(c), marker='.', alpha=.8)
    plt.legend()
    return fig


def plot_reconst(x,r):
    nimgs = x.shape[0]
    fig = plt.figure(figsize=(12,8))
    for i in range(nimgs):
        plt.subplot(nimgs,2,i*2+1)
        plt.imshow(x[i,:,:])
        plt.title("original")
        plt.subplot(nimgs,2,i*2+2)
        plt.imshow(r[i,:,:])
        plt.title("reconstruction")
    return fig
