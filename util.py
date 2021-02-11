import matplotlib.pyplot as plt

def plot_latent_space(X, classes):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(X[:,0], X[:,1], c=classes)
    return fig

def plot_reconst(x,r):
    nimgs = x.shape[0]
    fig = plt.figure()
    for i in range(nimgs):
        plt.subplot(nimgs,2,i*2+1)
        plt.imshow(x[i,:,:])
        plt.title("original")
        plt.subplot(nimgs,2,i*2+2)
        plt.imshow(r[i,:,:])
        plt.title("reconstruction")
    return fig
