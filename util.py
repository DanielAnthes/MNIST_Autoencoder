import matplotlib.pyplot as plt

def plot_latent_space(X, classes):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(X[:,0], X[:,1], c=classes)

def plot_reconst(x,r):
    plt.figure()
    plt.subplot(121)
    plt.imshow(x)
    plt.title("original")
    plt.subplot(122)
    plt.imshow(r)
    plt.title("reconstruction")
