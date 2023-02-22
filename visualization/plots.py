import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_img_from_fig(fig, dpi=180):
    """
    Converts a matplotlib
    figure into a cv2 ndarray
    Args:
        fig (plt.figure): figure of the graph
        dpi (int, optional): Dots per inch. Defaults to 180.
    Returns:
        np.ndarray: image of the plot
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def plot_features_2D(features_X, features_Y, sigma):
    fig = plt.figure(figsize=(10, 10))
    if features_X.shape[1] > 2:
        features_X = TSNE(n_components=2).fit_transform(features_X)
        features_Y = TSNE(n_components=2).fit_transform(features_Y)
    plt.scatter(features_X[:, 0], features_X[:, 1], label='X', c='b')
    plt.scatter(features_Y[:, 0], features_Y[:, 1], label='Y', c='y')
    plt.title("sigma = {:.4f}".format(sigma))

    fig_img =  get_img_from_fig(fig)
    plt.close()
    return fig_img

def plot_stat_histograms(stats):
    metrics = ['Power', 'cdf_arg', 'MMDu', 'mmd_variance']
    fig = plt.figure(figsize=(10, 10))
    factors = [*stats.keys()]
    for i, s in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        for factor in factors:
            plt.hist(stats[factor][f'{factor}/{s}'], alpha=0.7, bins=10, label=factor)
        plt.legend()
        plt.xlabel(s)

    fig_img = get_img_from_fig(fig)
    plt.close()
    return fig_img
