import os
import re
from collections import Counter
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Qt4Agg", warn=False, force=True)
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pjoin = os.path.join


def word_freqs(topk: int = 25):
    word_counts = Counter() 
    with open(pjoin("..", "txt", "hamlet.txt"), "rt") as f:
        text = " ".join(f.readlines())
        text = re.sub("[^0-9a-zA-Z ]+", "", text)
        all_words = [word.lower() for word in text.split(" ")]
        word_counts = Counter(all_words)

    n_words = len(all_words)
    words, counts = zip(*word_counts.most_common(topk))
    freqs = np.asarray([c / n_words for c in counts])
    return words, freqs

def plot_scatter_points(freqs):
    xs = np.arange(1, 1 + freqs.size)
    ys = freqs
    plt.figure(figsize=(10,10))
    plt.scatter(xs, ys)
    plt.title(r"Word Frequency vs. Rank", fontsize=20)
    plt.xlabel("Rank", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pjoin("..", "images", "shakespeare-freq-scatter.png"))

def fit_zipf_ols(freq_counts_desc: np.ndarray):
    ranks = np.arange(1, 1 + freq_counts_desc.size)
    A = np.stack([np.ones_like(freq_counts_desc), np.log(ranks)], axis=1)
    b = np.log(freq_counts_desc)
    f = lambda x: A @ x - b
    opt_result = least_squares(f, x0=np.zeros(2))

    C = np.exp(opt_result.x[0])
    alpha = opt_result.x[1]
    mse = np.sqrt(opt_result.cost)

    return C, alpha, mse

def plot_zipf_param_surface(empirical_freqs):
    c_plot = np.linspace(-10, 10, 100)
    alpha_plot = np.linspace(-1, 0.0, 200)

    C_mesh, Alpha_mesh = np.meshgrid(c_plot, alpha_plot)
    ranks = np.arange(1, freqs.size + 1)
    Z = np.empty_like(C_mesh)
    for i, alpha in enumerate(alpha_plot):
        for j, c in enumerate(c_plot):
            yhat = c * ranks ** alpha
            residual = empirical_freqs - yhat
            mse = np.sqrt(np.mean(np.square(residual)))
            Z[i, j] = mse
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(C_mesh, Alpha_mesh, Z, cmap="hot")
    ax.set_title(r"$\sum_{i=1}^N (f_i - C x_i^\alpha)^2$")
    ax.set_xlabel(r"$C$", fontsize=14)
    ax.set_ylabel(r"$\alpha$", fontsize=14)
    plt.savefig(pjoin("..", "images", "shakespeare-zipf-param-surface.png"))

def plot_zipf_transformed_param_surface(empirical_freqs):
    log_c_plot = np.log(np.linspace(0.01, 0.2, 100))
    alpha_plot = np.linspace(-1, 0.0, 200)

    C_mesh, Alpha_mesh = np.meshgrid(log_c_plot, alpha_plot)
    ranks = np.arange(1, freqs.size + 1)
    log_freqs = np.log(empirical_freqs)
    Z = np.empty_like(C_mesh)
    for i, alpha in enumerate(alpha_plot):
        for j, log_c in enumerate(log_c_plot):
            yhat = log_c + alpha * np.log(ranks)
            residual = log_freqs - yhat
            mse = np.sqrt(np.mean(np.square(residual)))
            Z[i, j] = mse
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(C_mesh, Alpha_mesh, Z, cmap="hot")
    ax.set_title(r"$\sum_{i=1}^N (f_i - \log(C) - \alpha \log(x_i))^2$")

    plt.savefig(pjoin("..", "images", "shakespeare-zipf-transformed-param-surface.png"))

def plot_transformed_scatter_points(freqs, N):
    xs = np.log(np.arange(1, 1 + freqs.size))
    ys = np.log(freqs)
    plt.figure(figsize=(10,10))
    plt.scatter(xs, ys)
    plt.title(r"$\log(f)$ vs. $\log(rank)$")
    plt.xlabel(r"$\log(rank)$", fontsize=14)
    plt.ylabel(r"$\log(f)$", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pjoin("..", "images", "shakespeare-zipf-transformed-param-scatter.png"))

def plot_zipf_transformed_param_contours(empirical_freqs):
    log_c_plot = np.log(np.linspace(0.01, 0.2, 100))
    alpha_plot = np.linspace(-1, 0.0, 200)

    C_mesh, Alpha_mesh = np.meshgrid(log_c_plot, alpha_plot)
    ranks = np.arange(1, freqs.size + 1)
    log_freqs = np.log(empirical_freqs)
    Z = np.empty_like(C_mesh)
    for i, alpha in enumerate(alpha_plot):
        for j, log_c in enumerate(log_c_plot):
            yhat = log_c + alpha * np.log(ranks)
            residual = log_freqs - yhat
            mse = np.sqrt(np.mean(np.square(residual)))
            Z[i, j] = mse

    fig = plt.figure(figsize=(10, 10))
    plt.contour(C_mesh, Alpha_mesh, Z, cmap="hot", levels=20)
    plt.title(r"$\sum_{i=1}^N (f_i - \log(C) - \alpha \log(x_i))^2$")
    plt.xlabel(r"$\log(C)$", fontsize=14)
    plt.ylabel(r"$\alpha$", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pjoin("..", "images", "shakespeare-zipf-transformed-param-contours.png"))

def plot_ols_zipf_fit(empirical_freqs, N):
    plt.figure(figsize=(10,10))
    plt.title("Word Frequency vs. Rank")
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Word Frequency", fontsize=14)
    plt.grid(True)
    
    C_ols, alpha_ols, _ = fit_zipf_ols(empirical_freqs)
    xs = np.asarray([i for i in range(1, N + 1)])
    plt.scatter(xs, empirical_freqs, alpha=0.9)
    plt.plot(xs, C_ols * xs ** alpha_ols, "g", linewidth=2, label=rf"$f_{{ols}}(x) = {C_ols:.2}x^{{{alpha_ols:.2}}}$")
    plt.ylim([0, 0.04])
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin("..", "images", "shakespeare-zipf-fit.png"))

if __name__ == "__main__":
    N = 100
    most_freq_words, freqs = word_freqs(N)
    plot_scatter_points(freqs)
    # plot_zipf_param_surface(freqs)
    plot_transformed_scatter_points(freqs, N)
    # plot_zipf_transformed_param_contours(freqs)
    # plot_zipf_transformed_param_surface(freqs)

    # plot_ols_zipf_fit(freqs, N)
