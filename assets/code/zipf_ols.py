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
        all_words = [word.lower() for word in text.split() if len(word) > 0]
        n_words = len(all_words)
        word_freqs = Counter(all_words)
        for word in word_freqs.keys():
            word_freqs[word] /= n_words

    words, freqs = zip(*word_freqs.most_common(topk))
    return words, np.asarray(freqs)

def plot_scatter_points(freqs):
    xs = np.arange(1, 1 + freqs.size)
    ys = freqs
    plt.figure(figsize=(10,10))
    plt.scatter(xs, ys)
    plt.title(r"Word Frequency vs. Rank", fontsize=20)
    plt.xlabel("Rank", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.grid(b=True, which="major", linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which="minor", linestyle='--')
    plt.tight_layout()
    plt.savefig(pjoin("..", "images", "shakespeare-freq-scatter.png"))

def fit_zipf_ols(freq_counts_desc: np.ndarray):
    ranks = np.arange(1, 1 + freq_counts_desc.size)
    A = np.stack([np.ones_like(freq_counts_desc), np.log(ranks)], axis=1)
    y = np.log(freq_counts_desc)
    f = lambda x: A @ x - y
    opt_result = least_squares(f, x0=np.random.randn(2), loss="linear")

    K = np.exp(opt_result.x[0])
    alpha = opt_result.x[1]
    mse = np.sqrt(2 * opt_result.cost) # least_squares has 0.5 in objective

    return K, alpha, mse

def plot_zipf_param_surface(empirical_freqs: np.ndarray):
    K_plot = np.linspace(-0.03, .05, 100)
    alpha_plot = np.linspace(-1, -0.3, 200)

    K_mesh, Alpha_mesh = np.meshgrid(K_plot, alpha_plot)
    ranks = np.arange(1, empirical_freqs.size + 1)
    Z = np.empty_like(K_mesh)
    for i, alpha in enumerate(alpha_plot):
        for j, c in enumerate(K_plot):
            yhat = c * ranks ** alpha
            residual = empirical_freqs - yhat
            mse = np.sqrt(np.mean(np.square(residual)))
            Z[i, j] = mse
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K_mesh, Alpha_mesh, Z, cmap="hot")
    ax.set_title(r"$\sum_{i=1}^N (f_i - K r_i^\alpha)^2$")
    ax.set_xlabel(r"$K$", fontsize=14)
    ax.set_ylabel(r"$\alpha$", fontsize=14)
    plt.tight_layout()
    plt.savefig(pjoin("..", "images", "shakespeare-zipf-param-surface.png"))

def plot_zipf_param_contours(empirical_freqs: np.ndarray):
    K_plot = np.linspace(-0.03, .07, 100)
    alpha_plot = np.linspace(-1, -0.3, 200)

    K_mesh, Alpha_mesh = np.meshgrid(K_plot, alpha_plot)
    ranks = np.arange(1, empirical_freqs.size + 1)
    Z = np.empty_like(K_mesh)
    for i, alpha in enumerate(alpha_plot):
        for j, c in enumerate(K_plot):
            yhat = c * ranks ** alpha
            residual = empirical_freqs - yhat
            mse = np.sqrt(np.mean(np.square(residual)))
            Z[i, j] = mse
    fig = plt.figure(figsize=(10, 10))
    plt.contour(K_mesh, Alpha_mesh, Z, cmap="hot", levels=20)
    plt.title(r"$\sum_{i=1}^N (f_i - K r_i^{\alpha_i})^2$")
    plt.xlabel(r"$K$", fontsize=14)
    plt.ylabel(r"$\alpha$", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pjoin("..", "images", "shakespeare-zipf-param-contours.png"))

def plot_zipf_transformed_param_surface(empirical_freqs):
    n_theta = 200
    n_radii = 100
    theta_plot = np.linspace(0, 2*np.pi, n_theta)
    r_plot = np.linspace(0, 3, n_radii)
    R_mesh, Theta_mesh = np.meshgrid(r_plot, theta_plot)

    ranks = np.arange(1, empirical_freqs.size + 1)
    design_mat = np.stack([np.ones_like(ranks), np.log(ranks)], axis=1)
    A = design_mat.T @ design_mat
    eig_vals, eig_vecs = np.linalg.eigh(A)
    major_axis = eig_vecs[:,0]
    angle = np.arctan2(major_axis[1], major_axis[0])
    rot_mat = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    a, b = 1/np.sqrt(eig_vals[0]), 1/np.sqrt(eig_vals[1])

    K_star, alpha_star, _ = fit_zipf_ols(empirical_freqs)
    center = np.asarray([[np.log(K_star)], [alpha_star]])
    scaled_x = (a * R_mesh * np.cos(Theta_mesh)).ravel()
    scaled_y = (b * R_mesh * np.sin(Theta_mesh)).ravel()
    plot_grid = rot_mat @ np.stack([scaled_x, scaled_y], axis=1).T + center

    Z = np.empty(R_mesh.size)
    for i in range(n_theta * n_radii):
        residual = design_mat @ plot_grid[:,i] - np.log(empirical_freqs)
        mse = np.sqrt(np.mean(np.square(residual)))
        Z[i] = mse
    X = plot_grid[0].reshape(R_mesh.shape)
    Y = plot_grid[1].reshape(R_mesh.shape)
    Z = Z.reshape(R_mesh.shape)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="hot")
    ax.set_title(r"$\sum_{i=1}^N (\log\,f_i - \log\,K - \alpha\,\log\, r_i)^2$")
    ax.set_xlabel(r"$\log\, K$", fontsize=14)
    ax.set_ylabel(r"$\alpha$", fontsize=14)
    plt.tight_layout()
    plt.savefig(pjoin("..", "images", "shakespeare-zipf-transformed-param-surface.png"))

def plot_zipf_transformed_param_contours(empirical_freqs):
    n_theta = 200
    n_radii = 100
    theta_plot = np.linspace(0, 2*np.pi, n_theta)
    r_plot = np.linspace(0, 3, n_radii)
    R_mesh, Theta_mesh = np.meshgrid(r_plot, theta_plot)

    ranks = np.arange(1, empirical_freqs.size + 1)
    design_mat = np.stack([np.ones_like(ranks), np.log(ranks)], axis=1)
    A = design_mat.T @ design_mat
    eig_vals, eig_vecs = np.linalg.eigh(A)
    major_axis = eig_vecs[:,0]
    angle = np.arctan2(major_axis[1], major_axis[0])
    rot_mat = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    a, b = 1/np.sqrt(eig_vals[0]), 1/np.sqrt(eig_vals[1])

    y = np.log(empirical_freqs)
    r = lambda x: design_mat @ x - y
    center = least_squares(r, x0=np.zeros(2)).x.reshape((-1,1))
    scaled_x = (a * R_mesh * np.cos(Theta_mesh)).reshape((-1, 1))
    scaled_y = (b * R_mesh * np.sin(Theta_mesh)).reshape((-1, 1))
    plot_grid = rot_mat @ np.hstack([scaled_x, scaled_y]).T + center

    Z = np.empty(R_mesh.size)
    for i in range(n_theta * n_radii):
        residual = r(plot_grid[:,i])
        mse = np.sqrt(np.mean(np.square(residual)))
        Z[i] = mse

    X = plot_grid[0].reshape(R_mesh.shape)
    Y = plot_grid[1].reshape(R_mesh.shape)
    Z = Z.reshape(R_mesh.shape)

    fig = plt.figure(figsize=(10, 10))
    plt.contour(X, Y, Z, cmap="hot", levels=20)
    plt.title(r"$\sum_{i=1}^N (\log\,f_i - \log\, K - \alpha\, \log\, r_i)^2$")
    plt.xlabel(r"$\log\,K$", fontsize=14)
    plt.ylabel(r"$\alpha$", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pjoin("..", "images", "shakespeare-zipf-transformed-param-contours.png"))

def plot_transformed_scatter_points(freqs):
    ranks = np.arange(1, 1 + freqs.size)
    xs = np.log(ranks)
    ys = np.log(freqs)

    plt.figure(figsize=(10,10))
    plt.scatter(xs, ys)
    plt.title(r"$\log(freq)$ vs. $\log(rank)$", fontsize=20)
    plt.xlabel(r"$\log(rank)$", fontsize=18)
    plt.ylabel(r"$\log(freq)$", fontsize=18)
    plt.grid(b=True, which="major", linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which="minor", linestyle='--')    
    plt.tight_layout()
    plt.savefig(pjoin("..", "images", "shakespeare-zipf-transformed-param-scatter.png"))

def plot_ols_zipf_fit(empirical_freqs):
    plt.figure(figsize=(10,10))
    plt.title("Word Frequency vs. Rank", fontsize=14)
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Word Frequency", fontsize=14)
    plt.grid(True)
    
    K_ols, alpha_ols, _ = fit_zipf_ols(empirical_freqs)
    xs = np.asarray([i for i in range(1, empirical_freqs.size + 1)])
    plt.scatter(xs, empirical_freqs, alpha=0.9)
    plt.plot(xs, K_ols * xs ** alpha_ols, "g", linewidth=2, label=rf"$f_{{ols}}(x) = {K_ols:.2}x^{{{alpha_ols:.2}}}$")
    plt.ylim([0, 0.04])
    plt.legend()
    plt.grid(b=True, which="major", linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which="minor", linestyle='--')
    plt.tight_layout()
    plt.savefig(pjoin("..", "images", "shakespeare-zipf-fit.png"))

def fit_zipf_nlls(empirical_freqs):
    ranks = np.arange(1, empirical_freqs.size + 1)
    r = lambda x: empirical_freqs - x[0] * ranks ** x[1]
    result = least_squares(r, x0=np.zeros(2))
    K = result.x[0]
    alpha = result.x[1]
    mse = np.sqrt(2 * result.cost)

    return K, alpha, mse

def plot_nlls_fit(empirical_freqs):
    K_nlls, alpha_nlls, mse = fit_zipf_nlls(freqs)
    print(f"K: {K_nlls}, alpha: {alpha_nlls}, mse: {mse}")
    plt.figure(figsize=(10,10))
    plt.title("Word Frequency vs. Rank", fontsize=14)
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Word Frequency", fontsize=14)
    plt.grid(True)
    
    xs = np.asarray([i for i in range(1, empirical_freqs.size + 1)])
    ys = K_nlls * xs ** alpha_nlls
    plt.scatter(xs, empirical_freqs, alpha=0.9)
    plt.plot(
        xs, ys,
        "g", linewidth=2, 
        label=rf"$f_{{nlls}}(x) = {K_nlls:.2}x^{{{alpha_nlls:.2}}}$",
    )
    plt.ylim([0, 0.04])
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(pjoin("..", "images", "shakespeare-zipf-fit.png"))

def plot_gauss_newton_convergence(empirical_freqs):
    ranks = np.arange(1, empirical_freqs.size + 1)
    zipf = lambda K, alpha: K * ranks ** alpha
    from gauss_newton import gauss_newton
    iterates, costs = gauss_newton(
        f = lambda x: empirical_freqs - zipf(x[0], x[1]),
        x0 = np.random.randn(2),
        # x0 = np.asarray([0.04, -0.5]),
        J = lambda x: np.stack(
            [
                -ranks ** x[1],
                -np.log(ranks) * zipf(x[0], x[1]),
            ], axis=1),
            # max_iter=5,
    )
    plt.plot(np.arange(costs.size), costs)
    print(iterates)
    plt.show()

def plot_levenberg_marquardt_convergence(empirical_freqs):
    ranks = np.arange(1, empirical_freqs.size + 1)
    zipf = lambda K, alpha: K * ranks ** alpha
    from levenberg_marquardt import levenberg_marquardt
    iterates, costs = levenberg_marquardt(
        f = lambda x: empirical_freqs - zipf(x[0], x[1]),
        x0 = np.random.randn(2),
        # x0 = np.asarray([0.04, -0.5]),
        J = lambda x: np.stack(
            [
                -ranks ** x[1],
                -np.log(ranks) * zipf(x[0], x[1]),
            ], axis=1),
            max_iter=20,
    )
    plt.plot(np.arange(costs.size), costs)
    print(iterates)
    plt.show()

if __name__ == "__main__":
    N = 100
    most_freq_words, freqs = word_freqs(N)
    # plot_scatter_points(freqs)
    # plot_zipf_param_surface(freqs)
    # plot_zipf_param_contours(freqs)
    # plot_transformed_scatter_points(freqs)
    # plot_zipf_transformed_param_contours(freqs)
    # plot_zipf_transformed_param_surface(freqs)
    # plot_nlls_fit(freqs)
    # plot_ols_zipf_fit(freqs)
    # plot_gauss_newton_convergence(freqs)
    plot_levenberg_marquardt_convergence(freqs)