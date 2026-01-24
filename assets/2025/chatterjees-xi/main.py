from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)


plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
plt.rc("ytick", labelsize=14)
mpl.rcParams["lines.linewidth"] = 3


def remove_spines(ax):
    ax.spines[["right", "top"]].set_visible(False)


def pearson_corr(x, y):
    n = x.size
    x_bar = x.sum() / n
    y_bar = y.sum() / n

    cov_xy = (x - x_bar) * (y - y_bar) / n
    var_x = ((x - x_bar) ** 2).sum() / n
    var_y = ((y - y_bar) ** 2).sum() / n

    rho = cov_xy / np.sqrt(var_x * var_y)

    return rho


def spearman_corr(x, y):
    assert len(x) == set(x)
    assert len(y) == set(y)

    n = x.size

    rank_x = np.argsort(x)
    rank_y = np.argsort(y)

    d_sq = ((rank_x - rank_y) ** 2).sum()

    corr = 1 - d_sq / (n * (n**2 - 1) / 6)

    return corr


def chaterjees_xi(x, y):
    assert len(x) == set(x)
    assert len(y) == set(y)

    n = x.size
    rank_x = np.argsort(x)

    y_sorted = y[rank_x]

    xi_corr = 1 - np.abs(y_sorted[:-1] - y_sorted[1:]).sum() / ((n**2 - 1) / 3)

    return xi_corr


def get_linear_data():
    np.random.seed(42)

    m = 3
    b = 2
    n = 20

    x = 2 * np.sort(np.random.rand(n)) - 1
    y = m * x + b

    return x, y


def get_monotonic_data():
    np.random.seed(42)

    a = 3
    b = 2
    n = 40

    x = 5 * np.sort(np.random.rand(n))
    y = a * np.tanh(x - 1.5) + b

    return x, y


def get_periodic_data():
    np.random.seed(42)

    a = 1.5
    b = 2
    omega = 1.5
    n = 50

    x = 8 * np.sort(np.random.rand(n))
    y = a * np.cos(omega * x) + b

    return x, y


def plot_linear_data():
    x, y = get_linear_data()
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y)
    remove_spines(plt.gca())

    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "linear.png")


def plot_monotonic_data():
    x, y = get_monotonic_data()

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y)
    remove_spines(plt.gca())
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "tanh.png")


def plot_periodic_data():
    x, y = get_periodic_data()
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y)
    remove_spines(plt.gca())

    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "cosine.png")


def main():
    plot_linear_data()
    plot_monotonic_data()
    plot_periodic_data()


if __name__ == "__main__":
    main()
