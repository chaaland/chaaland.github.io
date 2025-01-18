from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from aaa import aaa, simple_ols
from scipy.special import gamma

mpl.use("Agg")

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True, parents=True)

plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
plt.rc("ytick", labelsize=14)
mpl.rcParams["lines.linewidth"] = 3


def make_cartesian_plane(ax):
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")


def remove_spines(ax):
    ax.spines[["right", "top"]].set_visible(False)


def plot_gamma():
    xs = np.concatenate([np.linspace(i, i + 1, 500, endpoint=False) for i in range(-5, 5)])
    ys = gamma(xs)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys)
    plt.xlim([-5, 5])
    plt.ylim([-6, 6])
    plt.title(r"$\Gamma(x)$", fontsize=14)
    make_cartesian_plane(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "gamma_function.png")


def plot_simple_ols(f, degree: int, filename: str):
    xs = np.linspace(0.01, 5, 1000)
    ys = f(xs)

    a, b = simple_ols(f, xs, m=degree)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys)

    A = np.vander(xs, degree + 1, increasing=True)
    numerator = A @ a
    denominator = A[:, 1:] @ b
    rational = numerator / (1 + denominator)

    plt.plot(xs, rational, "--")
    plt.xlim([0, 5])
    plt.ylim([-2, 2])
    make_cartesian_plane(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / filename)

    return a, b


def plot_ols_gamma(f, degree: int, filename: str):
    xs = np.concatenate([np.linspace(i + 0.001, i + 1, 500, endpoint=False) for i in range(-5, 5)])
    ys = gamma(xs)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys)

    a, b = simple_ols(gamma, xs, m=degree)
    A = np.vander(xs, degree + 1, increasing=True)
    numerator = A @ a
    denominator = A[:, 1:] @ b
    rational = numerator / (1 + denominator)

    plt.plot(xs, rational, "--")
    plt.xlim([-5, 5])
    plt.ylim([-6, 6])
    make_cartesian_plane(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / filename)

    print(a, b)
    return a, b


def plot_aaa_gamma(f, degree: int, filename: str):
    xs = np.concatenate([np.linspace(i + 0.001, i + 1, 500, endpoint=False) for i in range(-5, 5)])
    ys = gamma(xs)

    a, b = simple_ols(gamma, xs, m=degree)
    aaa(gamma, xs, max_degree=5)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys)


def main():
    # plot_gamma()
    # for i in range(1, 4):
    #     plot_simple_ols(np.log, degree=i, filename=f"ols_logarithm_degree_{i}")

    # for i in range(1, 20):
    #     plot_ols_gamma(gamma, degree=i, filename=f"ols_gamma_degree_{i:02}")

    plot_aaa_gamma(gamma, degree=2, filename=f"aaa_gamma_degree_{2:02}")


if __name__ == "__main__":
    main()
