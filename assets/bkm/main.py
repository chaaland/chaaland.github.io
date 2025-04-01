from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from bkm import log

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


def plot_bkm_vs_log():
    # plot bkm for 2, 4 and 8 iterations and compare against the true logarithm
    xs = np.linspace(1, 4.5, 500)
    for n_iters in range(2, 9):
        plt.figure(figsize=(6, 6))
        plt.plot(xs, np.log(xs), "--", label=r"$\log(x)$")
        plt.plot(xs, [log(elem, n_iters=n_iters) for elem in xs], alpha=0.7, label=rf"$n_{{iters}}$={n_iters}")

        remove_spines(plt.gca())
        plt.legend(fontsize=12)
        plt.xlim(0, 4.5)
        plt.ylim(0, 1.5)
        plt.grid(visible=True, which="major", color="k", linestyle="-", alpha=0.8)
        plt.grid(visible=True, which="minor", color="k", linestyle="-", alpha=0.2)
        plt.minorticks_on()
        make_cartesian_plane(plt.gca())
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f"bkm_niters_{n_iters}.png")


def plot_bkm_artifacts():
    # show what happens when we go outside the interval (1, 4.768)
    xs = np.linspace(0.1, 6, 500)
    n_iters = 8
    plt.figure(figsize=(6, 6))
    plt.plot(xs, np.log(xs), "--", label=r"$\log(x)$")
    plt.plot(xs, [log(elem, n_iters=n_iters) for elem in xs], alpha=0.7, label=rf"$n_{{iters}}$={n_iters}")

    plt.legend()
    plt.ylim([-1, 2])
    make_cartesian_plane(plt.gca())
    plt.grid(visible=True, which="major", color="k", linestyle="-", alpha=0.8)
    plt.grid(visible=True, which="minor", color="k", linestyle="-", alpha=0.2)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "bkm_artifacts.png")


def plot_approx_fraction():
    xs = np.arange(10)
    ys = 1 / (1 + 2.0**-xs)
    ys_approx = 1 - 2.0**-xs

    plt.figure()
    plt.scatter(xs, ys, label=r"$\left(1+2^{-k}\right)^{-1}$", alpha=0.5)
    plt.scatter(xs, ys_approx, label=r"$1 - 2^{-k}$", alpha=0.5)

    remove_spines(plt.gca())
    plt.legend(frameon=False, fontsize=16)
    plt.savefig(IMAGE_DIR / "division_approx.png")


def main():
    plot_bkm_vs_log()
    plot_bkm_artifacts()
    plot_approx_fraction()


if __name__ == "__main__":
    main()
