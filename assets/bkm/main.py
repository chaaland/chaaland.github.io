from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from bkm import LOGARITHM_LOOKUP, exp, log

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
    plt.figure(figsize=(6, 6))
    plt.plot(xs, np.log(xs), "--", label=r"$\log(x)$")

    for n_iters in [2, 4, 8]:
        ys = [log(elem, n_iters=n_iters) for elem in xs]
        plt.plot(xs, ys, alpha=0.7, label=rf"$n_{{iters}}$={n_iters}")

    remove_spines(plt.gca())
    plt.legend(fontsize=12)
    plt.xlim(0, 4.5)
    plt.ylim(0, 1.5)
    plt.grid(visible=True, which="major", color="k", linestyle="-", alpha=0.8)
    plt.grid(visible=True, which="minor", color="k", linestyle="-", alpha=0.2)
    plt.minorticks_on()
    make_cartesian_plane(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "bkm_lmode.png")


def plot_bkm_vs_exp():
    # plot bkm for 2, 4 and 8 iterations and compare against the true logarithm
    xs = np.linspace(1, 4.5, 10000)
    plt.figure(figsize=(6, 6))
    plt.plot(xs, np.exp(xs), "--", label=r"$e^{x}$")
    for n_iters in [2, 4, 8]:
        ys = [exp(elem, n_iters=n_iters) for elem in xs]
        plt.plot(xs, ys, alpha=0.7, label=rf"$n_{{iters}}$={n_iters}")

    remove_spines(plt.gca())
    plt.legend(fontsize=12)
    plt.xlim(1, 1.5)
    plt.ylim(0, 5)
    plt.grid(visible=True, which="major", color="k", linestyle="-", alpha=0.8)
    plt.grid(visible=True, which="minor", color="k", linestyle="-", alpha=0.2)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "bkm_emode.png")


def plot_bkm_artifacts():
    # show what happens when we go outside the interval (1, ~4.768)
    xs = np.linspace(0.1, 6, 500)
    n_iters = 8
    plt.figure(figsize=(6, 6))
    plt.plot(xs, np.log(xs), "--", label=r"$\log(x)$")
    plt.plot(xs, [log(elem, n_iters=n_iters) for elem in xs], alpha=0.8, label=rf"$n_{{iters}}$={n_iters}")

    # plt.legend()
    plt.ylim([-1, 2])
    make_cartesian_plane(plt.gca())
    plt.grid(visible=True, which="major", color="k", linestyle="-", alpha=0.8)
    plt.grid(visible=True, which="minor", color="k", linestyle="-", alpha=0.2)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "bkm_log_artifacts.png")


def plot_bkm_exp_artifacts():
    # show what happens when we go outside the interval (0, ~1.562)
    xs = np.linspace(-1, 4.5, 500)
    n_iters = 8
    plt.figure(figsize=(6, 6))
    plt.plot(xs, np.exp(xs), "--", label=r"$e^{x}$")

    ys = [exp(elem, n_iters=n_iters) for elem in xs]
    plt.plot(xs, ys, alpha=0.7, label=rf"$n_{{iters}}$={n_iters}")

    remove_spines(plt.gca())
    plt.xlim(-1, 2)
    plt.ylim(0, 8)
    plt.grid(visible=True, which="major", color="k", linestyle="-", alpha=0.8)
    plt.grid(visible=True, which="minor", color="k", linestyle="-", alpha=0.2)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "bkm_exp_artifacts.png")


def plot_approx_fraction():
    xs = np.arange(10)
    ys = 1 / (1 + 2.0**-xs)
    ys_approx = 1 - 2.0**-xs

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, label=r"$\left(1+2^{-k}\right)^{-1}$", alpha=0.5)
    plt.scatter(xs, ys_approx, label=r"$1 - 2^{-k}$", alpha=0.5)

    remove_spines(plt.gca())
    plt.legend(frameon=False, fontsize=16)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "division_approx.png")


def plot_convergent_product():
    xs = np.arange(15)
    ys = 1 + 2.0**-xs

    plt.scatter(xs, np.cumprod(ys))
    plt.ylabel([0, 5])
    plt.xlabel(r"$N$", fontsize=14)
    plt.ylabel(r"$\prod_{k=0}^{N} (1+2^{-k})^{d_k}$", fontsize=14)
    remove_spines(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "cumulative_product.png")


def plot_log_pi_bkm():
    plt.figure(figsize=(6, 6))

    x = np.pi
    log_x = 0
    x_hat = 1
    n_iters = 10

    accepted = []
    rejected = []

    for k in range(n_iters):
        a_k = 1 + 2**-k
        tmp = x_hat * a_k
        if tmp <= x:
            log_x += LOGARITHM_LOOKUP[k]
            x_hat = tmp
            accepted.append((k, tmp, log_x))
        else:
            rejected.append((k, tmp, log_x + LOGARITHM_LOOKUP[k]))

    plt.scatter([elem for elem, _, _ in accepted], [elem for _, elem, _ in accepted], marker="o", color="tab:blue")
    plt.scatter([elem for elem, _, _ in rejected], [elem for _, elem, _ in rejected], marker="x", color="tab:red")
    plt.axhline(x, linestyle="--", alpha=0.5)
    plt.ylim([0, 4])
    plt.xlabel(r"$N$", fontsize=14)
    plt.ylabel(r"$\prod_{k=0}^{N} (1+2^{-k})^{d_k}$", fontsize=14)
    remove_spines(plt.gca())

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "log_pi.png")


def make_splash_image():
    plt.figure(figsize=(8, 4))
    xs = np.linspace(-1, 1, 1000)

    for a in [0.5, 1, 1.5]:
        plt.plot(xs, 1+np.exp(a * xs))

    make_cartesian_plane(plt.gca())
    plt.grid(True, which="both")
    plt.xlim([-1, 1])
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "splash_image.png")


def main():
    make_splash_image()
    # plot_convergent_product()
    # plot_bkm_vs_log()
    # plot_bkm_vs_exp()
    # plot_bkm_artifacts()
    # plot_bkm_exp_artifacts()
    # # plot_approx_fraction()
    # plot_log_pi_bkm()


if __name__ == "__main__":
    main()
