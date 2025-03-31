import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

LOGARITHM_LOOKUP = [math.log(1 + 2.**-k) for k in range(100)]


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


def bkm(x: float, n_iters: int):
    assert n_iters < 100

    log_x = 0
    x_hat = 1
    for k in range(n_iters):
        a_k = 1 + 2 ** -k
        tmp = x_hat * a_k
        if tmp <= x:
            log_x += LOGARITHM_LOOKUP[k]
            x_hat = tmp

    return log_x


def main():
    pass


if __name__ == "__main__":
    main()
