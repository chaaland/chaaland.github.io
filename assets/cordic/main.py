from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cordic import cordic_iter

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True, parents=True)

plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
plt.rc("ytick", labelsize=14)
mpl.rcParams["lines.linewidth"] = 3
mpl.rcParams["lines.markersize"] = 10


def make_cartesian_plane(ax):
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")


def remove_spines(ax):
    ax.spines[["right", "top"]].set_visible(False)


def rotation_mat(theta_val: float) -> np.array:
    return np.array(
        [
            [np.cos(theta_val), -np.sin(theta_val)],
            [np.sin(theta_val), np.cos(theta_val)],
        ]
    )


def plot_gain():
    n_steps = 20

    result = []
    total = 0
    for i in range(n_steps):
        total += np.log(1 / (1 + 2 ** (-2 * i)) ** 0.5)
        result.append(np.exp(total))

    plt.figure(figsize=(8, 8))
    plt.scatter(range(n_steps), result)
    remove_spines(plt.gca())
    plt.ylabel(r"$\prod_{i=0}^{N-1} \frac{1}{\sqrt{1+2^{-2i}}}$")
    plt.ylim([0.6, 0.75])
    plt.axhline(0.6072593500888125, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "K_gain.png")


def compare_gain_sequence():
    xs = range(10)
    y1 = [np.log(1 + 2 ** (-2 * k)) for k in xs]
    y2 = [2 ** (-2 * k) for k in xs]

    plt.figure(figsize=(8, 8))
    remove_spines(plt.gca())
    plt.scatter(xs, y1, label=r"$\log(1+2^{-2k})$", alpha=0.5)
    plt.scatter(xs, y2, label=r"$2^{-2k}$", alpha=0.5)
    plt.tight_layout()
    plt.legend(frameon=False, fontsize=14)
    plt.savefig(IMAGE_DIR / "gain_sequence.png")


def plot_angle_schedule():
    n_angles = 7
    xs = [2**-i for i in range(n_angles)]
    theta = [np.rad2deg(np.atan(elem)) for elem in xs]

    print(theta)
    plt.figure(figsize=(8, 8))
    plt.scatter(range(n_angles), theta, label=r"$\arctan(2^{-k})$", alpha=0.7)
    plt.scatter(range(n_angles), [45 / 2**k for k in range(n_angles)], label=r"$2^{-k}$", alpha=0.7)
    remove_spines(plt.gca())
    plt.ylabel(r"$\theta$ (degrees)")
    plt.ylim([0, 90])
    plt.legend(frameon=False)
    plt.savefig(IMAGE_DIR / "angles.png")


def plot_cordic_schedule():
    target_theta = np.pi / 5

    for n_steps in range(12):
        # assume first quadrant
        v = np.array([1, 0])
        theta_hat = 0.

        plt.figure(figsize=(8, 8))

        t = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(t), np.sin(t))
        plt.quiver(
            [0], [0], [np.cos(target_theta)], [np.sin(target_theta)], angles="xy", scale_units="xy", scale=1, color="r"
        )

        plt.quiver([0], [0], v[0], v[1], angles="xy", scale_units="xy", scale=1, alpha=0.7**n_steps)

        for i in range(n_steps):
            cos_theta, sin_theta = v
            print(f"[{i=}] {theta_hat=:5f}, {target_theta=:.5f}")
            if theta_hat == target_theta:
                return cos_theta, sin_theta

            ccw = theta_hat < target_theta
            if ccw:
                theta_hat += np.atan(2**-i)
            else:
                theta_hat -= np.atan(2**-i)

            v = cordic_iter(i, v, ccw, scale=True)
            plt.quiver([0], [0], v[0], v[1], angles="xy", scale_units="xy", scale=1, alpha=0.7 ** (n_steps - 1 - i))

        plt.title(rf"$\hat{{\theta}}$ = {theta_hat:.5f}, $\theta = {target_theta:.5f}$", fontsize=18)
        make_cartesian_plane(plt.gca())
        plt.xlim([0, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f"cordic_{n_steps:02}.png")


if __name__ == "__main__":
    plot_angle_schedule()
    plot_gain()
    compare_gain_sequence()
    plot_cordic_schedule()
