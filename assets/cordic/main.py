import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from cordic import cordic_iter, hyperbolic_cordic_iter

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
    n_steps = 15

    result = []
    total = 0
    for i in range(n_steps):
        total += math.log(1 / (1 + 2 ** (-2 * i)) ** 0.5)
        result.append(np.exp(total))

    plt.figure(figsize=(8, 8))
    plt.scatter(range(n_steps), result)
    remove_spines(plt.gca())
    plt.ylabel(r"$\prod_{i=0}^{N-1} \frac{1}{\sqrt{1+2^{-2i}}}$")
    plt.ylim([0.6, 0.75])
    plt.axhline(0.6072593500888125, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "circular_gain.png")

    result = []
    total = 0
    for i in range(n_steps):
        total += math.log(1 / (1 - 2 ** (-2 * (i + 1))) ** 0.5)
        result.append(math.exp(total))

    plt.figure(figsize=(8, 8))
    plt.scatter(range(n_steps), result)
    remove_spines(plt.gca())
    plt.ylabel(r"$\prod_{i=1}^{N} \frac{1}{\sqrt{1-2^{-2i}}}$")
    plt.ylim([1, 1.25])
    plt.axhline(1.20513636, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "hyperbolic_gain.png")


def compare_gain_sequence():
    xs = range(10)
    y1 = [math.log(1 + 2 ** (-2 * k)) for k in xs]
    y2 = [2 ** (-2 * k) for k in xs]

    plt.figure(figsize=(8, 8))
    remove_spines(plt.gca())
    plt.scatter(xs, y1, label=r"$\log(1+2^{-2k})$", alpha=0.5)
    plt.scatter(xs, y2, label=r"$2^{-2k}$", alpha=0.5)
    plt.tight_layout()
    plt.legend(frameon=False, fontsize=16)
    plt.savefig(IMAGE_DIR / "gain_sequence_circular.png")


def plot_angle_schedule():
    n_angles = 7
    xs = [2**-i for i in range(n_angles)]
    theta = [np.rad2deg(math.atan(elem)) for elem in xs]

    print(theta)
    plt.figure(figsize=(8, 8))
    plt.scatter(range(n_angles), theta, label=r"$\arctan(2^{-k})$", alpha=0.7)
    plt.scatter(range(n_angles), [45 / 2**k for k in range(n_angles)], label=r"$2^{-k}$", alpha=0.7)
    remove_spines(plt.gca())
    plt.ylabel(r"$\theta$ (degrees)")
    plt.ylim([0, 50])
    plt.legend(frameon=False, fontsize=16)

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "angles.png")


def plot_cordic_schedule():
    target_theta = math.pi / 5

    for n_steps in range(12):
        # assume first quadrant
        v = [1, 0]
        theta_hat = 0.0

        plt.figure(figsize=(8, 8))

        t = np.linspace(0, math.pi / 2, 100)
        plt.plot(np.cos(t), np.sin(t))
        plt.quiver(
            [0], [0], [np.cos(target_theta)], [np.sin(target_theta)], angles="xy", scale_units="xy", scale=1, color="r"
        )

        plt.quiver([0], [0], v[0], v[1], angles="xy", scale_units="xy", scale=1, alpha=0.7**n_steps)

        for i in range(n_steps):
            print(f"[{i=}] {theta_hat=:5f}, {target_theta=:.5f}")
            if theta_hat == target_theta:
                cos_theta, sin_theta = v
                return cos_theta, sin_theta

            ccw = theta_hat < target_theta
            delta_theta = math.atan(2**-i)
            if ccw:
                theta_hat += delta_theta
            else:
                theta_hat -= delta_theta

            v = cordic_iter(i, v, ccw, scale=True)
            plt.quiver([0], [0], v[0], v[1], angles="xy", scale_units="xy", scale=1, alpha=0.7 ** (n_steps - 1 - i))

        plt.title(rf"$\hat{{\theta}}$ = {theta_hat:.5f}, $\theta = {target_theta:.5f}$", fontsize=18)
        make_cartesian_plane(plt.gca())
        plt.xlim([0, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f"cordic_{n_steps:02}.png")


def plot_hyperbolic_cordic_schedule():
    target_theta = math.pi / 3.5

    for n_steps in range(8):
        # assume first quadrant
        v = [1, 0]
        theta_hat = 0.0

        plt.figure(figsize=(8, 8))

        t = np.linspace(-math.pi / 2, math.pi / 2, 100)
        plt.plot(np.cosh(t), np.sinh(t))
        plt.quiver(
            [0],
            [0],
            [np.cosh(target_theta)],
            [np.sinh(target_theta)],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="r",
        )

        plt.quiver([0], [0], v[0], v[1], angles="xy", scale_units="xy", scale=1, alpha=0.7**n_steps)

        for i in range(n_steps):
            print(f"[{i=}] {theta_hat=:5f}, {target_theta=:.5f}")
            if theta_hat == target_theta:
                cosh_theta, sinh_theta = v
                return cosh_theta, sinh_theta

            ccw = theta_hat < target_theta
            delta_theta = np.atanh(2 ** -(i + 1))
            if ccw:
                theta_hat += delta_theta
            else:
                theta_hat -= delta_theta

            v = hyperbolic_cordic_iter(i, v, ccw, scale=True)
            plt.quiver([0], [0], v[0], v[1], angles="xy", scale_units="xy", scale=1, alpha=0.5 ** (n_steps - 1 - i))

        plt.title(rf"$\hat{{\theta}}$ = {theta_hat:.5f}, $\theta = {target_theta:.5f}$", fontsize=18)
        make_cartesian_plane(plt.gca())
        plt.xlim([-1, 3])
        plt.ylim([-2, 2])
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f"hyperbolic_cordic_{n_steps:02}.png")

        print(f"{np.exp(target_theta)}")
        print(f"{np.sum(v)}")


def plot_circle(radius=1):
    ts = np.linspace(0, math.tau, 100)
    plt.plot(radius * np.cos(ts), radius * np.sin(ts), "k", alpha=0.5)


def plot_circular_angles():
    angles = [0, 0.5, 1, 1.5]

    for i, theta in enumerate(angles):
        plt.figure(figsize=(8, 8))
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plot_circle()
        plt.plot(
            np.cos(np.linspace(0, 2 * np.pi, 1000)), np.sin(np.linspace(0, 2 * np.pi, 1000)), label=r"$x^2 + y^2 = 1$"
        )

        x_fill = np.linspace(0, 1, 1000, endpoint=False)
        y_1 = np.tan(theta) * x_fill
        y_2 = np.sqrt(1 - x_fill**2)

        y_fill = np.minimum(y_1, y_2)

        plt.fill_between(
            x_fill, y_fill, color="blue", alpha=0.2, label=rf"$\theta$ = {theta:.1f}, Area = {theta / 2:.2f}"
        )
        plt.quiver([0], [0], [np.cos(theta)], [np.sin(theta)], angles="xy", scale_units="xy", scale=1)
        make_cartesian_plane(plt.gca())

        plt.legend(loc="upper right", frameon=False, fontsize=14)
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f"circular_angle_{i:02d}.png")


def plot_hyperbola_area():
    theta = 1.0

    plt.figure(figsize=(8, 8))
    plt.xlim([-1, 3])
    plt.ylim([-2, 2])

    t = np.linspace(-5, 5, 100)
    xs = np.cosh(t)
    ys = np.sinh(t)
    plt.plot(xs, ys, "tab:blue", label=r"$x^2 - y^2 = 1$")

    make_cartesian_plane(plt.gca())

    x_fill = np.linspace(0, np.cosh(theta), 1000, endpoint=False)
    y_1 = np.sinh(theta) * np.linspace(0, 1, 1000)
    y_2 = [0 if elem < 1 else np.sinh(np.acosh(elem)) for elem in x_fill]
    plt.fill_between(x_fill, y_1, y_2, color="blue", alpha=0.2)
    plt.quiver([0], [0], [np.cosh(theta)], [np.sinh(theta)], angles="xy", scale_units="xy", scale=1, zorder=10)

    x_fill = np.linspace(1, np.cosh(theta), 1000, endpoint=False)
    y_1 = (x_fill**2 - 1) ** 0.5
    plt.fill_between(x_fill, y_1, color="red", alpha=0.2)
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "area_hyperbola.png")


def plot_hyperbolic_angles():
    angles = [0, 0.5, 1, 1.5]

    for i, theta in enumerate(angles):
        t = np.linspace(-5, 5, 100)
        xs = np.cosh(t)
        ys = np.sinh(t)

        plt.figure(figsize=(8, 8))
        plt.xlim([-1, 5])
        plt.ylim([-3, 3])

        plt.plot(xs, ys, "tab:blue", label=r"$x^2 - y^2 = 1$")

        x_fill = np.linspace(0, np.cosh(theta), 1000, endpoint=False)
        y_1 = np.sinh(theta) * np.linspace(0, 1, 1000)
        y_2 = [0 if elem < 1 else np.sinh(np.acosh(elem)) for elem in x_fill]

        plt.fill_between(
            x_fill, y_1, y_2, color="blue", alpha=0.2, label=rf"$\theta$ = {theta:.1f}" + f"\nArea = {theta / 2:.2f}"
        )
        plt.quiver([0], [0], [np.cosh(theta)], [np.sinh(theta)], angles="xy", scale_units="xy", scale=1, zorder=10)
        plt.legend(loc="upper right", frameon=False, fontsize=14)
        make_cartesian_plane(plt.gca())

        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f"hyperbolic_angle_{i:02d}.png")


def plot_alternate_hyperbolic_angles():
    plt.figure(figsize=(8, 8))
    plt.xlim([-1, 3])
    plt.ylim([-1, 3])

    xs = np.linspace(0.1, 5, 1000)
    ys = 1 / xs
    plt.plot(xs, ys, "tab:blue")

    make_cartesian_plane(plt.gca())

    x_fill = np.linspace(0, 1, 1000, endpoint=False)
    y_1 = np.linspace(0, 1, 1000)
    y_2 = np.minimum(np.linspace(0, 4, 1000), 1 / x_fill)
    plt.fill_between(x_fill, y_1, y_2, color="blue", alpha=0.2)
    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "alt_hyperbolic_angles.png")
    # plt.quiver([0], [0], [1], [1], angles="xy", scale_units="xy", scale=1)
    # plt.quiver([0], [0], [0.5], [2], angles="xy", scale_units="xy", scale=1)

    # plt.legend(loc="upper right", frameon=False)


def plot_hyperbola(horizontal=True):
    ts = np.linspace(-5, 5, 100)

    if horizontal:
        plt.plot(np.cosh(ts), np.sinh(ts), "k", alpha=0.5)
        plt.plot(-np.cosh(ts), np.sinh(ts), "k", alpha=0.5)
    else:
        plt.plot(np.sinh(ts), np.cosh(ts), "k", alpha=0.5)
        plt.plot(np.sinh(ts), -np.cosh(ts), "k", alpha=0.5)


def plot_box_rotations():
    thetas = [0, 0.2, 0.4, 0.6, 0.8, 1]

    for k, theta in enumerate(thetas):
        circular_rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        hyperbolic_rot = np.array([[np.cosh(theta), np.sinh(theta)], [np.sinh(theta), np.cosh(theta)]])

        v1 = [1, 0]
        v2 = [0, 1]
        v3 = [-1, 0]
        v4 = [0, -1]
        points = [v1, v2, v3, v4]

        plt.figure(figsize=(8, 4))

        plt.subplot(121)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plot_circle(radius=1)

        rotated_points = []
        for v in points:
            rotated_points.append(circular_rot @ v)

        for i, p in enumerate(rotated_points):
            p_next = rotated_points[(i + 1) % 4]
            plt.plot([p[0], p_next[0]], [p[1], p_next[1]])

        make_cartesian_plane(plt.gca())

        plt.subplot(122)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plot_hyperbola()

        rotated_points = []
        for v in points:
            rotated_points.append(hyperbolic_rot @ v)

        for i, p in enumerate(rotated_points):
            p_next = rotated_points[(i + 1) % 4]
            plt.plot([p[0], p_next[0]], [p[1], p_next[1]])

        make_cartesian_plane(plt.gca())

        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f"rotations_{k:03d}.png")


def plot_circle_rotations():
    thetas = [0, 0.2, 0.4, 0.6, 0.8, 1]

    for k, theta in enumerate(thetas):
        circular_rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        hyperbolic_rot = np.array([[np.cosh(theta), np.sinh(theta)], [np.sinh(theta), np.cosh(theta)]])

        plt.figure(figsize=(8, 4))
        # Normalize x to the range [0, 1] for color mapping
        norm = mcolors.Normalize(vmin=0, vmax=1)

        # Create a colormap (rainbow)
        cmap = plt.cm.rainbow

        plt.subplot(121)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])

        rotated_points = [
            circular_rot @ np.array([np.cos(elem), np.sin(elem)]) for elem in np.linspace(0, math.tau, 1000)
        ]
        n_points = len(rotated_points)
        colors = cmap(norm(np.linspace(0, 1, n_points)))
        xs = [x for x, _ in rotated_points]
        ys = [y for _, y in rotated_points]
        for i in range(n_points - 1):
            plt.gca().plot(xs[i : i + 2], ys[i : i + 2], color=colors[i], linewidth=3)

        make_cartesian_plane(plt.gca())

        plt.subplot(122)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plot_hyperbola()

        rotated_points = [
            hyperbolic_rot @ np.array([np.cos(elem), np.sin(elem)]) for elem in np.linspace(0, math.tau, 1000)
        ]

        xs = [x for x, _ in rotated_points]
        ys = [y for _, y in rotated_points]
        for i in range(n_points - 1):
            plt.gca().plot(xs[i : i + 2], ys[i : i + 2], color=colors[i], linewidth=3)

        make_cartesian_plane(plt.gca())

        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f"circle_rotations_{k:03d}.png")


def make_splash_image():
    ts = np.linspace(0, 6 * math.pi, 1000)
    ys_1 = np.cos(ts)
    ys_2 = np.sin(ts)

    plt.figure(figsize=(8, 8))
    plt.plot(ts, ys_1)
    plt.plot(ts, ys_2)
    make_cartesian_plane(plt.gca())
    plt.grid(True, which="both")
    plt.ylim([-5, 5])
    plt.savefig(IMAGE_DIR / "splash_image.png")


if __name__ == "__main__":
    plot_angle_schedule()
    plot_gain()
    compare_gain_sequence()
    plot_hyperbola_area()
    plot_hyperbolic_angles()
    plot_alternate_hyperbolic_angles()
    plot_circular_angles()
    plot_box_rotations()
    plot_circle_rotations()

    plot_cordic_schedule()
    plot_hyperbolic_cordic_schedule()
    make_splash_image()
