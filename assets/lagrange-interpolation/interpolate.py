import functools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True, parents=True)

plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
plt.rc("ytick", labelsize=14)
mpl.rcParams["lines.linewidth"] = 3
mpl.rcParams["lines.markersize"] = 10


def remove_spines(ax):
    ax.spines[["right", "top"]].set_visible(False)


def get_example_points() -> list[tuple[int, int]]:
    x_1 = 1
    x_2 = 2
    x_3 = 4

    a_1 = 7
    a_2 = -1
    a_3 = 3.5

    return [(x_1, a_1), (x_2, a_2), (x_3, a_3)]


def plot_example_points():
    pts = get_example_points()
    x_1, a_1 = pts[0]
    x_2, a_2 = pts[1]
    x_3, a_3 = pts[2]

    plt.figure(figsize=(10, 10))
    plt.scatter(x_1, a_1, color="tab:orange")
    plt.scatter(x_2, a_2, color="tab:green")
    plt.scatter(x_3, a_3, color="tab:red")
    remove_spines(plt.gca())

    plt.xlim([0, 5])
    plt.ylim([-5, 20])
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "points.png")


def plot_single_point_interpolators():
    pts = get_example_points()
    x_1, a_1 = pts[0]
    x_2, a_2 = pts[1]
    x_3, a_3 = pts[2]

    xs = np.linspace(0, 5, 100)

    l_1 = a_1 * (xs - x_2) * (xs - x_3) / ((x_1 - x_2) * (x_1 - x_3))
    l_2 = a_2 * (xs - x_1) * (xs - x_3) / ((x_2 - x_1) * (x_2 - x_3))
    l_3 = a_3 * (xs - x_1) * (xs - x_2) / ((x_3 - x_1) * (x_3 - x_2))

    plt.figure(figsize=(10, 10))
    plt.scatter(x_1, a_1, color="tab:orange")
    plt.scatter(x_2, a_2, color="tab:green")
    plt.scatter(x_3, a_3, color="tab:red")

    plt.plot(xs, l_1, "--", color="tab:orange")
    plt.plot(xs, l_2, "--", color="tab:green")
    plt.plot(xs, l_3, "--", color="tab:red")

    plt.xlim([0, 5])
    plt.ylim([-5, 20])

    remove_spines(plt.gca())

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "one_point_interpolation.png")


def plot_two_point_interpolators():
    pts = get_example_points()
    x_1, a_1 = pts[0]
    x_2, a_2 = pts[1]
    x_3, a_3 = pts[2]

    xs = np.linspace(0, 5, 100)

    l_1 = a_1 * (xs - x_2) * (xs - x_3) / ((x_1 - x_2) * (x_1 - x_3))
    l_2 = a_2 * (xs - x_1) * (xs - x_3) / ((x_2 - x_1) * (x_2 - x_3))
    l_3 = a_3 * (xs - x_1) * (xs - x_2) / ((x_3 - x_1) * (x_3 - x_2))

    plt.figure(figsize=(10, 10))
    plt.scatter(x_1, a_1, color="tab:orange")
    plt.scatter(x_2, a_2, color="tab:green")
    plt.scatter(x_3, a_3, color="tab:red")

    plt.plot(xs, l_1 + l_2, "--", zorder=0, color="tab:blue")
    plt.plot(xs, l_2 + l_3, "--", zorder=0, color="tab:purple")
    plt.plot(xs, l_1 + l_3, "--", zorder=0, color="tab:gray")

    remove_spines(plt.gca())
    plt.ylim([-5, 20])
    plt.title("2 point quadratic interpolators", fontsize=14)

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "two_point_interpolation.png")


def plot_three_point_interpolators():
    pts = get_example_points()
    x_1, a_1 = pts[0]
    x_2, a_2 = pts[1]
    x_3, a_3 = pts[2]

    xs = np.linspace(0, 5, 100)

    l_1 = a_1 * (xs - x_2) * (xs - x_3) / ((x_1 - x_2) * (x_1 - x_3))
    l_2 = a_2 * (xs - x_1) * (xs - x_3) / ((x_2 - x_1) * (x_2 - x_3))
    l_3 = a_3 * (xs - x_1) * (xs - x_2) / ((x_3 - x_1) * (x_3 - x_2))

    plt.figure(figsize=(10, 10))
    plt.scatter(x_1, a_1, color="tab:orange")
    plt.scatter(x_2, a_2, color="tab:green")
    plt.scatter(x_3, a_3, color="tab:red")

    plt.plot(xs, l_1 + l_2 + l_3, zorder=0)

    remove_spines(plt.gca())
    plt.ylim([-5, 20])
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "three_point_interpolation.png")


def lagrange_interpolate(x, nodes: np.ndarray, vals: np.ndarray):
    assert len(nodes) == len(vals)

    n = len(nodes)
    result = 0
    for i in range(n):
        a_i = vals[i]
        x_i = nodes[i]
        num = 1
        denom = 1
        for j in range(n):
            x_j = nodes[j]

            if i != j:
                num *= x - x_j
                denom *= x_i - x_j

        ell_i = a_i * num / denom
        result += ell_i

    return result


def linear_interpolate_logarithm():
    nodes = np.arange(1, 2, 0.2)
    vals = np.log(nodes)  # true logarithm values (pretend they're from a table somewhere)

    lagrange_intepolate_fn = functools.partial(lagrange_interpolate, nodes=nodes[1:3], vals=vals[1:3])
    xs = np.linspace(0, 3, 1000)
    plt.figure(figsize=(10, 10))

    # plt.plot(xs, np.log(xs), label=r"$y_{true}$", zorder=0)
    plt.scatter(nodes, vals, alpha=0.5)
    plt.scatter(nodes[1:3], vals[1:3], c="tab:blue")
    plt.scatter(1.35, lagrange_intepolate_fn([1.35]), marker="*", s=300, zorder=1)
    plt.plot(xs, lagrange_intepolate_fn(xs), "--", label=r"$y_{linear}$", zorder=0)
    plt.xlim([0, 3])
    plt.ylim([-1, 1])
    ax = plt.gca()
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")
    plt.legend(frameon=False, fontsize=20)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "logarithm_lerp.png")


def estimate_logarithm():
    nodes = np.arange(1, 2, 0.2)
    vals = np.log(nodes)  # true logarithm values (pretend they're from a table somewhere)

    xs = np.linspace(0.1, 3, 1000)

    lagrange_intepolate_fn = functools.partial(lagrange_interpolate, nodes=nodes, vals=vals)

    plt.figure(figsize=(10, 10))

    plt.scatter(nodes, vals)
    plt.plot(xs, np.log(xs), label=r"$y_{true}$")
    plt.plot(xs, lagrange_intepolate_fn(xs), "--", label=r"$y_{lagrange}$")
    plt.xlim([0, 3])
    plt.ylim([-1, 1])
    ax = plt.gca()
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")
    plt.legend(frameon=False, fontsize=20)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "logarithm_interpolator.png")

    x_interp = np.arange(0.8, 2, 0.05)
    y_lagrange = lagrange_intepolate_fn(x_interp)
    y_true = np.log(x_interp)

    df = pl.DataFrame({"x": x_interp, "y_true": y_true, "y_lagrange": y_lagrange})
    df = df.with_columns(
        ((pl.col("y_lagrange") - pl.col("y_true")).abs() / pl.col("y_true").abs() * 100).alias("pct_error")
    )
    with pl.Config(tbl_rows=30):
        print(df)

    plt.figure(figsize=(10, 10))
    plt.scatter(df["x"], df["pct_error"])
    remove_spines(plt.gca())
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel("% error", fontsize=14)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "logarithm_interpolation_error.png")


def main():
    plot_example_points()
    plot_single_point_interpolators()
    plot_two_point_interpolators()
    plot_three_point_interpolators()
    linear_interpolate_logarithm()
    estimate_logarithm()


if __name__ == "__main__":
    main()
