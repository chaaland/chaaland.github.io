import functools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from interpolate import lagrange_interpolate

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True, parents=True)

plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
plt.rc("ytick", labelsize=14)
mpl.rcParams["lines.linewidth"] = 3
mpl.rcParams["lines.markersize"] = 15


def make_cartesian_plane(ax):
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")


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

    plt.figure(figsize=(7, 7))
    plt.scatter(x_1, a_1, color="tab:orange")
    plt.scatter(x_2, a_2, color="tab:green")
    plt.scatter(x_3, a_3, color="tab:red")
    make_cartesian_plane(plt.gca())

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

    plt.figure(figsize=(7, 7))
    plt.scatter(x_1, a_1, color="tab:orange")
    plt.scatter(x_2, a_2, color="tab:green")
    plt.scatter(x_3, a_3, color="tab:red")

    plt.plot(xs, l_1, "--", color="tab:orange")
    plt.plot(xs, l_2, "--", color="tab:green")
    plt.plot(xs, l_3, "--", color="tab:red")

    plt.xlim([0, 5])
    plt.ylim([-5, 20])

    make_cartesian_plane(plt.gca())
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

    plt.figure(figsize=(7, 7))
    plt.scatter(x_1, a_1, color="tab:orange")
    plt.scatter(x_2, a_2, color="tab:green")
    plt.scatter(x_3, a_3, color="tab:red")

    plt.plot(xs, l_1 + l_2, "--", zorder=0, color="tab:blue")
    plt.plot(xs, l_2 + l_3, "--", zorder=0, color="tab:purple")
    plt.plot(xs, l_1 + l_3, "--", zorder=0, color="tab:gray")

    make_cartesian_plane(plt.gca())
    plt.ylim([-5, 20])

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

    plt.figure(figsize=(7, 7))
    plt.scatter(x_1, a_1, color="tab:orange")
    plt.scatter(x_2, a_2, color="tab:green")
    plt.scatter(x_3, a_3, color="tab:red")

    plt.plot(xs, l_1 + l_2 + l_3, zorder=0)

    make_cartesian_plane(plt.gca())
    plt.ylim([-5, 20])
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "three_point_interpolation.png")


def linear_interpolate_logarithm():
    nodes = np.arange(1, 2, 0.2)
    vals = np.log(nodes)  # true logarithm values (pretend they're from a table somewhere)

    pts = list(zip(nodes[1:3], vals[1:3]))
    lagrange_intepolate_fn = functools.partial(lagrange_interpolate, pts=pts)
    xs = np.linspace(0, 3, 1000)
    plt.figure(figsize=(7, 7))

    plt.plot(xs, np.log(xs), label=r"$y_{true}$", zorder=0)
    plt.scatter(nodes, vals, alpha=0.5, zorder=1)
    plt.scatter(nodes[1:3], vals[1:3], c="tab:blue", zorder=1)
    plt.scatter(1.35, lagrange_intepolate_fn([1.35]), marker="*", s=300, zorder=1, c="tab:red")
    plt.plot(xs, lagrange_intepolate_fn(xs), "--", color="tab:orange", label=r"$y_{linear}$", zorder=0)
    plt.xlim([0, 3])
    plt.ylim([-1, 1])
    make_cartesian_plane(plt.gca())
    plt.legend(frameon=False, fontsize=20)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "logarithm_lerp.png")


def estimate_logarithm():
    nodes = np.arange(1, 2, 0.2)
    vals = np.log(nodes)  # true logarithm values (pretend they're from a table somewhere)
    pts = list(zip(nodes, vals))
    xs = np.linspace(0.1, 3, 1000)

    lagrange_intepolate_fn = functools.partial(lagrange_interpolate, pts=pts)

    plt.figure(figsize=(7, 7))

    plt.scatter(nodes, vals, zorder=10)
    plt.plot(xs, np.log(xs), label=r"$y_{true}$")
    plt.plot(xs, lagrange_intepolate_fn(xs), "--", label=r"$y_{lagrange}$")
    plt.xlim([0, 3])
    plt.ylim([-1, 1])
    ax = plt.gca()
    make_cartesian_plane(ax)
    plt.legend(frameon=False, fontsize=20)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "logarithm_interpolator.png")

    x_interp = np.arange(0.9, 2, 0.05)
    y_lagrange = lagrange_intepolate_fn(x_interp)
    y_true = np.log(x_interp)

    df = pl.DataFrame({"x": x_interp, "y_true": y_true, "y_lagrange": y_lagrange})

    pct_error_expr = (pl.col("y_lagrange") - pl.col("y_true")).abs() / pl.col("y_true").abs() * 100
    df = df.with_columns(pl.when(pl.col("y_true").abs() < 1e-9).then(0).otherwise(pct_error_expr).alias("pct_error"))
    with pl.Config(tbl_rows=30):
        print(df)

    plt.figure(figsize=(8, 8))
    plt.scatter(df["x"], df["pct_error"])
    remove_spines(plt.gca())
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel("% error", fontsize=14)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "logarithm_interpolation_error.png")


def compute_quadratic_coefficients():
    pts = get_example_points()
    x_1, y_1 = pts[0]
    x_2, y_2 = pts[1]
    x_3, y_3 = pts[2]

    A = np.vander(np.array([x_1, x_2, x_3]), N=3, increasing=False)
    b = np.array([y_1, y_2, y_3])
    return np.linalg.solve(A, b)


def make_splash_image():
    # Define points to interpolate
    x_points = np.array([-2, 0, 1, 3])
    y_points = np.array([4, 1, -1, 2])

    # Define the Lagrange basis polynomials
    def lagrange_basis(x, i, x_points):
        x_i = x_points[i]
        num = np.prod([(x - x_j) for j, x_j in enumerate(x_points) if j != i], axis=0)
        denom = np.prod([(x_i - x_j) for j, x_j in enumerate(x_points) if j != i])
        return num / denom

    # Generate the Lagrange interpolation polynomial
    def lagrange_interpolation(x, x_points, y_points):
        return sum(y_i * lagrange_basis(x, i, x_points) for i, y_i in enumerate(y_points))

    # Generate x values for a smooth plot
    x_range = np.linspace(min(x_points) - 1, max(x_points) + 1, 500)
    y_interpolated = lagrange_interpolation(x_range, x_points, y_points)

    # Plot the data points
    plt.figure(figsize=(10, 6))
    plt.scatter(x_points, y_points, color="red", zorder=5, label="Data points")

    # Plot the interpolation polynomial
    plt.plot(x_range, y_interpolated, color="blue", linewidth=2.5, label="Interpolating polynomial")

    # Plot individual basis polynomials
    colors = ["green", "orange", "purple", "brown"]
    for i in range(len(x_points)):
        basis_y = y_points[i] * lagrange_basis(x_range, i, x_points)
        plt.plot(x_range, basis_y, linestyle="--", linewidth=1.5, color=colors[i], label=rf"$\ell_{i}(x)$")

    # Customize the plot
    plt.grid(True, alpha=0.3)
    make_cartesian_plane(plt.gca())
    plt.tight_layout()

    # Save and show the image
    plt.savefig(IMAGE_DIR / "splash_image.png", dpi=300)


def main():
    plot_example_points()
    coeffs = compute_quadratic_coefficients()
    print(f"{coeffs[0]:.4}x^2 + {coeffs[1]:.4}x + {coeffs[2]:.4}")
    plot_single_point_interpolators()
    plot_two_point_interpolators()
    plot_three_point_interpolators()
    linear_interpolate_logarithm()
    estimate_logarithm()
    make_splash_image()


if __name__ == "__main__":
    main()
