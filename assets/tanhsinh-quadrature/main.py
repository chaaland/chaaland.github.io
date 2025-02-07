from enum import Enum
from pathlib import Path

import imageio
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from integrate import left_riemann_points, riemann_quadrature, right_riemann_points, trapezoidal_points

mpl.use("Agg")

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True, parents=True)

GIF_DIR = Path("gifs")
GIF_DIR.mkdir(exist_ok=True, parents=True)

plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
plt.rc("ytick", labelsize=14)
mpl.rcParams["lines.linewidth"] = 3


class QuadratureMethod(str, Enum):
    left_riemann = "left_riemann"
    right_riemann = "right_riemann"
    trapezoid = "trapezoid"
    tanhsinh = "tanhsinh"


def shifted_rsqrt(x):
    return 1 / np.sqrt(1 - x)


def make_cartesian_plane(ax):
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")


def remove_spines(ax):
    ax.spines[["right", "top"]].set_visible(False)


def plot_integral_with_asymptote():
    xs = np.linspace(-2, 1, 1000, endpoint=False)
    ys = shifted_rsqrt(xs)

    x_fill = np.linspace(-1, 1, 1000, endpoint=False)
    y_fill = shifted_rsqrt(x_fill)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys)
    ax = plt.gca()
    ax.fill_between(x_fill, y_fill, color="lightblue", alpha=0.5, label="Area Under Curve")
    plt.axvline([1], linestyle="--", color="gray")
    plt.xlim([-2, 2])
    plt.ylim([0, 4])
    plt.title(r"$f(x) = \frac{1}{\sqrt{1-x}}$", fontsize=14)
    make_cartesian_plane(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "shifted_rsqrt.png")


def plot_area_under_curve(image_file):
    def reciprocal(x):
        return 1 / x

    xs = np.linspace(0.1, 3, 1000, endpoint=False)
    x_fill = np.linspace(1, 2, 100, endpoint=False)
    ys = reciprocal(xs)

    y_fill = reciprocal(x_fill)
    plt.figure(figsize=(8, 8))

    ax = plt.gca()
    ax.plot(xs, ys)
    ax.fill_between(x_fill, y_fill, color="lightblue", alpha=0.5, label="Area Under Curve")

    make_cartesian_plane(ax)
    plt.xlim([0, 3])
    plt.ylim([0, 1.5])
    plt.tight_layout()
    plt.savefig(image_file)


def plot_simple_quadrature(f, a=1, b=2, n=10, quad_type=QuadratureMethod.left_riemann):
    # Define the interval and number of subintervals
    xs = np.linspace(0.1, 3, 1000, endpoint=False)
    ys = f(xs)

    # Calculate step size and partition points
    h = (b - a) / n
    match quad_type:
        case QuadratureMethod.left_riemann:
            # zero out the last edge
            x_k, w_k = left_riemann_points(a, b, n)
            image_file = IMAGE_DIR / "left_riemann.png"
        case QuadratureMethod.right_riemann:
            # zero out the last edge
            x_k, w_k = right_riemann_points(a, b, n)
            image_file = IMAGE_DIR / "right_riemann.png"
        case QuadratureMethod.trapezoid:
            # only half weight on first and last point
            # w_k = h / 2 * np.array([2.0 if 1 <= i < n else 1.0 for i, _ in enumerate(x_k)])
            x_k, w_k = trapezoidal_points(a, b, n)
            image_file = IMAGE_DIR / "trapezoid.png"

    # Left-hand Riemann sum
    f_xk = f(x_k)

    is_finite_mask = np.isfinite(f_xk)
    approx_integral = np.dot(w_k[is_finite_mask], f_xk[is_finite_mask])
    exact_integral = np.log(2)
    pct_error = np.abs((approx_integral - exact_integral) / exact_integral) * 100
    print(f"{quad_type} 1/x: {approx_integral:.4}, exact: {exact_integral:.4}, pct_error={(pct_error):.4}%")

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys)
    for i in range(n):
        match quad_type:
            case QuadratureMethod.left_riemann:
                plt.fill_between([x_k[i], x_k[i + 1]], 0, [f_xk[i], f_xk[i]], color="blue", alpha=0.3)
            case QuadratureMethod.right_riemann:
                plt.fill_between([x_k[i], x_k[i + 1]], 0, [f_xk[i + 1], f_xk[i + 1]], color="blue", alpha=0.3)
            case QuadratureMethod.trapezoid:
                plt.fill_between(
                    [x_k[i], x_k[i] + h],
                    [f_xk[i], f_xk[i + 1]],
                    color="blue",
                    alpha=0.3,
                )
            case _:
                raise ValueError(f"Unrecognised quadrature type: {quad_type}")

    make_cartesian_plane(plt.gca())
    plt.xlim([0, 3])
    plt.ylim([0, 1.5])
    plt.tight_layout()
    plt.savefig(image_file)


def plot_tanh_sinh():
    # tanh-sinh saturates super quickly!
    ts = np.linspace(-5, 5, 1000, endpoint=False)
    tanh_sinh_substitution = np.tanh(np.pi / 2 * np.sinh(ts))

    plt.figure(figsize=(8, 8))
    plt.plot(ts, np.tanh(ts), label=r"$\tanh(t)$")
    plt.plot(ts, tanh_sinh_substitution, label=r"$\tanh\left(\frac{\pi}{2} \sinh(t)\right)$")
    plt.legend(frameon=False, fontsize=20)
    make_cartesian_plane(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "tanhsinh.png")


def plot_splash_image():
    # Define a cubic function
    def cubic(x):
        return x**3 - 6 * x**2 + 9 * x + 2

    x = np.linspace(-1, 5, 500)  # Full range for the curve
    x_fill = np.linspace(1, 4, 500)  # Range for the area to be filled

    y = cubic(x)
    y_fill = cubic(x_fill)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y, label=r"$f(x) = x^3 - 6x^2 + 9x + 2$", color="purple", linewidth=2)
    ax.fill_between(x_fill, y_fill, color="lightblue", alpha=0.5, label="Area Under Curve")

    # Add labels, legend, and title
    ax.grid(alpha=0.4)
    plt.tight_layout()

    make_cartesian_plane(ax)
    plt.ylim([-2, 10])
    plt.savefig(IMAGE_DIR / "splash_image.png")


def plot_tanhsinh_substitution():
    t_plot_0_to_infty = np.linspace(1e-4, 3, 100)
    t_plot = np.concatenate([-t_plot_0_to_infty[::-1], t_plot_0_to_infty])
    x_plot_0_to_1 = np.tanh(np.pi / 2 * np.sinh(t_plot_0_to_infty))
    x_plot = np.tanh(np.pi / 2 * np.sinh(t_plot))
    y_plot = shifted_rsqrt(x_plot)

    sinh_term = np.pi / 2 * np.sinh(t_plot)
    x_k = np.tanh(sinh_term)
    dx = np.pi / 2 * np.cosh(t_plot) / (np.cosh(sinh_term) ** 2)
    y_plot_transformed = shifted_rsqrt(x_k) * dx

    # Normalize x to the range [0, 1] for color mapping
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Create a colormap (rainbow)
    cmap = plt.cm.rainbow

    n_points = len(t_plot)
    colors = cmap(norm(np.linspace(0, 1, n_points)))

    def generate_frames(theta):
        fig, ax = plt.subplots()

        if theta == 0:
            for i in range(n_points - 1):
                ax.plot(x_plot[i : i + 2], y_plot[i : i + 2], color=colors[i], linewidth=3)

            ax.set_xlim([-3, 3])
            ax.set_ylim([0, 8])
            make_cartesian_plane(ax)
            fig.savefig(IMAGE_DIR / "integrand.png")
        elif theta == 1:
            for i in range(n_points - 1):
                ax.plot(t_plot[i : i + 2], y_plot_transformed[i : i + 2], color=colors[i], linewidth=3)

            ax.set_xlim([-3, 3])
            ax.set_ylim([0, 8])
            make_cartesian_plane(ax)
            fig.savefig(IMAGE_DIR / "integrand_tanhsinh.png")
        else:
            blended_abscissa_0_to_1 = x_plot_0_to_1 ** (1 - theta) * t_plot_0_to_infty**theta
            blended_abscissa = np.concatenate([-blended_abscissa_0_to_1[::-1], blended_abscissa_0_to_1])
            blended_y = y_plot ** (1 - theta) * y_plot_transformed**theta

            # Generate colors for the line
            for i in range(n_points - 1):
                ax.plot(blended_abscissa[i : i + 2], blended_y[i : i + 2], color=colors[i], linewidth=3)

        ax.set_xlim([-3, 3])
        ax.set_ylim([0, 8])
        make_cartesian_plane(ax)
        plt.tight_layout()

        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = fig.canvas.buffer_rgba()
        plt.close(fig)
        return image

    plot_params = np.linspace(0, 0.1, 50).tolist()
    plot_params += np.linspace(0.1, 1, 100).tolist()

    imageio.mimsave(GIF_DIR / "tanhsinh_transform.gif", [generate_frames(theta) for theta in plot_params], fps=25)


def plot_tanhsinh_quadrature():
    # Define the interval and number of subintervals
    ts = np.linspace(-3, 3, 1000, endpoint=True)

    sinh_t = np.sinh(ts)
    cosh_t = np.cosh(ts)
    tanh_term = np.tanh(np.pi / 2 * sinh_t)
    cosh_term = np.cosh(np.pi / 2 * sinh_t)

    # Compute the new integrand post-substitution
    ys = 1 / np.sqrt(1 - tanh_term) * (np.pi / 2 * (cosh_t / (cosh_term * cosh_term)))

    # Calculate step size and partition points
    h = 0.3
    n = 10
    a = -h * n
    b = h * n
    t_k, w_k = left_riemann_points(a, b, 2 * n)

    # Left-hand Riemann sum
    sinh_t = np.sinh(t_k)
    cosh_t = np.cosh(t_k)
    tanh_term = np.tanh(np.pi / 2 * sinh_t)
    cosh_term = np.cosh(np.pi / 2 * sinh_t)
    f_xk = 1 / np.sqrt(1 - tanh_term) * (np.pi / 2 * cosh_t / (cosh_term * cosh_term))
    approx_integral = np.dot(w_k, f_xk)
    exact_integral = 2 * 2**0.5
    pct_error = np.abs((approx_integral - exact_integral) / exact_integral) * 100
    print(f"tanhsinh 1/sqrt(1-x): {approx_integral:.6}, exact: {exact_integral:.6}, pct_error={(pct_error):.6}%")

    plt.figure(figsize=(8, 8))
    plt.plot(ts, ys)
    for i in range(2 * n):
        plt.fill_between([t_k[i], t_k[i + 1]], 0, [f_xk[i], f_xk[i]], color="blue", alpha=0.3)

    make_cartesian_plane(plt.gca())
    plt.xlim([-3, 3])
    plt.ylim([0, 2])
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "tanhsinh_riemann.png")


def plot_sigmoid_substitution():
    t_plot = np.linspace(-10, 10, 100)

    sigmoid_term = 1 / (1 + np.exp(-t_plot))
    dx = sigmoid_term * (1 - sigmoid_term)
    x_k = sigmoid_term
    y_plot_transformed = shifted_rsqrt(x_k) * dx
    # Normalize x to the range [0, 1] for color mapping
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Create a colormap (rainbow)
    cmap = plt.cm.rainbow

    n_points = len(t_plot)
    colors = cmap(norm(np.linspace(0, 1, n_points)))
    plt.figure(figsize=(8, 8))
    for i in range(n_points - 1):
        plt.plot(t_plot[i : i + 2], y_plot_transformed[i : i + 2], color=colors[i], linewidth=3)

    make_cartesian_plane(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "integrand_sigmoid.png")


def compute_errors():
    import polars as pl
    from integrate import tanh_sinh_quadrature

    approx_integrals = []
    widths = [0.3, 0.15, 0.05, 0.01]
    exact_integral = 2**1.5
    for h in widths:
        # n * h = 3
        n = int(3 / h)
        approx_integrals.append(tanh_sinh_quadrature(lambda x: 1 / (1 - x) ** 0.5, n=n, h=h))
        print(f"{h=}, {n=}")
        # pct_error = np.abs(exact_integral - approx_integral) / exact_integral * 100
        # print(f"tanhsinh 1/sqrt(1-x): {approx_integral:.6}, actual 1/sqrt(1-x): {:.6}, pct_error: {pct_error:.6}")

    pct_error_expr = (pl.col("exact") - pl.col("approx")).abs() / pl.col("exact").abs() * 100
    df = pl.DataFrame({"width": widths, "exact": exact_integral, "approx": approx_integrals}).with_columns(
        pct_error_expr.alias("pct_error")
    )
    print(df)


def main():
    # misc images
    plot_splash_image()
    plot_area_under_curve(IMAGE_DIR / "auc.png")
    # plot_tanh_sinh()

    # # work with 1 / x
    # plot_simple_quadrature(lambda x: 1 / x, a=1, b=2, quad_type=QuadratureMethod.left_riemann)
    # plot_simple_quadrature(lambda x: 1 / x, a=1, b=2, quad_type=QuadratureMethod.right_riemann)
    # plot_simple_quadrature(lambda x: 1 / x, a=1, b=2, quad_type=QuadratureMethod.trapezoid)

    # # work with 1/ sqrt(1-x)
    # plot_integral_with_asymptote()
    # plot_tanhsinh_substitution()
    # plot_sigmoid_substitution()
    # plot_tanhsinh_quadrature()

    riemann_approx = riemann_quadrature(shifted_rsqrt, a=-1, b=1, n=20, side="left")
    print(f"{riemann_approx=:.6}")

    compute_errors()


if __name__ == "__main__":
    main()
