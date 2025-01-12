from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from integrate import left_riemann_points, right_riemann_points, trapezoidal_points

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True, parents=True)

plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
plt.rc("ytick", labelsize=14)
mpl.rcParams["lines.linewidth"] = 3
# mpl.rcParams["lines.markersize"] = 15


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


def plot_simple_quadrature(a=1, b=2, n=10, quad_type="left"):
    # Define the interval and number of subintervals
    xs = np.linspace(0.1, 3, 1000, endpoint=False)
    ys = 1 / xs

    # Calculate step size and partition points
    h = (b - a) / n
    match quad_type.lower():
        case "left":
            # zero out the last edge
            x_k, w_k = left_riemann_points(a, b, n)
            image_file = IMAGE_DIR / "left_riemann.png"
        case "right":
            # zero out the last edge
            x_k, w_k = right_riemann_points(a, b, n)
            image_file = IMAGE_DIR / "right_riemann.png"
        case "trapezoid":
            # only half weight on first and last point
            # w_k = h / 2 * np.array([2.0 if 1 <= i < n else 1.0 for i, _ in enumerate(x_k)])
            x_k, w_k = trapezoidal_points(a, b, n)
            image_file = IMAGE_DIR / "trapezoid.png"

    # Left-hand Riemann sum
    f_xk = 1 / x_k
    approx_integral = np.dot(w_k, f_xk)
    exact_inegral = np.log(2)
    pct_error = np.abs((approx_integral - exact_inegral) / exact_inegral) * 100
    print(f"{quad_type} 1/x: {approx_integral:.4}, exact: {exact_inegral:.4}, pct_error={(pct_error):.4}%")

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys)
    for i in range(n):
        match quad_type.lower():
            case "left":
                plt.fill_between([x_k[i], x_k[i + 1]], 0, [f_xk[i], f_xk[i]], color="blue", alpha=0.3)
            case "right":
                plt.fill_between([x_k[i], x_k[i + 1]], 0, [f_xk[i + 1], f_xk[i + 1]], color="blue", alpha=0.3)
            case "trapezoid":
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

    # Print results


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


def plot_nodes_and_weights():
    # TODO caseyh: mess with this function some. What are we tring to show exactly?
    t_k = np.linspace(-4, 4, 1000, endpoint=False)

    sinh_term = np.pi / 2 * np.sinh(t_k)
    x_k = np.tanh(sinh_term)

    w_k = np.pi / 2 * np.cosh(t_k) / (np.cosh(sinh_term) ** 2)
    # tanh_term goes identically to 1 which causes an error.
    # Really it should asymptote but it saturates so fast it becomes 1 to working precision
    f_xk = shifted_rsqrt(1 - x_k)
    y_k = f_xk * w_k

    plt.figure(figsize=(8, 8))
    plt.plot(t_k, y_k)
    remove_spines(plt.gca())
    plt.ylim([0, 2])

    h = 0.1
    t_k = np.arange(-3, 3, h)

    sinh_term = np.pi / 2 * np.sinh(t_k)
    x_k = np.tanh(sinh_term)
    w_k = np.pi / 2 * np.cosh(t_k) / (np.cosh(sinh_term) ** 2)

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.scatter(t_k, w_k)
    remove_spines(plt.gca())

    plt.subplot(132)
    plt.scatter(t_k, x_k)
    remove_spines(plt.gca())

    plt.subplot(133)
    plt.scatter(t_k, 1 / np.sqrt(1 - x_k) * w_k)
    plt.ylim([0, 2])
    remove_spines(plt.gca())

    plt.savefig(IMAGE_DIR / "tanhsinh_nodes_weights.png")


def plot_splash_image():
    # Define a cubic function
    def cubic(x):
        return x**3 - 6 * x**2 + 9 * x + 2

    # Define the range of the plot and the integration limits
    x = np.linspace(-1, 5, 500)  # Full range for the curve
    x_fill = np.linspace(1, 4, 500)  # Range for the area to be filled

    # Compute the y-values for the curve and the filled area
    y = cubic(x)
    y_fill = cubic(x_fill)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the cubic curve
    ax.plot(x, y, label=r"$f(x) = x^3 - 6x^2 + 9x + 2$", color="purple", linewidth=2)

    # Fill the area under the curve
    ax.fill_between(x_fill, y_fill, color="lightblue", alpha=0.5, label="Area Under Curve")

    # Add labels, legend, and title
    ax.grid(alpha=0.4)
    plt.tight_layout()

    make_cartesian_plane(ax)
    plt.savefig(IMAGE_DIR / "splash_image.png")


def main():
    # plot_splash_image()
    # plot_integral_with_asymptote()
    # plot_simple_quadrature(quad_type="left")
    # plot_simple_quadrature(quad_type="right")
    # plot_simple_quadrature(quad_type="trapezoid")

    from integrate import riemann_quadrature, trapezoidal_quadrature

    print(riemann_quadrature(shifted_rsqrt, side="left"))
    # print(riemann_quadrature(shifted_rsqrt, side="right"))
    # print(trapezoidal_quadrature(shifted_rsqrt))
    # approx_integral = tanh_sinh_quadrature(lambda x: 1 / x, n_points=10, h=0.2)
    # print(f"tanhsinh 1/x: {approx_integral:.4}")

    # plot_tanh_sinh()
    # plot_nodes_and_weights()
    # # Compute integrals
    # result1 = tanh_sinh_quadrature(f1, n_points=50)
    # # result2 = tanh_sinh_quadrature(f2, n_points=100)

    # print("Integral of 1/sqrt(1-x) from -1 to 1:")
    # print(f"Computed: {result1:.10f}")
    # print(f"Exact: {2 * np.sqrt(2):.10f}")


if __name__ == "__main__":
    main()
