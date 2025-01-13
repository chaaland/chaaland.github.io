from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from integrate import left_riemann_points, right_riemann_points, trapezoidal_points

mpl.use("Agg")

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True, parents=True)

GIF_DIR = Path("gifs")
GIF_DIR.mkdir(exist_ok=True, parents=True)

plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
plt.rc("ytick", labelsize=14)
mpl.rcParams["lines.linewidth"] = 3


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


def plot_tanhsinh_substitution():
    import imageio

    # x_plot = np.concatenate([-2**-np.linspace(0, 10, 100, endpoint=False), np.linspace(0.9, 0.9999, 50)])
    # more points near 1 since that's where singularity is
    x_plot_0_to_1 = 1 - 2 ** np.linspace(-16, 0, 250, endpoint=False)[::-1]
    x_plot = np.concatenate([-x_plot_0_to_1[::-1], x_plot_0_to_1])  # from -1 to 1 now
    y_plot = shifted_rsqrt(x_plot)

    t_plot_0_to_infty = np.asinh(np.atanh(x_plot_0_to_1) / (np.pi / 2))
    t_plot = np.asinh(np.atanh(x_plot) / (np.pi / 2))
    # breakpoint()
    sinh_term = np.pi / 2 * np.sinh(t_plot)
    x_k = np.tanh(sinh_term)
    dx = np.pi / 2 * np.cosh(t_plot) / (np.cosh(sinh_term) ** 2)
    y_plot_transformed = shifted_rsqrt(x_k) * dx

    def generate_frames(theta):
        fig, ax = plt.subplots()
        # how to deform x-axis into t-axis??
        if theta == 0:
            ax.plot(x_plot, y_plot, lw=3)
        elif theta == 1:
            ax.plot(t_plot, y_plot_transformed, lw=3)
        else:
            blended_abscissa_0_to_1 = x_plot_0_to_1 ** (1 - theta) * t_plot_0_to_infty**theta
            blended_abscissa = np.concatenate([-blended_abscissa_0_to_1[::-1], blended_abscissa_0_to_1])

            colors = ["tab:red", "tab:orange", "tab:blue", "tab:green"]
            for i, c in enumerate(colors):
                n_points = len(x_plot) // len(colors)  # prob want ceil
                blended_y = y_plot ** (1 - theta) * y_plot_transformed**theta
                ax.plot(
                    blended_abscissa[i * n_points : (i + 1) * n_points],
                    blended_y[i * n_points : (i + 1) * n_points],
                    lw=3,
                    color=c,
                )
            # ax.plot(blended_abscissa, y_plot ** (1 - theta) * y_plot_transformed**theta, lw=3)

        # ax.set(xlabel="x", ylabel="y", title="Gaussian Process Regression")
        ax.set_xlim([-2, 2])
        ax.set_ylim([0, 4])
        make_cartesian_plane(ax)
        plt.tight_layout()

        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = fig.canvas.buffer_rgba()
        plt.close(fig)
        return image

    plot_params = np.arange(0, 1, 0.01 / 2).tolist()
    imageio.mimsave(GIF_DIR / "tanhsinh_transform.gif", [generate_frames(theta) for theta in plot_params], fps=25)


# def create_gif(frame_generator, filename="animation.gif", fps=10, dpi=100, title=None, **kwargs):
#     """
#     Creates a GIF from a sequence of matplotlib figures.

#     Parameters:
#         frame_generator: Iterator yielding (fig, ax) tuples for each frame
#         filename: Output GIF filename
#         fps: Frames per second in the output GIF
#         dpi: Resolution of the output GIF
#         title: Optional title for the animation
#         **kwargs: Additional keyword arguments passed to frame_generator

#     Example usage:
#     def generate_frames():
#         for i in range(10):
#             fig, ax = plt.subplots()
#             ax.plot([0, i], [0, i**2])
#             ax.set_xlim(0, 10)
#             ax.set_ylim(0, 100)
#             yield fig, ax
#             plt.close(fig)  # Important: close figure to free memory

#     create_gif(generate_frames)
#     """
#     # Create a list to store the figures
#     frames = []

#     # Get the frames from the generator
#     for fig, ax in frame_generator(**kwargs):
#         # If a title was provided, set it
#         if title:
#             ax.set_title(title)

#         # Convert figure to image array and append to frames
#         # We need to draw the canvas first to ensure the figure is rendered
#         fig.canvas.draw()
#         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
#         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         frames.append(image)

#     # Create the GIF using PIL (Python Imaging Library)
#     writer = PillowWriter(fps=fps)

#     # Create a new figure for the animation
#     fig = plt.figure()
#     plt.axis("off")  # Hide axes for the animation
#     im = plt.imshow(frames[0])

#     def update(frame):
#         im.set_array(frame)
#         return [im]

#     # Create the animation
#     anim = animation.ArtistAnimation(fig, [(im,) for frame in frames], interval=1000 / fps)

#     # Save the animation
#     anim.save(filename, writer=writer, dpi=dpi)
#     plt.close(fig)  # Clean up

#     return filename


# # Example usage with a simple sine wave animation
# def generate_tanhsinh_frames(num_frames=30):
#     """
#     Example frame generator that creates a moving sine wave.
#     """
#     t = np.linspace(0, 2 * np.pi, 100)

#     for i in range(num_frames):
#         # Create new figure for each frame
#         fig, ax = plt.subplots()

#         # Calculate phase shift for this frame
#         phase = 2 * np.pi * i / num_frames

#         # Plot the sine wave
#         ax.plot(t, np.sin(t + phase))

#         # Set consistent axes limits
#         ax.set_xlim(0, 2 * np.pi)
#         ax.set_ylim(-1.5, 1.5)

#         # Add grid for better visualization
#         ax.grid(True)

#         yield fig, ax
#         plt.close(fig)  # Clean up


def main():
    # plot_splash_image()
    # plot_integral_with_asymptote()
    # plot_simple_quadrature(quad_type="left")
    # plot_simple_quadrature(quad_type="right")
    # plot_simple_quadrature(quad_type="trapezoid")

    plot_tanhsinh_substitution()
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
