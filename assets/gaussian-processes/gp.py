from pathlib import Path

import numpy as np
from sklearn import tree
from sklearn.gaussian_process import GaussianProcessRegressor

GIF_DIR = Path("gifs")
GIF_DIR.mkdir(exist_ok=True, parents=True)


def fn(x_vals):
    return 2 * np.sin(x_vals) + 2 / 3 * np.sin(3 * x_vals) + 1 / 2 * np.sin(5 * x_vals)


def vandermonde(x_vals, degree: int = 3):
    cols = [x_vals.squeeze() ** d for d in range(degree + 1)]
    return np.stack(cols, axis=1)


def rbf_kernel(x1, x2, length_scale: float = 1.0):
    euclidean_mat = (
        np.square(np.linalg.norm(x1, axis=1))[:, np.newaxis]
        - 2 * x1 @ x2.T
        + np.square(np.linalg.norm(x2, axis=1))[np.newaxis, :]
    )
    return np.exp(-euclidean_mat / (2 * length_scale))


def gpr_mean_rbf(x_test, x_train, y_train, length_scale: float = 1.0):
    sigma_22 = rbf_kernel(x_train, x_train, length_scale)
    sigma_12 = rbf_kernel(x_test, x_train, length_scale)

    return sigma_12 @ np.linalg.solve(sigma_22, y_train)


def gaussian_1d_pdf(x, mu, variance):
    return 1 / np.sqrt(2 * np.pi * variance) * np.exp(-0.5 * np.square(x - mu) / variance)


def plot_1d_example():
    x_vals = np.linspace(-4, 4, 1000)

    fig, ax = plt.subplots()

    def plot_gaussian_1d_pdf(mu, variance):
        y_vals = gaussian_1d_pdf(x_vals, mu, variance)

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, lw=3)
        ax.set(xlabel="x", ylabel="y", title="1D Gaussian PDF")

        ax.set_ylim(0.00, 0.75)
        ax.set_xlim(-3.05, 3.05)
        ax.fill_between(x_vals, y_vals, 0, facecolor="blue")  # , alpha=0.85)
        # plt.tight_layout()
        ax.set(xlabel=r"$x$", ylabel=r"$f_X(x)$", title=rf"$\mu$={mu:.2f}, $\sigma^2$={variance:.2f}")
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    plot_params = []
    plot_params += [(i, 1.0) for i in np.linspace(0, 1.0, 10, endpoint=True)]
    plot_params += [(i, 1.0) for i in reversed(np.linspace(-0.5, 1.0, 10, endpoint=True))]
    plot_params += [(-0.5, i) for i in np.linspace(1, 2, 10, endpoint=True)]
    plot_params += [(i, 2) for i in reversed(np.linspace(0, -0.5, 5, endpoint=True))]
    plot_params += [(0, i) for i in reversed(np.linspace(0.5, 2.0, 5, endpoint=True))]
    plot_params += [(0, i) for i in reversed(np.linspace(1.0, 0.5, 10, endpoint=True))]
    imageio.mimsave(
        GIF_DIR / "1d-gaussian.gif",
        [plot_gaussian_1d_pdf(mu, sigma_sq) for mu, sigma_sq in plot_params],
        fps=5,
    )


def plot_gpr_1d():
    np.random.seed(31415)
    x_train = 5 * np.random.rand(15)
    y_train = fn(x_train)

    x_plot = np.linspace(0, 5, 200)

    fig, ax = plt.subplots()

    def plot_gpr(length_scale):
        y_vals = gpr_mean_rbf(x_plot[:, np.newaxis], x_train[:, np.newaxis], y_train, length_scale)

        fig, ax = plt.subplots()
        ax.scatter(x_train, y_train)
        ax.plot(x_plot, y_vals, lw=3)
        ax.set(xlabel="x", ylabel="y", title="Gaussian Process Regression")
        ax.set_xlim([0, 5])
        ax.set_ylim([-2, 2])
        plt.tight_layout()

        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    plot_params = []
    plot_params += list(10 ** np.linspace(-2.5, -0.5, 20, endpoint=True))
    plot_params += list(reversed(10 ** np.linspace(-2.5, -0.5, 10, endpoint=True)))
    imageio.mimsave(GIF_DIR / "1d-gpr.gif", [plot_gpr(length_scale) for length_scale in plot_params], fps=5)


def plot_splash_image():
    ...


def plot_regression_example():
    x_vals = 5 * np.random.rand(15)
    y_vals = fn(x_vals)

    x_plot = np.linspace(0, 5, 100)
    y_plot = fn(x_plot)

    A1 = vandermonde(x_vals)
    theta = np.linalg.lstsq(A1, y_vals, rcond=None)[0]
    y1 = vandermonde(x_plot) @ theta

    regression_tree = tree.DecisionTreeRegressor(max_depth=1)
    regression_tree = regression_tree.fit(x_vals[:, np.newaxis], y_vals)
    y2 = regression_tree.predict(x_plot[:, np.newaxis])

    gpr = GaussianProcessRegressor(random_state=0, n_restarts_optimizer=10).fit(x_vals[:, np.newaxis], y_vals)
    y3 = gpr.predict(x_plot[:, np.newaxis])

    # plt.figure(figsize=(8,10))
    plt.scatter(x_vals, y_vals)
    plt.plot(x_plot, y1)
    plt.plot(x_plot, y2)
    plt.plot(x_plot, y3)
    plt.xlim([0, 5])

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.show()


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("TkAgg")
    import imageio
    import matplotlib.pyplot as plt

    plot_1d_example()
    plot_gpr_1d()
    # plot_regression_example()
    # plot_3d_example()
#   plot_splash_image()
