import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    return mo, np, plt


@app.cell
def _(mo, np):
    x_slider = mo.ui.slider(-2, 2, 0.1)
    y_slider = mo.ui.slider(-2, 2, 0.1)
    theta_slider = mo.ui.slider(0, 2 * np.pi, 0.1)
    return theta_slider, x_slider, y_slider


@app.cell
def _(mo, theta_slider, x_slider, y_slider):
    mo.vstack(
        [
            mo.md(f"x_center: {x_slider}"),
            mo.md(f"y_center: {y_slider}"),
            mo.md(rf"$\theta$: {theta_slider}"),
        ]
    )
    return


@app.cell
def _(np, theta_slider, x_slider, y_slider):
    a, b = 1, 4
    D = np.diag(np.array([a**-0.5, b**-0.5]))
    V = np.array(
        [
            [np.cos(theta_slider.value), -np.sin(theta_slider.value)],
            [np.sin(theta_slider.value), np.cos(theta_slider.value)],
        ]
    )
    center = np.array([x_slider.value, y_slider.value])
    return a, b, center


@app.cell
def _(np):
    def generate_ellipse_grid(
        r_low: float,
        r_high: float,
        n_r: int,
        n_theta: int = 50,
        a: float = 1,
        b: float = 1,
        angle: float = 0,
        center: np.ndarray = np.zeros((2, 1)),
    ):
        r = np.linspace(r_low, r_high, n_r)
        theta = np.linspace(0, 2 * np.pi, n_theta)

        r_mesh, theta_mesh = np.meshgrid(r, theta)
        x_mesh = a * (r_mesh * np.cos(theta_mesh))
        y_mesh = b * (r_mesh * np.sin(theta_mesh))

        xy_stacked = np.hstack([x_mesh.reshape((-1, 1)), y_mesh.reshape((-1, 1))]).T

        rot_mat = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        # print(r_mesh.shape)
        plot_grid = rot_mat @ xy_stacked + center.reshape((2, 1))
        X = plot_grid[0, :].reshape(x_mesh.shape)
        Y = plot_grid[1, :].reshape(y_mesh.shape)

        return X, Y


    def batch_quad_form(x: np.ndarray, A: np.ndarray):
        if A.ndim != 2:
            raise ValueError(f"Expected `A` to be a 2d array, got {A.ndim}")

        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Expected `A` to be a square array, got {A.shape}")

        n, _ = A.shape
        if x.shape[0] != n:
            raise ValueError(f"Expected first dimension of `x` to be {n}, got {x.shape[0]}")

        partial_quad = A @ x  # n x m

        return np.sum(x * partial_quad, axis=0)
    return (generate_ellipse_grid,)


@app.cell
def _(a, b, center, generate_ellipse_grid, plt, theta_slider):
    X, Y = generate_ellipse_grid(
        r_low=0.0001, r_high=5, n_r=1000, n_theta=500, a=a, b=b, angle=theta_slider.value, center=center
    )
    plt.scatter(X.ravel(), Y.ravel())
    # Z = batch_quad_form(np.stack([X.ravel(), Y.ravel()], axis=0), V @ D @ V.T).reshape(X.shape)

    # plt.contour(X, Y, Z, levels=50)
    # plt.tight_layout()
    # plt.xlim([-10, 10])
    # plt.ylim([-10, 10])
    # plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
