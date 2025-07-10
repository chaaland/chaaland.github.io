import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.linalg import solve_triangular
    from scipy.special import comb, factorial
    import math

    from interpolate import newton_gregory
    return mo, newton_gregory, np, plt


@app.cell
def _():
    def make_cartesian_plane(ax):
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")


    def remove_spines(ax):
        ax.spines[["right", "top"]].set_visible(False)
    return (make_cartesian_plane,)


@app.cell
def _(make_cartesian_plane, mo, np, plt):
    xs = np.linspace(-0.9, 3, 100)
    ys = np.log1p(xs)

    h = 0.5
    x_0 = 0
    x_pts = x_0 + h * np.arange(4)  # np.array([1, 1.5, 2, 2.5])
    y_pts = np.log1p(x_pts)

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys)
    plt.scatter(x_pts, y_pts, alpha=0.5)
    make_cartesian_plane(plt.gca())
    mo.mpl.interactive(plt.gca())
    return h, x_0, x_pts, xs, y_pts, ys


@app.cell
def _(
    h,
    make_cartesian_plane,
    mo,
    newton_gregory,
    plt,
    x_0,
    x_pts,
    xs,
    y_pts,
    ys,
):
    result = newton_gregory(xs, x_0=x_0, ys=y_pts, h=h)
    plt.plot(xs, ys, label=r"$\log(x)$")
    plt.plot(xs, result, "--", label=f"Newton-Gregory (n={len(y_pts)})")
    plt.scatter(x_pts, y_pts, alpha=0.5)
    make_cartesian_plane(plt.gca())
    mo.mpl.interactive(plt.gca())
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(0.01, 0.5, 0.01)
    return (slider,)


@app.cell
def _(make_cartesian_plane, mo, newton_gregory, np, plt, slider, x_0, xs, ys):
    _x_pts = x_0 + slider.value * np.arange(4)  # np.array([1, 1.5, 2, 2.5])
    _y_pts = np.log1p(_x_pts)

    plt.plot(xs, ys, label=r"$\log(x)$")
    plt.plot(xs, xs - xs**2 / 2 + xs**3 / 3, label=f"Taylor Polynomial degree {len(_y_pts)-1}")

    _result = newton_gregory(xs, x_0=x_0, ys=_y_pts, h=slider.value)

    plt.plot(xs, _result, "--", label=f"Newton-Gregory (n={len(_y_pts)}, h={slider.value})")
    plt.scatter(_x_pts, _y_pts, alpha=0.5)
    plt.ylim([-2.5, 1.5])
    make_cartesian_plane(plt.gca())
    plt.legend()
    # mo.vstack([mo.mpl.interactive(plt.gca()), mo.md(f"Choose a value: {slider}")])
    mo.md(f"Choose a value: {slider}"), plt.gcf()
    return


if __name__ == "__main__":
    app.run()
