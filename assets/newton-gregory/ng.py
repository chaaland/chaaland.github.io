import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    from interpolate import newton_gregory, compute_newton_gregory_coeffs
    return compute_newton_gregory_coeffs, mo, newton_gregory, np, plt


@app.cell
def _(mo):
    mo.md(r"""$$y = y_0 + u \Delta y_1 + {x(x-x_0-h) \over 2h^2} \Delta^2 y_2 + {x(x-x_0-h)(x-x_0-2h) \over 3!h^3} \Delta^3 y_3 + \cdots  $$""")
    return


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
def _(mo):
    x_slider = mo.ui.slider(0.1, 1, 0.01)
    return (x_slider,)


@app.cell
def _(make_cartesian_plane, newton_gregory, np, plt, x_slider):
    xs = np.linspace(-0.2, 1.0, 100)
    ys = np.log1p(xs)

    n_pts = 10
    h = 0.1
    x_0 = 0
    x_pts = x_0 + h * np.arange(n_pts)  # np.array([1, 1.5, 2, 2.5])
    y_pts = np.log1p(x_pts)

    plt.plot(xs, ys)
    plt.scatter(x_pts, y_pts, alpha=0.5)

    _idx = sorted(np.argsort(np.abs(x_pts - x_slider.value))[:2])
    lerp_ys = newton_gregory(xs, x_0=x_pts[_idx[0]], ys=y_pts[_idx], h=h)
    plt.plot(xs, lerp_ys, "--")
    plt.scatter(x_pts[_idx], y_pts[_idx], zorder=10)

    y_slider_interp_value = newton_gregory(np.array([x_slider.value]), x_0=x_pts[_idx[0]], ys=y_pts[_idx], h=h)
    plt.scatter(np.array([x_slider.value]), y_slider_interp_value, marker="*", s=150, zorder=20)

    plt.ylim([-0.25, 1])

    make_cartesian_plane(plt.gca())
    log_fig = plt.gcf()
    return h, log_fig, n_pts, x_0, x_pts, xs, y_pts, ys


@app.cell
def _(mo):
    mo.md(r"""Logarithm LERP approxmation""")
    return


@app.cell
def _(log_fig, mo, x_slider):
    mo.hstack([log_fig, mo.md(f"x: {x_slider}")])
    return


@app.cell
def _(mo, n_pts):
    n_points_slider = mo.ui.slider(2, n_pts, 1)
    return (n_points_slider,)


@app.cell
def _(
    compute_newton_gregory_coeffs,
    h,
    make_cartesian_plane,
    n_points_slider,
    np,
    plt,
    x_pts,
    xs,
    y_pts,
    ys,
):
    coeffs = compute_newton_gregory_coeffs(y_pts[: n_points_slider.value], h)
    result = np.zeros_like(ys)

    polynomial_label = []
    for k, a in enumerate(coeffs):
        val = np.ones_like(xs)
        if a == 0:
            continue

        monomial_label = [f"{a:.4f}"]
        for ell in range(k):
            val *= xs - x_pts[ell]
            if x_pts[ell] == 0:
                monomial_label.append("x")
            else:
                monomial_label.append(f"(x-{x_pts[ell]})")
        polynomial_label.append("".join(monomial_label))
        result += a * val

    plt.figure(figsize=(7, 7))
    plt.plot(xs, ys, label=r"$\log(x)$", linewidth=2)

    poly_str = "\n+".join(polynomial_label)
    poly_str = poly_str.replace("+-", "-")

    plt.plot(xs, result, "--", label=poly_str, linewidth=2)
    plt.scatter(x_pts[: n_points_slider.value], y_pts[: n_points_slider.value], alpha=0.5)
    plt.legend(loc="lower right", frameon=False)
    make_cartesian_plane(plt.gca())
    plt.ylim([-2.0, 2])

    fig = plt.gcf()
    return (fig,)


@app.cell
def _(fig, mo, n_points_slider):
    mo.hstack([fig, mo.md(f"n_pts: {n_points_slider}")])
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(0.01, 0.5, 0.01)
    n_pts_slider = mo.ui.slider(2, 5, 1)
    return n_pts_slider, slider


@app.cell
def _(
    make_cartesian_plane,
    n_pts_slider,
    newton_gregory,
    np,
    plt,
    slider,
    x_0,
    xs,
    ys,
):
    _x_pts = x_0 + slider.value * np.arange(n_pts_slider.value)  # np.array([1, 1.5, 2, 2.5])
    _y_pts = np.log1p(_x_pts)

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, label=r"$\log(x)$", linewidth=2)

    _taylor_approx = np.zeros_like(xs)
    a_i = -1
    for i in range(1, n_pts_slider.value):
        a_i *= -xs
        _taylor_approx += a_i / i

    plt.plot(xs, _taylor_approx, label=f"Taylor Approx degree {len(_y_pts) - 1}", linewidth=2)

    _result = newton_gregory(xs, x_0=x_0, ys=_y_pts, h=slider.value)

    plt.plot(xs, _result, "--", label=f"Newton-Gregory (n={len(_y_pts)}, h={slider.value})", linewidth=2)
    plt.scatter(_x_pts, _y_pts, alpha=0.5)
    plt.ylim([-2.5, 1.5])
    make_cartesian_plane(plt.gca())

    plt.legend(frameon=False, loc="lower right")

    taylor_fig = plt.gcf()
    return (taylor_fig,)


@app.cell
def _(mo, n_pts_slider, slider, taylor_fig):
    mo.hstack([taylor_fig, mo.vstack([mo.md(f"h: {slider}"), mo.md(f"N: {n_pts_slider}")])])
    return


@app.cell
def _(mo):
    n_pts_slider_2 = mo.ui.slider(2, 6, 1)
    return (n_pts_slider_2,)


@app.cell
def _(
    compute_newton_gregory_coeffs,
    h,
    make_cartesian_plane,
    mo,
    n_pts_slider_2,
    np,
    plt,
    x_0,
    y_pts,
):
    _h = 0.1
    _x_pts = x_0 + _h * np.arange(n_pts_slider_2.value)
    _y_pts = np.log1p(_x_pts)

    A = np.vander(_x_pts, N=n_pts_slider_2.value, increasing=True)
    vander_coeffs = np.linalg.solve(A, _y_pts)

    ng_coeffs = compute_newton_gregory_coeffs(y_pts[: n_pts_slider_2.value], h)

    make_cartesian_plane(plt.gca())
    plt.scatter(np.arange(n_pts_slider_2.value), vander_coeffs, label="Vandermonde")
    plt.scatter(np.arange(n_pts_slider_2.value), ng_coeffs, label="Newton-Gregory")
    plt.xlim([0, 6])
    plt.ylim([-1, 1])

    plt.legend()

    mo.hstack([plt.gcf(), mo.md(f"N={n_pts_slider_2}")])
    return


if __name__ == "__main__":
    app.run()
