import marimo

__generated_with = "0.13.15"
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


app._unparsable_cell(
    r"""
    ![](newton-gregory/public/Screenshot 2025-07-12 at 00.13.11.png)def make_cartesian_plane(ax):
        ax.spines[\"top\"].set_color(\"none\")
        ax.spines[\"bottom\"].set_position(\"zero\")
        ax.spines[\"left\"].set_position(\"zero\")
        ax.spines[\"right\"].set_color(\"none\")


    def remove_spines(ax):
        ax.spines[[\"right\", \"top\"]].set_visible(False)
    """,
    name="_"
)


@app.cell
def _(make_cartesian_plane, mo, np, plt):
    xs = np.linspace(-0.9, 3, 100)
    ys = np.log1p(xs)

    n_pts = 6
    h = 0.5
    x_0 = 0
    x_pts = x_0 + h * np.arange(n_pts)  # np.array([1, 1.5, 2, 2.5])
    y_pts = np.log1p(x_pts)

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys)
    plt.scatter(x_pts, y_pts, alpha=0.5)
    make_cartesian_plane(plt.gca())
    mo.mpl.interactive(plt.gca())
    return h, n_pts, x_0, x_pts, xs, y_pts, ys


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
    coeffs = compute_newton_gregory_coeffs(y_pts[:n_points_slider.value], h)
    result = np.zeros_like(ys)

    polynomial_label = []
    for k, a in enumerate(coeffs):
        val = np.ones_like(xs)
        if a == 0:
            continue
        
        monomial_label = [f"{a:.4f}"]
        for ell in range(k):
            val *= (xs - x_pts[ell])
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
    plt.scatter(x_pts[:n_points_slider.value], y_pts[:n_points_slider.value], alpha=0.5)
    plt.legend(loc="lower right", frameon=False)
    make_cartesian_plane(plt.gca())
    plt.ylim([-2., 2])

    fig = plt.gcf()
    return (fig,)


@app.cell
def _(fig, mo, n_points_slider):
    mo.hstack([fig, mo.md(f"n_pts: {n_points_slider}")])
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(0.01, 0.5, 0.01)
    return (slider,)


@app.cell
def _(make_cartesian_plane, newton_gregory, np, plt, slider, x_0, xs, ys):
    _x_pts = x_0 + slider.value * np.arange(4)  # np.array([1, 1.5, 2, 2.5])
    _y_pts = np.log1p(_x_pts)

    plt.figure(figsize=(6,6))
    plt.plot(xs, ys, label=r"$\log(x)$", linewidth=2)
    plt.plot(xs, xs - xs**2 / 2 + xs**3 / 3, label=f"Taylor Polynomial degree {len(_y_pts)-1}", linewidth=2)

    _result = newton_gregory(xs, x_0=x_0, ys=_y_pts, h=slider.value)

    plt.plot(xs, _result, "--", label=f"Newton-Gregory (n={len(_y_pts)}, h={slider.value})", linewidth=2)
    plt.scatter(_x_pts, _y_pts, alpha=0.5)
    plt.ylim([-2.5, 1.5])
    make_cartesian_plane(plt.gca())

    plt.legend(frameon=False)

    taylor_fig = plt.gcf()
    return (taylor_fig,)


@app.cell
def _(mo, slider, taylor_fig):
    mo.hstack([taylor_fig, mo.md(f"h: {slider}")])

    return


if __name__ == "__main__":
    app.run()
