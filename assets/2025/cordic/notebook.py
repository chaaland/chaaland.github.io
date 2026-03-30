import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import math
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from cordic import cordic_iter, hyperbolic_cordic_iter

    return cordic_iter, hyperbolic_cordic_iter, math, mcolors, mo, mpl, np, plt


@app.cell
def _(mpl, np, plt):
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)
    mpl.rcParams["lines.linewidth"] = 3
    mpl.rcParams["lines.markersize"] = 10

    def make_cartesian_plane(ax):
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")

    def remove_spines(ax):
        ax.spines[["right", "top"]].set_visible(False)

    def rotation_mat(theta_val: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(theta_val), -np.sin(theta_val)],
                [np.sin(theta_val), np.cos(theta_val)],
            ]
        )

    return make_cartesian_plane, remove_spines


@app.cell
def _(math, np, plt, remove_spines):
    _n_steps = 15
    _result = []
    _total = 0
    for _i in range(_n_steps):
        _total += math.log(1 / (1 + 2 ** (-2 * _i)) ** 0.5)
        _result.append(np.exp(_total))

    plt.figure(figsize=(5, 5))
    plt.scatter(range(_n_steps), _result)
    remove_spines(plt.gca())
    plt.ylabel(r"$\prod_{i=0}^{N-1} \frac{1}{\sqrt{1+2^{-2i}}}$")
    plt.ylim([0.6, 0.75])
    plt.axhline(0.6072593500888125, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(math, plt, remove_spines):
    _n_steps = 15
    _result = []
    _total = 0
    for _i in range(_n_steps):
        _total += math.log(1 / (1 - 2 ** (-2 * (_i + 1))) ** 0.5)
        _result.append(math.exp(_total))

    plt.figure(figsize=(5, 5))
    plt.scatter(range(_n_steps), _result)
    remove_spines(plt.gca())
    plt.ylabel(r"$\prod_{i=1}^{N} \frac{1}{\sqrt{1-2^{-2i}}}$")
    plt.ylim([1, 1.25])
    plt.axhline(1.20513636, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(math, plt, remove_spines):
    _xs = range(10)
    _y1 = [math.log(1 + 2 ** (-2 * k)) for k in _xs]
    _y2 = [2 ** (-2 * k) for k in _xs]

    plt.figure(figsize=(5, 5))
    remove_spines(plt.gca())
    plt.scatter(_xs, _y1, label=r"$\log(1+2^{-2k})$", alpha=0.5)
    plt.scatter(_xs, _y2, label=r"$2^{-2k}$", alpha=0.5)
    plt.legend(frameon=False, fontsize=16)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(math, np, plt, remove_spines):
    _n_angles = 7
    _xs = [2**-i for i in range(_n_angles)]
    _theta = [np.rad2deg(math.atan(elem)) for elem in _xs]

    plt.figure(figsize=(5, 5))
    plt.scatter(range(_n_angles), _theta, label=r"$\arctan(2^{-k})$", alpha=0.7)
    plt.scatter(range(_n_angles), [45 / 2**k for k in range(_n_angles)], label=r"$2^{-k}$", alpha=0.7)
    remove_spines(plt.gca())
    plt.ylabel(r"$\theta$ (degrees)")
    plt.ylim([0, 50])
    plt.legend(frameon=False, fontsize=16)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    cordic_steps_slider = mo.ui.slider(0, 11, value=6, label="CORDIC steps")
    cordic_steps_slider
    return (cordic_steps_slider,)


@app.cell
def _(
    cordic_iter,
    cordic_steps_slider,
    make_cartesian_plane,
    math,
    mo,
    np,
    plt,
):
    _target_theta = math.pi / 5
    _n_steps = cordic_steps_slider.value

    _v = [1, 0]
    _theta_hat = 0.0

    plt.figure(figsize=(5, 5))

    _t = np.linspace(0, math.pi / 2, 100)
    plt.plot(np.cos(_t), np.sin(_t))
    plt.quiver(
        [0], [0], [np.cos(_target_theta)], [np.sin(_target_theta)],
        angles="xy", scale_units="xy", scale=1, color="r",
    )
    plt.quiver([0], [0], _v[0], _v[1], angles="xy", scale_units="xy", scale=1, alpha=0.7**_n_steps)

    for _i in range(_n_steps):
        _ccw = _theta_hat < _target_theta
        _delta_theta = math.atan(2**-_i)
        if _ccw:
            _theta_hat += _delta_theta
        else:
            _theta_hat -= _delta_theta
        _v = cordic_iter(_i, _v, _ccw, scale=True)
        plt.quiver(
            [0], [0], _v[0], _v[1],
            angles="xy", scale_units="xy", scale=1, alpha=0.7 ** (_n_steps - 1 - _i),
        )

    plt.title(rf"$\hat{{\theta}}$ = {_theta_hat:.5f}, $\theta = {_target_theta:.5f}$", fontsize=18)
    make_cartesian_plane(plt.gca())
    plt.xlim([0, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo):
    hyp_steps_slider = mo.ui.slider(0, 7, value=4, label="Hyperbolic CORDIC steps")
    hyp_steps_slider
    return (hyp_steps_slider,)


@app.cell
def _(
    hyp_steps_slider,
    hyperbolic_cordic_iter,
    make_cartesian_plane,
    math,
    mo,
    np,
    plt,
):
    _target_theta = math.pi / 3.5
    _n_steps = hyp_steps_slider.value

    _v = [1, 0]
    _theta_hat = 0.0

    plt.figure(figsize=(5, 5))

    _t = np.linspace(-math.pi / 2, math.pi / 2, 100)
    plt.plot(np.cosh(_t), np.sinh(_t))
    plt.quiver(
        [0], [0], [np.cosh(_target_theta)], [np.sinh(_target_theta)],
        angles="xy", scale_units="xy", scale=1, color="r",
    )
    plt.quiver([0], [0], _v[0], _v[1], angles="xy", scale_units="xy", scale=1, alpha=0.7**_n_steps)

    for _i in range(_n_steps):
        _ccw = _theta_hat < _target_theta
        _delta_theta = np.atanh(2 ** -(_i + 1))
        if _ccw:
            _theta_hat += _delta_theta
        else:
            _theta_hat -= _delta_theta
        _v = hyperbolic_cordic_iter(_i, _v, _ccw, scale=True)
        plt.quiver(
            [0], [0], _v[0], _v[1],
            angles="xy", scale_units="xy", scale=1, alpha=0.5 ** (_n_steps - 1 - _i),
        )

    plt.title(rf"$\hat{{\theta}}$ = {_theta_hat:.5f}, $\theta = {_target_theta:.5f}$", fontsize=18)
    make_cartesian_plane(plt.gca())
    plt.xlim([-1, 3])
    plt.ylim([-2, 2])
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo):
    circular_angle_slider = mo.ui.slider(0, 3, value=1, label="Circular angle index")
    circular_angle_slider
    return (circular_angle_slider,)


@app.cell
def _(circular_angle_slider, make_cartesian_plane, np, plt):
    _angles = [0, 0.5, 1, 1.5]
    _theta = _angles[circular_angle_slider.value]

    plt.figure(figsize=(5, 5))
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.plot(
        np.cos(np.linspace(0, 2 * np.pi, 1000)),
        np.sin(np.linspace(0, 2 * np.pi, 1000)),
        label=r"$x^2 + y^2 = 1$",
    )

    _x_fill = np.linspace(0, 1, 1000, endpoint=False)
    _y_1 = np.tan(_theta) * _x_fill
    _y_2 = np.sqrt(1 - _x_fill**2)
    _y_fill = np.minimum(_y_1, _y_2)

    plt.fill_between(
        _x_fill, _y_fill, color="blue", alpha=0.2,
        label=rf"$\theta$ = {_theta:.1f}, Area = {_theta / 2:.2f}",
    )
    plt.quiver([0], [0], [np.cos(_theta)], [np.sin(_theta)], angles="xy", scale_units="xy", scale=1)
    make_cartesian_plane(plt.gca())
    plt.legend(loc="upper right", frameon=False, fontsize=14)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(make_cartesian_plane, np, plt):
    _theta = 1.0

    plt.figure(figsize=(5, 5))
    plt.xlim([-1, 3])
    plt.ylim([-2, 2])

    _t = np.linspace(-5, 5, 100)
    plt.plot(np.cosh(_t), np.sinh(_t), "tab:blue", label=r"$x^2 - y^2 = 1$")
    make_cartesian_plane(plt.gca())

    _x_fill = np.linspace(0, np.cosh(_theta), 1000, endpoint=False)
    _y_1 = np.sinh(_theta) * np.linspace(0, 1, 1000)
    _y_2 = [0 if elem < 1 else np.sinh(np.acosh(elem)) for elem in _x_fill]
    plt.fill_between(_x_fill, _y_1, _y_2, color="blue", alpha=0.2)
    plt.quiver([0], [0], [np.cosh(_theta)], [np.sinh(_theta)], angles="xy", scale_units="xy", scale=1, zorder=10)

    _x_fill2 = np.linspace(1, np.cosh(_theta), 1000, endpoint=False)
    _y_1b = (_x_fill2**2 - 1) ** 0.5
    plt.fill_between(_x_fill2, _y_1b, color="red", alpha=0.2)
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    hyp_angle_slider = mo.ui.slider(0, 3, value=1, label="Hyperbolic angle index")
    hyp_angle_slider
    return (hyp_angle_slider,)


@app.cell
def _(hyp_angle_slider, make_cartesian_plane, np, plt):
    _angles = [0, 0.5, 1, 1.5]
    _theta = _angles[hyp_angle_slider.value]

    _t = np.linspace(-5, 5, 100)

    plt.figure(figsize=(5, 5))
    plt.xlim([-1, 5])
    plt.ylim([-3, 3])
    plt.plot(np.cosh(_t), np.sinh(_t), "tab:blue", label=r"$x^2 - y^2 = 1$")

    _x_fill = np.linspace(0, np.cosh(_theta), 1000, endpoint=False)
    _y_1 = np.sinh(_theta) * np.linspace(0, 1, 1000)
    _y_2 = [0 if elem < 1 else np.sinh(np.acosh(elem)) for elem in _x_fill]
    plt.fill_between(
        _x_fill, _y_1, _y_2, color="blue", alpha=0.2,
        label=rf"$\theta$ = {_theta:.1f}" + f"\nArea = {_theta / 2:.2f}",
    )
    plt.quiver([0], [0], [np.cosh(_theta)], [np.sinh(_theta)], angles="xy", scale_units="xy", scale=1, zorder=10)
    plt.legend(loc="upper right", frameon=False, fontsize=14)
    make_cartesian_plane(plt.gca())
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(make_cartesian_plane, np, plt):
    plt.figure(figsize=(5, 5))
    plt.xlim([-1, 3])
    plt.ylim([-1, 3])

    _xs = np.linspace(0.1, 5, 1000)
    _ys = 1 / _xs
    plt.plot(_xs, _ys, "tab:blue")
    make_cartesian_plane(plt.gca())

    _x_fill = np.linspace(0, 1, 1000, endpoint=False)
    _y_1 = np.linspace(0, 1, 1000)
    _y_2 = np.minimum(np.linspace(0, 4, 1000), 1 / _x_fill)
    plt.fill_between(_x_fill, _y_1, _y_2, color="blue", alpha=0.2)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    rotation_slider = mo.ui.slider(0, 5, value=0, label="Rotation index")
    rotation_slider
    return (rotation_slider,)


@app.cell
def _(make_cartesian_plane, math, np, plt, rotation_slider):
    _thetas = [0, 0.2, 0.4, 0.6, 0.8, 1]
    _theta = _thetas[rotation_slider.value]

    _circular_rot = np.array([[np.cos(_theta), -np.sin(_theta)], [np.sin(_theta), np.cos(_theta)]])
    _hyperbolic_rot = np.array([[np.cosh(_theta), np.sinh(_theta)], [np.sinh(_theta), np.cosh(_theta)]])
    _points = [[1, 0], [0, 1], [-1, 0], [0, -1]]

    def _plot_circle_local(radius=1):
        _ts = np.linspace(0, math.tau, 100)
        plt.plot(radius * np.cos(_ts), radius * np.sin(_ts), "k", alpha=0.5)

    def _plot_hyperbola_local():
        _ts = np.linspace(-5, 5, 100)
        plt.plot(np.cosh(_ts), np.sinh(_ts), "k", alpha=0.5)
        plt.plot(-np.cosh(_ts), np.sinh(_ts), "k", alpha=0.5)

    plt.figure(figsize=(5, 2.5))

    plt.subplot(121)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    _plot_circle_local(radius=1)
    _rotated = [_circular_rot @ np.array(_p) for _p in _points]
    for _i_r, _p_r in enumerate(_rotated):
        _p_next = _rotated[(_i_r + 1) % 4]
        plt.plot([_p_r[0], _p_next[0]], [_p_r[1], _p_next[1]])
    make_cartesian_plane(plt.gca())

    plt.subplot(122)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    _plot_hyperbola_local()
    _rotated_h = [_hyperbolic_rot @ np.array(_p) for _p in _points]
    for _i_r, _p_r in enumerate(_rotated_h):
        _p_next = _rotated_h[(_i_r + 1) % 4]
        plt.plot([_p_r[0], _p_next[0]], [_p_r[1], _p_next[1]])
    make_cartesian_plane(plt.gca())

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(make_cartesian_plane, math, mcolors, np, plt, rotation_slider):
    _thetas = [0, 0.2, 0.4, 0.6, 0.8, 1]
    _theta = _thetas[rotation_slider.value]

    _circular_rot = np.array([[np.cos(_theta), -np.sin(_theta)], [np.sin(_theta), np.cos(_theta)]])
    _hyperbolic_rot = np.array([[np.cosh(_theta), np.sinh(_theta)], [np.sinh(_theta), np.cosh(_theta)]])

    _norm = mcolors.Normalize(vmin=0, vmax=1)
    _cmap = plt.cm.rainbow

    def _plot_hyperbola_local():
        _ts = np.linspace(-5, 5, 100)
        plt.plot(np.cosh(_ts), np.sinh(_ts), "k", alpha=0.5)
        plt.plot(-np.cosh(_ts), np.sinh(_ts), "k", alpha=0.5)

    plt.figure(figsize=(5, 2.5))

    plt.subplot(121)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    _rotated_circ = [_circular_rot @ np.array([np.cos(e), np.sin(e)]) for e in np.linspace(0, math.tau, 1000)]
    _n_pts = len(_rotated_circ)
    _colors = _cmap(_norm(np.linspace(0, 1, _n_pts)))
    _xs_c = [x for x, _ in _rotated_circ]
    _ys_c = [y for _, y in _rotated_circ]
    for _i_c in range(_n_pts - 1):
        plt.gca().plot(_xs_c[_i_c:_i_c + 2], _ys_c[_i_c:_i_c + 2], color=_colors[_i_c], linewidth=3)
    make_cartesian_plane(plt.gca())

    plt.subplot(122)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    _plot_hyperbola_local()
    _rotated_hyp = [_hyperbolic_rot @ np.array([np.cos(e), np.sin(e)]) for e in np.linspace(0, math.tau, 1000)]
    _xs_h = [x for x, _ in _rotated_hyp]
    _ys_h = [y for _, y in _rotated_hyp]
    for _i_h in range(_n_pts - 1):
        plt.gca().plot(_xs_h[_i_h:_i_h + 2], _ys_h[_i_h:_i_h + 2], color=_colors[_i_h], linewidth=3)
    make_cartesian_plane(plt.gca())

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(make_cartesian_plane, math, np, plt):
    _ts = np.linspace(0, 6 * math.pi, 1000)

    plt.figure(figsize=(5, 5))
    plt.plot(_ts, np.cos(_ts))
    plt.plot(_ts, np.sin(_ts))
    make_cartesian_plane(plt.gca())
    plt.grid(True, which="both")
    plt.ylim([-5, 5])
    plt.gcf()
    return


@app.cell
def _(math, np, plt, remove_spines):
    from cordic import cordic

    _xs = np.linspace(0, math.pi / 2, 500)

    plt.figure(figsize=(10, 5))
    for _n_iters in [5, 6, 7, 10]:
        _sin_approx = np.array([cordic(x, n_iters=_n_iters)[1] for x in _xs])
        _err = np.abs(_sin_approx - np.sin(_xs))
        plt.plot(_xs, _err, label=rf"$n={_n_iters}$")

    remove_spines(plt.gca())
    plt.xlabel(r"$x$")
    plt.ylabel(r"$|\sin(x) - \text{CORDIC } \sin(x)|$")
    plt.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
