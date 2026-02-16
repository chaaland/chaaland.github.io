import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    import functools
    import polars as pl
    import marimo as mo

    IMAGE_DIR = Path("images")
    IMAGE_DIR.mkdir(exist_ok=True)
    return IMAGE_DIR, functools, np, plt


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
def _(make_cartesian_plane, np, plt):
    def absolute_deviation(x, y, beta):
        return np.mean(np.abs(beta * x - y))


    np.random.seed(31)
    n_points = 7
    x = np.random.randint(size=n_points, low=1, high=25)
    y = np.random.randint(size=n_points, low=-25, high=25)
    knots = y / x
    beta = np.linspace(-2.5, 2.5, 1000)

    absolute_deviations = np.array([absolute_deviation(x, y, beta_i) for beta_i in beta])

    for _i in range(n_points):
        plt.figure()

        plt.plot(beta, np.mean(np.abs(np.outer(beta, x[: _i + 1]) - y[: _i + 1]), axis=1), color="k")

        for _j in range(_i + 1):
            if _i == 0:
                break
            plt.plot(beta, np.abs(x[_j] * beta - y[_j]), alpha=0.5)

        make_cartesian_plane(plt.gca())
        plt.ylim([0, 50])
        plt.tight_layout()
        plt.savefig(f"1d-abs-deviation-{_i:02}.png")

    # plt.plot(beta, absolute_deviations, linewidth=2)
    # plt.ylim([0, 40])
    plt.show()
    # plt.scatter(knots, np.array([absolute_deviation(x, y, beta_i) for beta_i in knots]), alpha=0.5)
    return absolute_deviation, absolute_deviations, beta, knots, x, y


@app.cell
def _(absolute_deviation, beta, knots, make_cartesian_plane, np, plt, x, y):
    np.random.seed(31)

    absolute_deviations = np.array([absolute_deviation(x, y, beta_i) for beta_i in beta])

    plt.plot(beta, absolute_deviations, linewidth=2)
    plt.ylim([0, 40])
    make_cartesian_plane(plt.gca())
    plt.scatter(knots, np.array([absolute_deviation(x, y, beta_i) for beta_i in knots]), alpha=0.5)
    return (absolute_deviations,)


@app.function
def golden_section_minimize(obj_fn, a, b, n_iters: int = 5):
    assert b > a
    assert n_iters >= 0

    L = b - a
    alpha = (-1 + 5**0.5) / 2
    x1 = b - alpha * L
    x2 = a + alpha * L

    assert a < x1 < x2 < b

    f_a = obj_fn(beta=a)
    f_x1 = obj_fn(beta=x1)
    f_x2 = obj_fn(beta=x2)
    f_b = obj_fn(beta=b)

    for i in range(n_iters):
        if f_x1 < f_x2:
            b, f_b = x2, f_x2
            L = b - a

            x2, f_x2 = x1, f_x1
            
            x1 = b - alpha * L
            f_x1 = obj_fn(beta=x1)
        else:
            a, f_a = x1, f_x1
            L = b - a

            x1, f_x1 = x2, f_x2
            x2 = a + alpha * L

            f_x2 = obj_fn(beta=x2)

        assert a < x1 < x2 < b, f"{a}, {x1}, {x2}, {b}, {i}"

    return [a, x1, x2, b]


@app.cell
def _(
    absolute_deviation,
    absolute_deviations,
    beta,
    functools,
    make_cartesian_plane,
    np,
    plt,
    x,
    y,
):
    obj_fn = functools.partial(absolute_deviation, x=x, y=y)


    def minimize_lad(x, y):
        knots = y / x

        f_min = np.inf
        beta_star = np.inf
        for knot in knots:
            f_val = absolute_deviation(beta=knot, x=x, y=y)
            if f_val < f_min:
                f_min = f_val
                beta_star = knot
        return beta_star


    def plot_golden_section_minimize_algo(knots, n_iters: int = 2):
        fig, ax = plt.subplots()
        a, x1, x2, b = golden_section_minimize(obj_fn, a=min(knots), b=max(knots), n_iters=n_iters)
        make_cartesian_plane(plt.gca())

        plt.plot(beta, absolute_deviations, linewidth=2, color="k")
        plt.ylim([0, 50])
        plt.scatter(knots, np.array([absolute_deviation(x, y, beta_i) for beta_i in knots]), alpha=0.5, color='k')
        plt.title(f"{n_iters=}")
        _ax = plt.gca()
        _ax.vlines(
            [a, x1, x2, b], ymin=0, ymax=50, linestyle="--", colors=["tab:red", "tab:blue", "tab:green", "tab:orange"]
        )

        # Add labels underneath the x-axis
        label_y = -3  # Position below ymin=0
        labels = ["a", "x₁", "x₂", "b"]
        colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
        for pos, label, color in zip([a, x1, x2, b], labels, colors):
            _ax.text(pos, label_y, label, ha="center", va="top", fontsize=14, color=color)

        plt.tight_layout() 

        return fig, ax
    return (plot_golden_section_minimize_algo,)


@app.cell
def _(IMAGE_DIR, knots, plot_golden_section_minimize_algo, plt):
    for _i in range(4):
        _fig, _ax = plot_golden_section_minimize_algo(knots, n_iters=_i)
        # plt.show()
        plt.savefig(IMAGE_DIR / f"golden-section-{_i}.png")
    return


app._unparsable_cell(
    r"""
    # minimize_lad(x, y)


    def coordinate_descent_lad(X, y, beta: np.ndarray):
        n, d = X.shape
        for k in rang(d):
            r = y 
            if k > 0:
                r-= X[:, :k] @ beta[:k] 
            elif k >= d- 1
                r-= X[:, k + 1 :] @ beta[k + 1 :]
            a, x1, x2, b = golden_section_minimize(obj_fn, a=min(knots), b=max(knots), n_iters=n_iters)
    """,
    name="_"
)


@app.cell
def _():
    # show example fitting a line to data and using coordinate descent
    return


@app.cell
def _():
    # Example situations for the minimizer
    return


@app.cell
def _(IMAGE_DIR, make_cartesian_plane, np, plt):
    _beta = np.linspace(-2, 2, 1000)

    plt.plot(_beta, np.abs(_beta - 1) + np.abs(_beta - 0.5) + np.abs(_beta + 0.5), "k")
    _ax = plt.gca()

    _a = -0.5
    _b = 1
    _rho = 0.6
    _L = _b - _a
    _x1 = _b - _rho * _L
    _x2 = _a + _rho * _L

    _ax.vlines(
        [_a, _x1, _x2, _b],
        ymin=0,
        ymax=7,
        linestyle="--",
        colors=["tab:red", "tab:blue", "tab:green", "tab:orange"],
    )
    label_y = -0.5  # Position below ymin=0
    labels = ["a", "x₁", "x₂", "b"]
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for pos, label, color in zip([_a, _x1, _x2, _b], labels, colors):
        _ax.text(pos, label_y, label, ha="center", va="top", fontsize=24, color=color)

    make_cartesian_plane(_ax)
    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "case_1.png")
    plt.show()
    return


@app.cell
def _(IMAGE_DIR, make_cartesian_plane, np, plt):
    _beta = np.linspace(-2, 2, 1000)

    plt.plot(_beta, np.abs(_beta - 1) + np.abs(_beta - 0.5) + np.abs(_beta + 0.5), "k")
    _ax = plt.gca()
    _a = -0.5
    _b = 1
    _rho = 0.8
    _L = _b - _a
    _x1 = _b - _rho * _L
    _x2 = _a + _rho * _L

    _ax.vlines(
        [_a, _x1, _x2, _b],
        ymin=0,
        ymax=7,
        linestyle="--",
        colors=["tab:red", "tab:blue", "tab:green", "tab:orange"],
    )

    _label_y = -0.5  # Position below ymin=0
    _labels = ["a", "x₁", "x₂", "b"]
    _colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for _pos, _label, _color in zip([_a, _x1, _x2, _b], _labels, _colors):
        _ax.text(_pos, _label_y, _label, ha="center", va="top", fontsize=24, color=_color)
    make_cartesian_plane(_ax)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "case_2.png")

    plt.show()
    return


@app.cell
def _(IMAGE_DIR, make_cartesian_plane, np, plt):
    _beta = np.linspace(-2, 2, 1000)

    plt.plot(_beta, np.abs(_beta - 1) + np.abs(_beta + 1.5) + np.abs(_beta + 0.5), "k")
    _ax = plt.gca()
    _a = -1.5
    _b = 1
    _rho = 0.55
    _L = _b - _a
    _x1 = _b - _rho * _L
    _x2 = _a + _rho * _L

    _ax.vlines(
        [_a, _x1, _x2, _b],
        ymin=0,
        ymax=7,
        linestyle="--",
        colors=["tab:red", "tab:blue", "tab:green", "tab:orange"],
    )
    _label_y = -0.5  # Position below ymin=0
    _labels = ["a", "x₁", "x₂", "b"]
    _colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for _pos, _label, _color in zip([_a, _x1, _x2, _b], _labels, _colors):
        _ax.text(_pos, _label_y, _label, ha="center", va="top", fontsize=16, color=_color)
    make_cartesian_plane(_ax)
    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "case_3.png")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
