import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def imports():
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
def plot_helpers(plt):
    PALETTE = {
        "a": "#dc2626",
        "x1": "#2563eb",
        "x2": "#059669",
        "b": "#d97706",
        "curve": "#1e1e1e",
        "terms": "#7c3aed",
    }
    ABSCISSA_COLORS = [PALETTE["a"], PALETTE["x1"], PALETTE["x2"], PALETTE["b"]]

    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "text.color": "#1e1e1e",
            "axes.labelcolor": "#1e1e1e",
            "xtick.color": "#888888",
            "ytick.color": "#888888",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "figure.figsize": (8, 5),
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.facecolor": "#ffffff",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "lines.linewidth": 2.2,
            "font.size": 12,
            "font.family": "monospace",
        }
    )


    def make_cartesian_plane(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_position("zero")
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_color("#cccccc")
        ax.spines["left"].set_color("#cccccc")


    def remove_spines(ax):
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    return ABSCISSA_COLORS, PALETTE, make_cartesian_plane


@app.cell
def absolute_deviation_data(np):
    def absolute_deviation(x, y, beta):
        return np.mean(np.abs(beta * x - y))

    np.random.seed(31)
    n_points = 7
    x = np.random.randint(size=n_points, low=1, high=25)
    y = np.random.randint(size=n_points, low=-25, high=25)
    knots = y / x
    beta = np.linspace(-2.5, 2.5, 1000)
    return absolute_deviation, beta, knots, x, y


@app.cell
def lad_progression(
    IMAGE_DIR,
    PALETTE,
    beta,
    make_cartesian_plane,
    np,
    plt,
    x,
    y,
):
    _n = len(x)
    _muted = "#aaaaaa"

    for _i in range(_n):
        _fig, _ax = plt.subplots()

        # Individual terms — translucent
        for _j in range(_i + 1):
            _ax.plot(
                beta,
                np.abs(x[_j] * beta - y[_j]),
                color=_muted,
                alpha=0.4,
                linewidth=1.5,
                zorder=2,
            )

        # Sum total — solid
        _sum = np.mean(np.abs(np.outer(beta, x[: _i + 1]) - y[: _i + 1]), axis=1)
        _ax.plot(beta, _sum, color=PALETTE["curve"], linewidth=2.5, zorder=3)

        # Kink markers: dot on the axis + short tick below
        _kinks_i = y[: _i + 1] / x[: _i + 1]
        for _kink in _kinks_i:
            _ax.scatter([_kink], [0], color=_muted, s=55, zorder=5, clip_on=False)

        make_cartesian_plane(_ax)
        _ax.set_ylim([-4, 50])
        _fig.tight_layout()
        _fig.savefig(IMAGE_DIR / f"1d-abs-deviation-{_i:02}.png")
        plt.close(_fig)
    return


@app.cell
def knots_overview(
    PALETTE,
    absolute_deviation,
    beta,
    knots,
    make_cartesian_plane,
    np,
    plt,
    x,
    y,
):
    np.random.seed(31)

    absolute_deviations = np.array([absolute_deviation(x, y, beta_i) for beta_i in beta])

    _fig, _ax = plt.subplots()
    _ax.plot(beta, absolute_deviations, color=PALETTE["curve"], linewidth=2.5)
    _ax.set_ylim([0, 40])
    make_cartesian_plane(_ax)
    _ax.scatter(
        knots,
        np.array([absolute_deviation(x, y, beta_i) for beta_i in knots]),
        color=PALETTE["x1"],
        s=55,
        alpha=0.85,
        zorder=4,
        edgecolors="none",
    )
    plt.show()
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
def golden_section_helpers(
    ABSCISSA_COLORS,
    PALETTE,
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
        make_cartesian_plane(ax)

        ax.plot(beta, absolute_deviations, color=PALETTE["curve"], linewidth=2.5, zorder=2)
        ax.set_ylim([0, 50])
        ax.scatter(
            knots,
            np.array([absolute_deviation(x, y, beta_i) for beta_i in knots]),
            color=PALETTE["curve"],
            s=40,
            alpha=0.45,
            zorder=3,
            edgecolors="none",
        )

        _abscissae = [a, x1, x2, b]
        _labels = ["a", "x₁", "x₂", "b"]

        ax.vlines(
            _abscissae,
            ymin=0,
            ymax=50,
            linestyle="--",
            colors=ABSCISSA_COLORS,
            linewidth=1.5,
            alpha=0.85,
            zorder=4,
        )

        _label_y = -3.5
        for _pos, _label, _color in zip(_abscissae, _labels, ABSCISSA_COLORS):
            ax.text(
                _pos,
                _label_y,
                _label,
                ha="center",
                va="top",
                fontsize=13,
                color=_color,
                fontweight="bold",
            )

        ax.set_title(f"iteration {n_iters}", color="#888888", fontsize=11, pad=10)

        return fig, ax

    return (plot_golden_section_minimize_algo,)


@app.cell
def save_golden_section_plots(
    IMAGE_DIR,
    knots,
    plot_golden_section_minimize_algo,
    plt,
):
    for _i in range(4):
        _fig, _ax = plot_golden_section_minimize_algo(knots, n_iters=_i)
        _fig.savefig(IMAGE_DIR / f"golden-section-{_i}.png")
        plt.close(_fig)
    return


@app.cell
def three_points_data(np):
    p1, p2, p3 = 0.1, 0.5, 0.9
    y1, y2, y3 = 2.0, 1.0, 1.8
    m_a = 0.30  # minimum x for curve A (left of p2)
    m_b = 0.70  # minimum x for curve B (right of p2)

    # min values derived by extrapolating each outer segment to the minimum x
    y_min_a = y2 + (y3 - y2) / (p3 - p2) * (m_a - p2)  # ≈ 0.6
    y_min_b = y2 + (y1 - y2) / (p1 - p2) * (m_b - p2)  # = 0.5


    def curve_a(x):
        return np.where(
            x <= m_a,
            y_min_a + (y1 - y_min_a) / (p1 - m_a) * (x - m_a),
            y_min_a + (y2 - y_min_a) / (p2 - m_a) * (x - m_a),
        )


    def curve_b(x):
        return np.where(
            x <= m_b,
            y_min_b + (y2 - y_min_b) / (p2 - m_b) * (x - m_b),
            y_min_b + (y3 - y_min_b) / (p3 - m_b) * (x - m_b),
        )


    probe_xs = np.array([p1, p2, p3])
    probe_ys = np.array([y1, y2, y3])
    return curve_a, curve_b, probe_xs, probe_ys


@app.function
def three_points_axes_style(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.spines["left"].set_color("#cccccc")
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim([0, 2.5])


@app.function
def draw_probes(ax, probe_xs, probe_ys, palette):
    ax.scatter(
        probe_xs,
        probe_ys,
        color=palette["curve"],
        s=120,
        zorder=7,
        edgecolors="white",
        linewidths=1.5,
    )
    for xp, yp, label in zip(probe_xs, probe_ys, ["a", "x₁", "b"]):
        ax.text(
            xp,
            yp + 0.08,
            label,
            ha="center",
            va="bottom",
            fontsize=14,
            color=palette["curve"],
            fontweight="bold",
        )


@app.cell
def three_points_probes_only(IMAGE_DIR, PALETTE, plt, probe_xs, probe_ys):
    _fig, _ax = plt.subplots()
    _ax.scatter(
        probe_xs,
        probe_ys,
        color=PALETTE["curve"],
        s=120,
        zorder=7,
        edgecolors="white",
        linewidths=1.5,
    )
    for _xp, _yp, _label in zip(probe_xs, probe_ys, ["a", "x₁", "b"]):
        _ax.text(
            _xp,
            _yp + 0.08,
            _label,
            ha="center",
            va="bottom",
            fontsize=14,
            color=PALETTE["curve"],
            fontweight="bold",
        )
    three_points_axes_style(_ax)
    _fig.tight_layout()
    _fig.savefig(IMAGE_DIR / "three_points_probes_only.png")
    plt.show()
    return


@app.cell
def three_points_left_min(
    IMAGE_DIR,
    PALETTE,
    curve_a,
    np,
    plt,
    probe_xs,
    probe_ys,
):
    _x = np.linspace(0.02, 0.98, 1000)
    _fig, _ax = plt.subplots()
    _ax.plot(_x, curve_a(_x), color=PALETTE["x1"], linewidth=2.5, linestyle="--", alpha=0.45)
    draw_probes(_ax, probe_xs, probe_ys, PALETTE)
    three_points_axes_style(_ax)
    _fig.tight_layout()
    _fig.savefig(IMAGE_DIR / "three_points_left_min.png")
    plt.show()
    return


@app.cell
def three_points_right_min(
    IMAGE_DIR,
    PALETTE,
    curve_b,
    np,
    plt,
    probe_xs,
    probe_ys,
):
    _x = np.linspace(0.02, 0.98, 1000)
    _fig, _ax = plt.subplots()
    _ax.plot(_x, curve_b(_x), color=PALETTE["x2"], linewidth=2.5, linestyle="--", alpha=0.45)
    draw_probes(_ax, probe_xs, probe_ys, PALETTE)
    three_points_axes_style(_ax)
    _fig.tight_layout()
    _fig.savefig(IMAGE_DIR / "three_points_right_min.png")
    plt.show()
    return


@app.function
def plot_rho_iterations(rho, ax, abscissa_colors):
    a, b = 0.0, 1.0
    L = b - a

    x1_0 = b - rho * L
    x2_0 = a + rho * L

    # Iteration 1: assume f(x1) < f(x2), keep left subinterval [a, x2_0]
    L1 = x2_0 - a
    x1_1 = x2_0 - rho * L1
    x2_1 = a + rho * L1

    iter0_xs = [a, x1_0, x2_0, b]
    iter1_xs = [a, x1_1, x2_1, x2_0]  # new b = old x2_0

    tol = 1e-9
    is_reused = [any(abs(xi - x0) < tol for x0 in iter0_xs) for xi in iter1_xs]

    y0, y1 = 1.0, 0.0

    ax.hlines([y0, y1], a, b, color="#d1d5db", linewidth=1.5, zorder=0)

    for xi, label, color in zip(iter0_xs, ["a", "x₁", "x₂", "b"], abscissa_colors):
        ax.scatter(xi, y0, color=color, s=90, zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(xi, y0 + 0.14, label, ha="center", va="bottom", fontsize=13, color=color, fontweight="bold")

    for xi_1, reused in zip(iter1_xs, is_reused):
        if reused:
            ax.plot([xi_1, xi_1], [y0, y1], color="#d1d5db", linewidth=1.2, linestyle=":", zorder=1)

    for xi, label, reused, color in zip(iter1_xs, ["a", "x₁", "x₂", "b"], is_reused, abscissa_colors):
        alpha = 0.35 if reused else 1.0
        marker = "o" if reused else "D"
        ax.scatter(xi, y1, color=color, s=90, zorder=3, edgecolors="white", linewidths=1.5, marker=marker, alpha=alpha)
        ax.text(xi, y1 - 0.14, label, ha="center", va="top", fontsize=13, color=color, fontweight="bold", alpha=alpha)

    ax.text(a - 0.06, y0, "iter 0", ha="right", va="center", fontsize=10, color="#9ca3af", style="italic")
    ax.text(a - 0.06, y1, "iter 1", ha="right", va="center", fontsize=10, color="#9ca3af", style="italic")

    ax.set_xlim(a - 0.16, b + 0.04)
    ax.set_ylim(-0.6, 1.6)
    ax.axis("off")


@app.cell
def rho_06(ABSCISSA_COLORS, IMAGE_DIR, plt):
    _fig, _ax = plt.subplots(figsize=(8, 2.5))
    plot_rho_iterations(rho=0.6, ax=_ax, abscissa_colors=ABSCISSA_COLORS)
    _ax.set_title("ρ = 0.6", fontsize=11, color="#6b7280", pad=14)
    _fig.tight_layout()
    _fig.savefig(IMAGE_DIR / "rho_06.png")
    plt.show()
    return


@app.cell
def rho_07(ABSCISSA_COLORS, IMAGE_DIR, plt):
    _fig, _ax = plt.subplots(figsize=(8, 2.5))
    plot_rho_iterations(rho=0.7, ax=_ax, abscissa_colors=ABSCISSA_COLORS)
    _ax.set_title("ρ = 0.7", fontsize=11, color="#6b7280", pad=14)
    _fig.tight_layout()
    _fig.savefig(IMAGE_DIR / "rho_07.png")
    plt.show()
    return


@app.cell
def rho_golden_section(ABSCISSA_COLORS, IMAGE_DIR, plt):
    _alpha = (-1 + 5**0.5) / 2
    _fig, _ax = plt.subplots(figsize=(8, 2.5))
    plot_rho_iterations(rho=_alpha, ax=_ax, abscissa_colors=ABSCISSA_COLORS)
    _ax.set_title(f"ρ = 1/φ ≈ {_alpha:.3f}", fontsize=11, color="#6b7280", pad=14)
    _fig.tight_layout()
    _fig.savefig(IMAGE_DIR / "rho_golden_section.png")
    plt.show()
    return


@app.cell
def case_1_bare(IMAGE_DIR, np, plot_case_bare, plt):
    def _obj_fn(beta):
        return np.abs(beta - 1) + np.abs(beta - 0.5) + np.abs(beta + 0.5)


    _fig, _ax = plot_case_bare(a=-0.5, b=1, rho=0.6, obj_fn=_obj_fn)
    _fig.savefig(IMAGE_DIR / "case_1_bare.png")
    plt.show()
    return


@app.cell
def case_2_bare(IMAGE_DIR, np, plot_case_bare, plt):
    def _obj_fn(beta):
        return np.abs(beta - 1) + np.abs(beta - 0.5) + np.abs(beta + 0.5)


    _fig, _ax = plot_case_bare(a=-0.5, b=1, rho=0.8, obj_fn=_obj_fn)
    _fig.savefig(IMAGE_DIR / "case_2_bare.png")
    plt.show()
    return


@app.cell
def case_3_bare(IMAGE_DIR, np, plot_case_bare, plt):
    def _obj_fn(beta):
        return np.abs(beta - 1) + np.abs(beta + 1.5) + np.abs(beta + 0.5)


    _fig, _ax = plot_case_bare(a=-1.5, b=1, rho=0.55, obj_fn=_obj_fn)
    _fig.savefig(IMAGE_DIR / "case_3_bare.png")
    plt.show()
    return


@app.cell
def bare_case_helpers(ABSCISSA_COLORS, make_cartesian_plane, plt):
    def plot_case_bare(a, b, rho, obj_fn):
        L = b - a
        x1 = b - rho * L
        x2 = a + rho * L
        abscissae = [a, x1, x2, b]
        y_vals = [obj_fn(pos) for pos in abscissae]
        y_max = max(y_vals)

        fig, ax = plt.subplots()

        ax.vlines(
            abscissae,
            ymin=0,
            ymax=y_vals,
            colors=ABSCISSA_COLORS,
            linewidth=2,
            linestyle="--",
            alpha=0.85,
            zorder=3,
        )
        ax.scatter(abscissae, y_vals, color=ABSCISSA_COLORS, s=80, zorder=4, edgecolors="white", linewidths=1.5)

        _label_y = -0.6
        for _pos, _label, _color in zip(abscissae, ["a", "x₁", "x₂", "b"], ABSCISSA_COLORS):
            ax.text(_pos, _label_y, _label, ha="center", va="top", fontsize=22, color=_color, fontweight="bold")

        make_cartesian_plane(ax)
        ax.set_ylim([-1, 5])
        fig.tight_layout()
        return fig, ax

    return (plot_case_bare,)


@app.cell
def case_1(ABSCISSA_COLORS, IMAGE_DIR, PALETTE, make_cartesian_plane, np, plt):
    _a, _b, _rho = -0.5, 1, 0.6
    _L = _b - _a
    _beta = np.linspace(_a - _L * 0.2, _b + _L * 0.2, 1000)
    _fig, _ax = plt.subplots()

    _ax.plot(
        _beta,
        np.abs(_beta - 1) + np.abs(_beta - 0.5) + np.abs(_beta + 0.5),
        color=PALETTE["curve"],
        linewidth=2.5,
        linestyle="--",
        alpha=0.35,
    )

    _x1 = _b - _rho * _L
    _x2 = _a + _rho * _L
    _abscissae = [_a, _x1, _x2, _b]
    _y_vals = [np.abs(p - 1) + np.abs(p - 0.5) + np.abs(p + 0.5) for p in _abscissae]

    _ax.vlines(_abscissae, ymin=0, ymax=_y_vals, linestyle="--", colors=ABSCISSA_COLORS, linewidth=1.5, alpha=0.85)
    _ax.scatter(_abscissae, _y_vals, color=ABSCISSA_COLORS, s=80, zorder=4, edgecolors="white", linewidths=1.5)
    _ax.set_ylim([-1, 5])

    _label_y = -0.6
    for _pos, _label, _color in zip(_abscissae, ["a", "x₁", "x₂", "b"], ABSCISSA_COLORS):
        _ax.text(_pos, _label_y, _label, ha="center", va="top", fontsize=22, color=_color, fontweight="bold")

    make_cartesian_plane(_ax)
    _fig.tight_layout()
    _fig.savefig(IMAGE_DIR / "case_1.png")
    plt.show()
    return


@app.cell
def case_2(ABSCISSA_COLORS, IMAGE_DIR, PALETTE, make_cartesian_plane, np, plt):
    _a, _b, _rho = -0.5, 1, 0.8
    _L = _b - _a
    _beta = np.linspace(_a - _L * 0.2, _b + _L * 0.2, 1000)
    _fig, _ax = plt.subplots()

    _ax.plot(
        _beta,
        np.abs(_beta - 1) + np.abs(_beta - 0.5) + np.abs(_beta + 0.5),
        color=PALETTE["curve"],
        linewidth=2.5,
        linestyle="--",
        alpha=0.35,
    )

    _x1 = _b - _rho * _L
    _x2 = _a + _rho * _L
    _abscissae = [_a, _x1, _x2, _b]
    _y_vals = [np.abs(p - 1) + np.abs(p - 0.5) + np.abs(p + 0.5) for p in _abscissae]

    _ax.vlines(_abscissae, ymin=0, ymax=_y_vals, linestyle="--", colors=ABSCISSA_COLORS, linewidth=1.5, alpha=0.85)
    _ax.scatter(_abscissae, _y_vals, color=ABSCISSA_COLORS, s=80, zorder=4, edgecolors="white", linewidths=1.5)
    _ax.set_ylim([-1, 5])

    _label_y = -0.6
    for _pos, _label, _color in zip(_abscissae, ["a", "x₁", "x₂", "b"], ABSCISSA_COLORS):
        _ax.text(_pos, _label_y, _label, ha="center", va="top", fontsize=22, color=_color, fontweight="bold")

    make_cartesian_plane(_ax)
    _fig.tight_layout()
    _fig.savefig(IMAGE_DIR / "case_2.png")
    plt.show()
    return


@app.cell
def case_3(ABSCISSA_COLORS, IMAGE_DIR, PALETTE, make_cartesian_plane, np, plt):
    _a, _b, _rho = -1.5, 1, 0.55
    _L = _b - _a
    _beta = np.linspace(_a - _L * 0.2, _b + _L * 0.2, 1000)
    _fig, _ax = plt.subplots()

    _ax.plot(
        _beta,
        np.abs(_beta - 1) + np.abs(_beta + 1.5) + np.abs(_beta + 0.5),
        color=PALETTE["curve"],
        linewidth=2.5,
        linestyle="--",
        alpha=0.35,
    )

    _x1 = _b - _rho * _L
    _x2 = _a + _rho * _L
    _abscissae = [_a, _x1, _x2, _b]
    _y_vals = [np.abs(p - 1) + np.abs(p + 1.5) + np.abs(p + 0.5) for p in _abscissae]

    _ax.vlines(_abscissae, ymin=0, ymax=_y_vals, linestyle="--", colors=ABSCISSA_COLORS, linewidth=1.5, alpha=0.85)
    _ax.scatter(_abscissae, _y_vals, color=ABSCISSA_COLORS, s=80, zorder=4, edgecolors="white", linewidths=1.5)
    _ax.set_ylim([-1, 5])

    _label_y = -0.6
    for _pos, _label, _color in zip(_abscissae, ["a", "x₁", "x₂", "b"], ABSCISSA_COLORS):
        _ax.text(_pos, _label_y, _label, ha="center", va="top", fontsize=22, color=_color, fontweight="bold")

    make_cartesian_plane(_ax)
    _fig.tight_layout()
    _fig.savefig(IMAGE_DIR / "case_3.png")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
