import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _():
    INSIDE_COLOR = "#2a6ebb"
    OUTSIDE_COLOR = "#aaaaaa"
    REFERENCE_COLOR = "#999999"
    GRID_LINE_COLOR = "#dddddd"
    return GRID_LINE_COLOR, INSIDE_COLOR, OUTSIDE_COLOR, REFERENCE_COLOR


@app.cell
def _(INSIDE_COLOR, np, plt):
    t = np.linspace(0, 2 * np.pi, 500)
    x_ellipse = 2 * np.cos(t)
    y_ellipse = np.sin(t)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_ellipse, y_ellipse, color=INSIDE_COLOR, linewidth=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$\frac{x^2}{4} + y^2 = 1$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    fig
    return


@app.function
def f_ellipse(x: float, y: float) -> float:
    return x**2 / 4 + y**2 - 1


@app.function
def f_tilted_ellipse(x: float, y: float) -> float:
    # f_ellipse rotated 45°: semi-axes a=2, b=1
    return 5 * x**2 / 8 - 3 * x * y / 4 + 5 * y**2 / 8 - 1


@app.function
def f_periodic(x: float, y: float) -> float:
    import math
    return math.sin(x) * math.sin(y) - 0.5


@app.cell
def _():
    from collections.abc import Callable
    from typing import TypeAlias

    Point: TypeAlias = tuple[float, float]
    Segment: TypeAlias = tuple[Point, Point]
    ScalarField: TypeAlias = Callable[[float, float], float]
    return Point, ScalarField, Segment


@app.cell
def _(Point: "TypeAlias", ScalarField: "TypeAlias", Segment: "TypeAlias", np):
    def _interp(corners: list[Point], fv: list[float], a: int, b: int) -> Point:
        xa, ya = corners[a]
        xb, yb = corners[b]
        t = fv[a] / (fv[a] - fv[b])
        return xa + t * (xb - xa), ya + t * (yb - ya)

    def _cell_segs(corners: list[Point], fv: list[float], mx: float, my: float, f: ScalarField) -> list[Segment]:
        crossed: list[Point] = []
        for k in range(4):
            nk = (k + 1) % 4
            if (fv[k] > 0) != (fv[nk] > 0):
                crossed.append(_interp(corners, fv, k, nk))
        if len(crossed) == 2:
            return [(crossed[0], crossed[1])]
        if len(crossed) == 4:
            # Saddle case: corner signs alternate (case 5 = 0,2 outside / 1,3 inside;
            # case 10 = 0,2 inside / 1,3 outside), so all 4 edges are crossed and two
            # segment pairings are topologically possible.  The asymptotic decider
            # evaluates f at the cell center to break the tie: whichever region (inside
            # or outside) contains the center is the *connected* one, so the opposite
            # region is split into two isolated pockets — each pocket is enclosed by one
            # arc using the *adjacent* crossing pairing.  For case 5: center outside →
            # inside is split → adjacent pairing; center inside → outside is split →
            # skip pairing.  Case 10 is symmetric (inside/outside roles swapped).
            case = sum((1 if fv[k] > 0 else 0) << k for k in range(4))
            center_in = f(mx, my) < 0
            if (case == 5) != center_in:
                return [(crossed[0], crossed[1]), (crossed[2], crossed[3])]
            return [(crossed[0], crossed[3]), (crossed[1], crossed[2])]
        return []

    def march_squares(xs: np.ndarray, ys: np.ndarray, f: ScalarField) -> tuple[np.ndarray, list[Segment]]:
        f_grid = np.array([[f(x, y) for x in xs] for y in ys])
        segs = []
        for j in range(len(ys) - 1):
            for i in range(len(xs) - 1):
                lower_left = (xs[i], ys[j])
                lower_right = (xs[i + 1], ys[j])
                upper_right = (xs[i + 1], ys[j + 1])
                upper_left = (xs[i], ys[j + 1])
                corners = [lower_left, lower_right, upper_right, upper_left]

                f_lower_left = f_grid[j, i]
                f_lower_right = f_grid[j, i + 1]
                f_upper_right = f_grid[j + 1, i + 1]
                f_upper_left = f_grid[j + 1, i]
                fv = [f_lower_left, f_lower_right, f_upper_right, f_upper_left]
                mx = (xs[i] + xs[i + 1]) / 2
                my = (ys[j] + ys[j + 1]) / 2
                segs += _cell_segs(corners, fv, mx, my, f)
        return f_grid, segs

    return (march_squares,)


@app.cell
def _(
    GRID_LINE_COLOR,
    INSIDE_COLOR,
    OUTSIDE_COLOR,
    REFERENCE_COLOR,
    march_squares,
    mo,
    np,
    plt,
):
    # ─── shared reference ellipse arc ───
    _t_ref = np.linspace(0, 2 * np.pi, 500)

    # ─── Left: 3×3 grid overview ───
    _n3 = 3
    _xs3 = np.linspace(-3, 3, _n3)
    _ys3 = np.linspace(-2, 2, _n3)
    _fg3, _segs3 = march_squares(_xs3, _ys3, f_ellipse)

    # highlighted cell: col=1, row=1 → x∈[0,3], y∈[0,2]
    _hi_i, _hi_j = 1, 1
    _cx0, _cx1 = float(_xs3[_hi_i]), float(_xs3[_hi_i + 1])
    _cy0, _cy1 = float(_ys3[_hi_j]), float(_ys3[_hi_j + 1])

    _fig_ov, _ax_ov = plt.subplots(figsize=(5, 4))
    _ax_ov.plot(2 * np.cos(_t_ref), np.sin(_t_ref), color=REFERENCE_COLOR, linewidth=1, linestyle="--", alpha=0.6)
    for _xi in _xs3:
        _ax_ov.axvline(_xi, color=GRID_LINE_COLOR, linewidth=0.8, zorder=0)
    for _yi in _ys3:
        _ax_ov.axhline(_yi, color=GRID_LINE_COLOR, linewidth=0.8, zorder=0)
    _ax_ov.fill_between([_cx0, _cx1], _cy0, _cy1, color=INSIDE_COLOR, alpha=0.1, zorder=1)
    _ax_ov.plot([_cx0, _cx1, _cx1, _cx0, _cx0], [_cy0, _cy0, _cy1, _cy1, _cy0],
                color=INSIDE_COLOR, linewidth=2, zorder=2)
    for _j, _y in enumerate(_ys3):
        for _i, _x in enumerate(_xs3):
            _c = INSIDE_COLOR if _fg3[_j, _i] < 0 else OUTSIDE_COLOR
            _ax_ov.plot(_x, _y, "o", color=_c, markersize=10, zorder=3)
    for (_x1, _y1), (_x2, _y2) in _segs3:
        _ax_ov.plot([_x1, _x2], [_y1, _y2], color=INSIDE_COLOR, linewidth=2.5, zorder=2)
    _ax_ov.set_aspect("equal")
    _ax_ov.spines[["top", "right"]].set_visible(False)
    _ax_ov.set_xlim(-3.3, 3.3)
    _ax_ov.set_ylim(-2.3, 2.3)
    _ax_ov.set_xlabel("x")
    _ax_ov.set_ylabel("y")
    _ax_ov.set_title("3×3 grid")
    _fig_ov.tight_layout()

    # ─── Right: single cell detail ───
    _cell_corners = [(_cx0, _cy0), (_cx1, _cy0), (_cx1, _cy1), (_cx0, _cy1)]
    _cell_fvals = [f_ellipse(x, y) for x, y in _cell_corners]

    _crossings = []
    for _k in range(4):
        _nk = (_k + 1) % 4
        _fa, _fb = _cell_fvals[_k], _cell_fvals[_nk]
        if (_fa > 0) != (_fb > 0):
            _tc = _fa / (_fa - _fb)
            _crossings.append((
                _cell_corners[_k][0] + _tc * (_cell_corners[_nk][0] - _cell_corners[_k][0]),
                _cell_corners[_k][1] + _tc * (_cell_corners[_nk][1] - _cell_corners[_k][1]),
                _tc,
            ))

    _fig_det, _ax_det = plt.subplots(figsize=(5, 5))
    _ax_det.plot(2 * np.cos(_t_ref), np.sin(_t_ref), color=REFERENCE_COLOR, linewidth=1.5, linestyle="--", alpha=0.7)
    _ax_det.plot(
        [c[0] for c in _cell_corners] + [_cell_corners[0][0]],
        [c[1] for c in _cell_corners] + [_cell_corners[0][1]],
        color="#555555", linewidth=1.5, zorder=2,
    )
    # corner dots + f-value labels
    _c_ha = ["left",  "left",  "left",  "left"]
    _c_va = ["top",   "top",   "bottom", "bottom"]
    _c_dx = [-0.55,   0.25,    0.25,   -0.55]
    _c_dy = [-0.25,  -0.25,    0.25,    0.25]
    for _k, ((_px, _py), _fv) in enumerate(zip(_cell_corners, _cell_fvals)):
        _col = INSIDE_COLOR if _fv < 0 else OUTSIDE_COLOR
        _ax_det.plot(_px, _py, "o", color=_col, markersize=13, zorder=5, clip_on=False)
        _ax_det.text(
            _px + _c_dx[_k], _py + _c_dy[_k],
            f"$f({_px:.0f},\\ {_py:.0f}) = {_fv:.2f}$",
            ha=_c_ha[_k], va=_c_va[_k], fontsize=9, color=_col, fontweight="bold",
        )
    # edge crossing markers + interpolation parameter labels
    _ann_dx = [0.0,  0.35]
    _ann_dy = [-0.35, 0.0]
    _ann_ha = ["center", "left"]
    _ann_va = ["top",    "center"]
    for _idx, (_px, _py, _tc) in enumerate(_crossings):
        _ax_det.plot(_px, _py, "D", color=INSIDE_COLOR, markersize=9, zorder=6,
                     markeredgecolor="white", markeredgewidth=0.8)
        _ax_det.annotate(
            f"$t = {_tc:.3f}$",
            xy=(_px, _py),
            xytext=(_px + _ann_dx[_idx], _py + _ann_dy[_idx]),
            fontsize=8.5, ha=_ann_ha[_idx], va=_ann_va[_idx], color=INSIDE_COLOR,
            arrowprops=dict(arrowstyle="->", color=INSIDE_COLOR, lw=0.9),
        )
    if len(_crossings) == 2:
        _ax_det.plot(
            [_crossings[0][0], _crossings[1][0]],
            [_crossings[0][1], _crossings[1][1]],
            color=INSIDE_COLOR, linewidth=2.5, zorder=4,
        )
    _ax_det.set_aspect("equal")
    _ax_det.set_xlim(_cx0 - 0.85, _cx1 + 1.8)
    _ax_det.set_ylim(_cy0 - 0.85, _cy1 + 0.85)
    _ax_det.spines[["top", "right"]].set_visible(False)
    _ax_det.set_xlabel("x")
    _ax_det.set_ylabel("y")
    _ax_det.set_title("Cell detail")
    _ax_det.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.3)
    _fig_det.tight_layout()

    mo.hstack([_fig_ov, _fig_det], justify="start", align="center")
    return


@app.cell
def _(mo):
    n_slider = mo.ui.slider(2, 16, value=3, step=1, label="Grid size")
    return (n_slider,)


@app.cell
def _(
    GRID_LINE_COLOR,
    INSIDE_COLOR,
    OUTSIDE_COLOR,
    REFERENCE_COLOR,
    march_squares,
    mo,
    n_slider,
    np,
    plt,
):
    _n = n_slider.value
    _xs = np.linspace(-3, 3, _n)
    _ys = np.linspace(-2, 2, _n)
    _fg, _segs = march_squares(_xs, _ys, f_ellipse)
    _ms = max(3, round(18 / _n))

    _fig, _ax = plt.subplots(figsize=(6, 4))

    _t = np.linspace(0, 2 * np.pi, 500)
    _ax.plot(2 * np.cos(_t), np.sin(_t), color=REFERENCE_COLOR, linewidth=1, linestyle="--", alpha=0.6)

    for _xi in _xs:
        _ax.axvline(_xi, color=GRID_LINE_COLOR, linewidth=0.8, zorder=0)
    for _yi in _ys:
        _ax.axhline(_yi, color=GRID_LINE_COLOR, linewidth=0.8, zorder=0)

    for _j, _y in enumerate(_ys):
        for _i, _x in enumerate(_xs):
            _c = INSIDE_COLOR if _fg[_j, _i] < 0 else OUTSIDE_COLOR
            _ax.plot(_x, _y, "o", color=_c, markersize=_ms, zorder=3)

    for (_x1, _y1), (_x2, _y2) in _segs:
        _ax.plot([_x1, _x2], [_y1, _y2], color=INSIDE_COLOR, linewidth=2.5, zorder=2)

    _ax.plot([], [], "o", color=INSIDE_COLOR, markersize=7, label=r"inside ($f < 0$)")
    _ax.plot([], [], "o", color=OUTSIDE_COLOR, markersize=7, label=r"outside ($f \geq 0$)")
    _ax.plot([], [], color=INSIDE_COLOR, linewidth=2.5, label="marching contour")
    _ax.plot([], [], color=REFERENCE_COLOR, linewidth=1, linestyle="--", alpha=0.6, label="true contour")
    _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, framealpha=0.9)
    _fig.tight_layout()

    _ax.set_aspect("equal")
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_title(f"Square marching — {_n}×{_n} grid (axis-aligned)")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.set_xlim(-3.3, 3.3)
    _ax.set_ylim(-2.3, 2.3)
    _ax.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.3)
    mo.hstack([_fig, n_slider], align="center", justify="start")
    return


@app.cell
def _(
    GRID_LINE_COLOR,
    INSIDE_COLOR,
    OUTSIDE_COLOR,
    REFERENCE_COLOR,
    march_squares,
    mo,
    np,
    plt,
):
    _t_ref_t = np.linspace(0, 2 * np.pi, 500)
    _x_ref_t = (2 * np.cos(_t_ref_t) - np.sin(_t_ref_t)) / np.sqrt(2)
    _y_ref_t = (2 * np.cos(_t_ref_t) + np.sin(_t_ref_t)) / np.sqrt(2)

    # ─── Left: 4×4 grid overview ───
    _n4 = 4
    _xs4 = np.linspace(-3, 3, _n4)
    _ys4 = np.linspace(-3, 3, _n4)
    _fg4, _segs4 = march_squares(_xs4, _ys4, f_tilted_ellipse)

    # center cell: col=1, row=1 → x∈[-1,1], y∈[-1,1]
    _hi_i4, _hi_j4 = 1, 1
    _cx0t, _cx1t = float(_xs4[_hi_i4]), float(_xs4[_hi_i4 + 1])
    _cy0t, _cy1t = float(_ys4[_hi_j4]), float(_ys4[_hi_j4 + 1])

    _fig_ov_t, _ax_ov_t = plt.subplots(figsize=(5, 5))
    _ax_ov_t.plot(_x_ref_t, _y_ref_t, color=REFERENCE_COLOR, linewidth=1, linestyle="--", alpha=0.6)
    for _xi in _xs4:
        _ax_ov_t.axvline(_xi, color=GRID_LINE_COLOR, linewidth=0.8, zorder=0)
    for _yi in _ys4:
        _ax_ov_t.axhline(_yi, color=GRID_LINE_COLOR, linewidth=0.8, zorder=0)
    _ax_ov_t.fill_between([_cx0t, _cx1t], _cy0t, _cy1t, color=INSIDE_COLOR, alpha=0.1, zorder=1)
    _ax_ov_t.plot(
        [_cx0t, _cx1t, _cx1t, _cx0t, _cx0t],
        [_cy0t, _cy0t, _cy1t, _cy1t, _cy0t],
        color=INSIDE_COLOR, linewidth=2, zorder=2,
    )
    for _j, _y in enumerate(_ys4):
        for _i, _x in enumerate(_xs4):
            _c = INSIDE_COLOR if _fg4[_j, _i] < 0 else OUTSIDE_COLOR
            _ax_ov_t.plot(_x, _y, "o", color=_c, markersize=8, zorder=3)
    for (_x1, _y1), (_x2, _y2) in _segs4:
        _ax_ov_t.plot([_x1, _x2], [_y1, _y2], color=INSIDE_COLOR, linewidth=2.5, zorder=2)
    _ax_ov_t.set_aspect("equal")
    _ax_ov_t.spines[["top", "right"]].set_visible(False)
    _ax_ov_t.set_xlim(-3.3, 3.3)
    _ax_ov_t.set_ylim(-3.3, 3.3)
    _ax_ov_t.set_xlabel("x")
    _ax_ov_t.set_ylabel("y")
    _ax_ov_t.set_title("4×4 grid (tilted 45°)")
    _fig_ov_t.tight_layout()

    # ─── Right: center cell detail ───
    _cell_corners_t = [(_cx0t, _cy0t), (_cx1t, _cy0t), (_cx1t, _cy1t), (_cx0t, _cy1t)]
    _cell_fvals_t = [f_tilted_ellipse(x, y) for x, y in _cell_corners_t]

    _crossings_t = []
    for _k in range(4):
        _nk = (_k + 1) % 4
        _fa, _fb = _cell_fvals_t[_k], _cell_fvals_t[_nk]
        if (_fa > 0) != (_fb > 0):
            _tc = _fa / (_fa - _fb)
            _crossings_t.append((
                _cell_corners_t[_k][0] + _tc * (_cell_corners_t[_nk][0] - _cell_corners_t[_k][0]),
                _cell_corners_t[_k][1] + _tc * (_cell_corners_t[_nk][1] - _cell_corners_t[_k][1]),
                _tc,
            ))

    _case_t = sum((1 if _fv > 0 else 0) << _k for _k, _fv in enumerate(_cell_fvals_t))
    _mx_t = (_cx0t + _cx1t) / 2
    _my_t = (_cy0t + _cy1t) / 2
    _center_fval_t = f_tilted_ellipse(_mx_t, _my_t)
    _center_in_t = _center_fval_t < 0

    _fig_det_t, _ax_det_t = plt.subplots(figsize=(5, 5))
    _ax_det_t.plot(_x_ref_t, _y_ref_t, color=REFERENCE_COLOR, linewidth=1.5, linestyle="--", alpha=0.7)
    _ax_det_t.plot(
        [c[0] for c in _cell_corners_t] + [_cell_corners_t[0][0]],
        [c[1] for c in _cell_corners_t] + [_cell_corners_t[0][1]],
        color="#555555", linewidth=1.5, zorder=2,
    )

    # corner dots + f-value labels (outward from each corner)
    _c_ha_t = ["right", "left",  "left",  "right"]
    _c_va_t = ["top",   "top",   "bottom", "bottom"]
    _c_dx_t = [-0.12,   0.12,    0.12,   -0.12]
    _c_dy_t = [-0.15,  -0.15,    0.15,    0.15]
    for _k, ((_px, _py), _fv) in enumerate(zip(_cell_corners_t, _cell_fvals_t)):
        _col = INSIDE_COLOR if _fv < 0 else OUTSIDE_COLOR
        _ax_det_t.plot(_px, _py, "o", color=_col, markersize=13, zorder=5, clip_on=False)
        _ax_det_t.text(
            _px + _c_dx_t[_k], _py + _c_dy_t[_k],
            f"$f({_px:.0f},\\ {_py:.0f}) = {_fv:.2f}$",
            ha=_c_ha_t[_k], va=_c_va_t[_k], fontsize=9, color=_col, fontweight="bold",
        )

    # crossing markers + t-value annotations (outward from cell boundary)
    _ann_off_t = [
        (0.0,  -0.35, "center", "top"),    # bottom edge
        (0.35,  0.0,  "left",   "center"), # right edge
        (0.0,   0.35, "center", "bottom"), # top edge
        (-0.35, 0.0,  "right",  "center"), # left edge
    ]
    for (_px, _py, _tc), (_adx, _ady, _ha, _va) in zip(_crossings_t, _ann_off_t):
        _ax_det_t.plot(_px, _py, "D", color=INSIDE_COLOR, markersize=9, zorder=6,
                       markeredgecolor="white", markeredgewidth=0.8)
        _ax_det_t.annotate(
            f"$t = {_tc:.3f}$",
            xy=(_px, _py), xytext=(_px + _adx, _py + _ady),
            fontsize=8.5, ha=_ha, va=_va, color=INSIDE_COLOR,
            arrowprops=dict(arrowstyle="->", color=INSIDE_COLOR, lw=0.9),
        )

    # both contour segments (adjacent pairing chosen by asymptotic decider)
    if len(_crossings_t) == 4:
        _ax_det_t.plot(
            [_crossings_t[0][0], _crossings_t[1][0]],
            [_crossings_t[0][1], _crossings_t[1][1]],
            color=INSIDE_COLOR, linewidth=2.5, zorder=4,
        )
        _ax_det_t.plot(
            [_crossings_t[2][0], _crossings_t[3][0]],
            [_crossings_t[2][1], _crossings_t[3][1]],
            color=INSIDE_COLOR, linewidth=2.5, zorder=4,
        )

    # mark cell center (the asymptotic decider sample point)
    _ax_det_t.plot(_mx_t, _my_t, "*", color="#555555", markersize=11, zorder=7)
    _ax_det_t.annotate(
        f"$f(0,\\ 0) = {_center_fval_t:.2f}$ ({'inside' if _center_in_t else 'outside'})",
        xy=(_mx_t, _my_t), xytext=(0.35, -0.5),
        fontsize=8.5, ha="left", va="top", color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
    )

    _ax_det_t.set_aspect("equal")
    _ax_det_t.set_xlim(_cx0t - 1.5, _cx1t + 1.5)
    _ax_det_t.set_ylim(_cy0t - 1.5, _cy1t + 1.5)
    _ax_det_t.spines[["top", "right"]].set_visible(False)
    _ax_det_t.set_xlabel("x")
    _ax_det_t.set_ylabel("y")
    _ax_det_t.set_title(
        f"Case {_case_t} ({_case_t:04b}$_2$) — alternating corners, 4 crossings (saddle)"
    )
    _ax_det_t.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.3)
    _fig_det_t.tight_layout()

    mo.hstack([_fig_ov_t, _fig_det_t], justify="start", align="center")
    return


@app.cell
def _(mo):
    n_slider_tilted = mo.ui.slider(2, 16, value=3, step=1, label="Grid size")
    return (n_slider_tilted,)


@app.cell
def _(
    GRID_LINE_COLOR,
    INSIDE_COLOR,
    OUTSIDE_COLOR,
    REFERENCE_COLOR,
    march_squares,
    mo,
    n_slider_tilted,
    np,
    plt,
):
    _n = n_slider_tilted.value
    _xs = np.linspace(-3, 3, _n)
    _ys = np.linspace(-3, 3, _n)
    _fg, _segs = march_squares(_xs, _ys, f_tilted_ellipse)
    _ms = max(3, round(18 / _n))

    _fig, _ax = plt.subplots(figsize=(6, 6))

    # True parametric tilted ellipse: rotate (a cos t, b sin t) by 45°
    _t = np.linspace(0, 2 * np.pi, 500)
    _x_ref = (2 * np.cos(_t) - np.sin(_t)) / np.sqrt(2)
    _y_ref = (2 * np.cos(_t) + np.sin(_t)) / np.sqrt(2)
    _ax.plot(_x_ref, _y_ref, color=REFERENCE_COLOR, linewidth=1, linestyle="--", alpha=0.6)

    for _xi in _xs:
        _ax.axvline(_xi, color=GRID_LINE_COLOR, linewidth=0.8, zorder=0)
    for _yi in _ys:
        _ax.axhline(_yi, color=GRID_LINE_COLOR, linewidth=0.8, zorder=0)

    for _j, _y in enumerate(_ys):
        for _i, _x in enumerate(_xs):
            _c = INSIDE_COLOR if _fg[_j, _i] < 0 else OUTSIDE_COLOR
            _ax.plot(_x, _y, "o", color=_c, markersize=_ms, zorder=3)

    for (_x1, _y1), (_x2, _y2) in _segs:
        _ax.plot([_x1, _x2], [_y1, _y2], color=INSIDE_COLOR, linewidth=2.5, zorder=2)

    _ax.plot([], [], "o", color=INSIDE_COLOR, markersize=7, label=r"inside ($f < 0$)")
    _ax.plot([], [], "o", color=OUTSIDE_COLOR, markersize=7, label=r"outside ($f \geq 0$)")
    _ax.plot([], [], color=INSIDE_COLOR, linewidth=2.5, label="marching contour")
    _ax.plot([], [], color=REFERENCE_COLOR, linewidth=1, linestyle="--", alpha=0.6, label="true contour")
    _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, framealpha=0.9)
    _fig.tight_layout()

    _ax.set_aspect("equal")
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_title(f"Square marching — {_n}×{_n} grid (tilted 45°)")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.set_xlim(-2.5, 2.5)
    _ax.set_ylim(-2.5, 2.5)
    _ax.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.3)
    mo.hstack([_fig, n_slider_tilted], align="center", justify="start")
    return


@app.cell
def _(mo):
    n_slider_periodic = mo.ui.slider(2, 32, value=9, step=1, label="Grid size")
    return (n_slider_periodic,)


@app.cell
def _(
    GRID_LINE_COLOR,
    INSIDE_COLOR,
    OUTSIDE_COLOR,
    REFERENCE_COLOR,
    march_squares,
    mo,
    n_slider_periodic,
    np,
    plt,
):
    _n = n_slider_periodic.value
    _lim = 2 * np.pi
    _xs = np.linspace(-_lim, _lim, _n)
    _ys = np.linspace(-_lim, _lim, _n)
    _fg, _segs = march_squares(_xs, _ys, f_periodic)
    _ms = max(2, round(14 / _n))

    _fig, _ax = plt.subplots(figsize=(6, 6))

    # Dense reference contour — no parametric form exists for this function
    _xd = np.linspace(-_lim, _lim, 500)
    _yd = np.linspace(-_lim, _lim, 500)
    _Xd, _Yd = np.meshgrid(_xd, _yd)
    _ax.contour(_xd, _yd, np.sin(_Xd) * np.sin(_Yd) - 0.5, levels=[0],
                colors=[REFERENCE_COLOR], linewidths=1, linestyles="--", alpha=0.6)

    for _xi in _xs:
        _ax.axvline(_xi, color=GRID_LINE_COLOR, linewidth=0.8, zorder=0)
    for _yi in _ys:
        _ax.axhline(_yi, color=GRID_LINE_COLOR, linewidth=0.8, zorder=0)

    for _j, _y in enumerate(_ys):
        for _i, _x in enumerate(_xs):
            _c = INSIDE_COLOR if _fg[_j, _i] < 0 else OUTSIDE_COLOR
            _ax.plot(_x, _y, "o", color=_c, markersize=_ms, zorder=3)

    for (_x1, _y1), (_x2, _y2) in _segs:
        _ax.plot([_x1, _x2], [_y1, _y2], color=INSIDE_COLOR, linewidth=2.5, zorder=2)

    _ax.plot([], [], "o", color=INSIDE_COLOR, markersize=7, label=r"inside ($f < 0$)")
    _ax.plot([], [], "o", color=OUTSIDE_COLOR, markersize=7, label=r"outside ($f \geq 0$)")
    _ax.plot([], [], color=INSIDE_COLOR, linewidth=2.5, label="marching contour")
    _ax.plot([], [], color=REFERENCE_COLOR, linewidth=1, linestyle="--", alpha=0.6, label="true contour")
    _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, framealpha=0.9)
    _fig.tight_layout()

    _ax.set_aspect("equal")
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_title(rf"Square marching — {_n}×{_n} grid ($\sin x \cdot \sin y = \frac{{1}}{{2}}$)")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.set_xlim(-_lim - 0.3, _lim + 0.3)
    _ax.set_ylim(-_lim - 0.3, _lim + 0.3)
    _ax.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.3)
    mo.hstack([_fig, n_slider_periodic], align="center", justify="start")
    return


if __name__ == "__main__":
    app.run()
