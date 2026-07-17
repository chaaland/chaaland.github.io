import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def imports():
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    from scipy.optimize import nnls
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    return LineCollection, LinearSegmentedColormap, mo, nnls, np, plt


@app.cell
def colors():
    A1 = "#2563eb"  # blue: OLS scatter / f₁ / a₁ vector
    A2 = "#d97706"  # amber: f₂ / a₂ vector
    B_COLOR = "#7c3aed"  # purple: NNLS scatter / constrained solution / b vector
    CLAMP = "#e11d48"  # rose: naive clamp (suboptimal)
    CONE = "#3b82f6"  # cone fill / feasible region
    DIM = "#64748b"  # axes, origin dot
    DARK = "#0f172a"  # population star markers / target y
    SPINE = "#e2e8f0"  # spine edges
    GRID = "#f1f5f9"  # grid lines
    return A1, A2, B_COLOR, CLAMP, CONE, DARK, DIM, GRID, SPINE


@app.cell
def motivation(mo):
    mo.md(r"""
    ## Combining Forecasts

    Suppose several models each forecast tomorrow's sales. Each captures genuine signal
    but none is perfect. A natural idea: learn a weighted sum

    $$\hat{y} = w_1 f_1 + w_2 f_2 + \cdots + w_k f_k$$

    and choose the weights by minimising squared error on historical data — ordinary least
    squares (OLS).

    This sounds sensible. But when forecasts are built from the same underlying drivers,
    they can be so correlated that OLS produces surprising — and practically useless — weights.
    """)
    return


@app.cell
def collinear_intro(mo):
    mo.md(r"""
    ### Two correlated forecasters

    Consider a simplified setting. The true signal is

    $$y = 2X_1 + X_2 + \varepsilon$$

    Two analysts each build a model, arriving at

    $$f_1 = 3X_1 + 2X_2, \qquad f_2 = X_1 + X_2$$

    Both forecasts carry genuine signal — individually, each is highly correlated with
    the target:

    $$\operatorname{Corr}(f_1, y)
      = \frac{\operatorname{Cov}(3X_1 + 2X_2,\; 2X_1 + X_2)}{\sqrt{\operatorname{Var}(f_1)\,\operatorname{Var}(y)}}
      = \frac{8}{\sqrt{65}} \approx 0.99$$

    $$\operatorname{Corr}(f_2, y)
      = \frac{\operatorname{Cov}(X_1 + X_2,\; 2X_1 + X_2)}{\sqrt{\operatorname{Var}(f_2)\,\operatorname{Var}(y)}}
      = \frac{3}{\sqrt{10}} \approx 0.95$$

    The problem is not with either forecast in isolation — it is that they are nearly
    collinear with *each other*:

    $$\operatorname{Corr}(f_1, f_2)
      = \frac{\operatorname{Cov}(3X_1 + 2X_2,\; X_1 + X_2)}{\sqrt{\operatorname{Var}(f_1)\,\operatorname{Var}(f_2)}}
      = \frac{5}{\sqrt{26}} \approx 0.98$$

    When OLS looks for $w_1, w_2$ such that

    $$(3w_1 + w_2)\,X_1 + (2w_1 + w_2)\,X_2 \approx 2X_1 + X_2$$

    matching coefficients gives $3w_1 + w_2 = 2$ and $2w_1 + w_2 = 1$, so
    $w_1 = 1$, $w_2 = -1$. OLS *subtracts* $f_2$ — not because $f_2$ is harmful,
    but because the near-cancellation $f_1 - f_2 = 2X_1 + X_2$ happens to equal
    the target exactly.

    A negative weight on a useful forecast is conceptually troubling. The deeper problem
    is statistical: near-collinearity makes $(F^\top F)$ nearly singular, blowing up
    $\operatorname{Var}(\hat{w}) = \sigma^2(F^\top F)^{-1}$. A small perturbation in the
    training sample can swing the estimated weights dramatically while barely changing the
    in-sample fit. Use the slider below to see how the OLS and NNLS weight estimates
    scatter across repeated samples of size $n$.
    """)
    return


@app.cell
def stability_slider(mo):
    n_samples_slider = mo.ui.slider(10, 500, step=10, value=30, label="n")
    return (n_samples_slider,)


@app.cell
def stability_helpers(A1, B_COLOR, CONE, DARK, DIM, GRID, SPINE, nnls, np):
    _N_TRIALS = 500
    _SCATTER_KW = {"alpha": 0.15, "s": 18, "linewidths": 0}


    def simulate_weights(n, rng, make_F):
        ols_w = np.zeros((_N_TRIALS, 2))
        nnls_w = np.zeros((_N_TRIALS, 2))
        for i in range(_N_TRIALS):
            X1 = rng.standard_normal(n)
            X2 = rng.standard_normal(n)
            y = 2 * X1 + X2 + rng.standard_normal(n)
            F = make_F(X1, X2)
            ols_w[i] = np.linalg.lstsq(F, y, rcond=None)[0]
            nnls_w[i] = nnls(F, y)[0]
        return ols_w, nnls_w


    def style_weight_ax(ax, xlim, ylim):
        xlo = max(0.0, xlim[0])
        ylo = max(0.0, ylim[0])
        if xlo < xlim[1] and ylo < ylim[1]:
            ax.fill_between([xlo, xlim[1]], ylo, ylim[1], color=CONE, alpha=0.06, zorder=0)
        ax.axhline(0, color=DIM, lw=0.7, alpha=0.5)
        ax.axvline(0, color=DIM, lw=0.7, alpha=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(SPINE)
            sp.set_linewidth(0.8)
        ax.tick_params(labelsize=8, length=3, width=0.6)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$w_1$", fontsize=11)
        ax.set_ylabel("$w_2$", fontsize=11)


    def render_weight_panels(
        axes,
        ols_w,
        nnls_w,
        xlim,
        ylim,
        pop_ols,
        ols_label,
        ols_pop_sol,
        pop_nnls,
        nnls_label,
        nnls_pop_sol,
    ):
        for ax in axes:
            style_weight_ax(ax, xlim, ylim)

        axes[0].scatter(ols_w[:, 0], ols_w[:, 1], color=A1, **_SCATTER_KW)
        axes[0].scatter([pop_ols[0]], [pop_ols[1]], color=DARK, s=120, zorder=6, marker="*", label=ols_label)
        axes[0].set_title(
            f"OLS   $\\sigma(w_1)={ols_w[:, 0].std():.2f},\\ \\sigma(w_2)={ols_w[:, 1].std():.2f}$\n{ols_pop_sol}",
            fontsize=10,
        )
        axes[0].legend(fontsize=9)

        axes[1].scatter(nnls_w[:, 0], nnls_w[:, 1], color=B_COLOR, **_SCATTER_KW)
        axes[1].scatter([pop_nnls[0]], [pop_nnls[1]], color=DARK, s=120, zorder=6, marker="*", label=nnls_label)
        axes[1].set_title(
            f"NNLS   $\\sigma(w_1)={nnls_w[:, 0].std():.2f},\\ \\sigma(w_2)={nnls_w[:, 1].std():.2f}$\n{nnls_pop_sol}",
            fontsize=10,
        )
        axes[1].legend(fontsize=9)

    return render_weight_panels, simulate_weights, style_weight_ax


@app.cell
def stability_collinear_data(n_samples_slider, np, simulate_weights):
    _rng = np.random.default_rng(0)
    collinear_ols_w, collinear_nnls_w = simulate_weights(
        n_samples_slider.value,
        _rng,
        make_F=lambda X1, X2: np.column_stack([3 * X1 + 2 * X2, X1 + X2]),
    )
    return collinear_nnls_w, collinear_ols_w


@app.cell
def stability_plot_ols(
    A1,
    DARK,
    collinear_ols_w,
    mo,
    n_samples_slider,
    plt,
    style_weight_ax,
):
    _fig, _ax = plt.subplots(figsize=(5.5, 5))
    style_weight_ax(_ax, xlim=(-0.5, 2.5), ylim=(-3.0, 1.0))
    _ax.scatter(collinear_ols_w[:, 0], collinear_ols_w[:, 1], color=A1, alpha=0.15, s=18, linewidths=0)
    _ax.scatter([1], [-1], color=DARK, s=120, zorder=6, marker="*", label=r"$(1,\ -1)$ population OLS")
    _ax.set_title(
        f"OLS   $\\sigma(w_1)={collinear_ols_w[:, 0].std():.2f},\\ \\sigma(w_2)={collinear_ols_w[:, 1].std():.2f}$\npop. solution: $2X_1 + X_2$",
        fontsize=10,
    )
    _ax.legend(fontsize=9)
    plt.tight_layout(pad=0.6)
    _controls = mo.vstack([mo.md(f"Sample size $n$: {n_samples_slider}")])
    mo.hstack([_controls, mo.mpl.interactive(_fig)], align="center", gap="0.5rem")
    return


@app.cell
def stability_plot_nnls(
    B_COLOR,
    DARK,
    collinear_nnls_w,
    mo,
    n_samples_slider,
    plt,
    style_weight_ax,
):
    _fig, _ax = plt.subplots(figsize=(5.5, 5))
    style_weight_ax(_ax, xlim=(-0.5, 2.5), ylim=(-3.0, 1.0))
    _ax.scatter(collinear_nnls_w[:, 0], collinear_nnls_w[:, 1], color=B_COLOR, alpha=0.15, s=18, linewidths=0)
    _ax.scatter([8 / 13], [0], color=DARK, s=120, zorder=6, marker="*", label=r"$(8/13,\ 0)$ population NNLS")
    _ax.set_title(
        f"NNLS   $\\sigma(w_1)={collinear_nnls_w[:, 0].std():.2f},\\ \\sigma(w_2)={collinear_nnls_w[:, 1].std():.2f}$\n"
        + f"pop. solution: ${24 / 13:.2f}X_1 + {16 / 13:.2f}X_2$",
        fontsize=10,
    )
    _ax.legend(fontsize=9)
    plt.tight_layout(pad=0.6)
    _controls = mo.vstack([mo.md(f"Sample size $n$: {n_samples_slider}")])
    mo.hstack([_controls, mo.mpl.interactive(_fig)], align="center", gap="0.5rem")
    return


@app.cell
def collinear_geometry_intro(mo):
    mo.md(r"""
    ### The geometry of instability

    The numerical scatter has a clean geometric interpretation. In *coefficient space* —
    where each axis represents the loading on a single predictor $X_i$ — each forecast
    $f_j$ is a vector from the origin. Non-negative weights confine the combined
    prediction to the cone

    $$\mathcal{C} = \{\,\lambda_1 f_1 + \lambda_2 f_2 \mid \lambda_1, \lambda_2 \geq 0\,\}$$

    Because $f_1 = (3, 2)$ and $f_2 = (1, 1)$ point in nearly the same direction, this
    cone is narrow. The target $y = (2, 1)$ lies just *outside* it — reachable only by
    subtracting $f_2$.

    NNLS finds the closest point on the cone boundary. This introduces a small bias
    (the projection), but eliminates the variance explosion that comes from allowing
    weights in the subtraction direction.
    """)
    return


@app.cell
def colspace_plot(
    A1,
    A2,
    B_COLOR,
    CONE,
    DARK,
    DIM,
    GRID,
    R,
    SPINE,
    cone_fill_polygon,
    draw_vector,
    mo,
    np,
    plt,
):
    # Coefficient vectors: f1 = 3X1 + 2X2, f2 = X1 + X2
    _f1 = np.array([3.0, 2.0])
    _f2 = np.array([1.0, 1.0])
    # OLS: w=(1,-1) → f1 - f2 = (2,1), which equals the true signal 2X1 + X2
    _ols = np.array([2.0, 1.0])
    # NNLS: w=(8/13, 0) → (8/13)*f1 — on the f1 ray, inside the cone
    _nnls = (8.0 / 13) * _f1

    _theta1 = np.arctan2(_f1[1], _f1[0])
    _theta2 = np.arctan2(_f2[1], _f2[0])
    _poly_x, _poly_y, _ray1, _ray2 = cone_fill_polygon(_theta1, _theta2, R)

    # Right-angle marker: OLS → NNLS is perpendicular to the f1 ray
    _f1_unit = _f1 / np.linalg.norm(_f1)
    _perp = np.array([_f1_unit[1], -_f1_unit[0]])
    if np.dot(_perp, _ols - _nnls) < 0:
        _perp = -_perp
    _sq = 0.08
    _sq_pts = np.array(
        [
            _nnls + _sq * _perp,
            _nnls + _sq * _perp + _sq * _f1_unit,
            _nnls + _sq * _f1_unit,
        ]
    )

    _fig, _ax = plt.subplots(figsize=(6, 5))
    for _sp in _ax.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)

    _ax.fill(_poly_x, _poly_y, color=CONE, alpha=0.10, zorder=1)
    _ray_kw = {"color": CONE, "lw": 0.9, "ls": "--", "alpha": 0.5, "zorder": 2}
    _ax.plot([0, R * np.cos(_ray1)], [0, R * np.sin(_ray1)], **_ray_kw)
    _ax.plot([0, R * np.cos(_ray2)], [0, R * np.sin(_ray2)], **_ray_kw)

    draw_vector(_ax, _f1, r"$f_1 = (3,2)$", A1, label_offset=0.22)
    draw_vector(_ax, _f2, r"$f_2 = (1,1)$", A2, label_offset=0.22)

    # Dashed line from OLS to NNLS with right-angle marker
    _ax.plot(
        [_ols[0], _nnls[0]],
        [_ols[1], _nnls[1]],
        ls="--",
        color=DIM,
        lw=1.0,
        alpha=0.7,
        zorder=3,
    )
    _ax.plot(_sq_pts[:, 0], _sq_pts[:, 1], color=DIM, lw=0.9, alpha=0.6, zorder=4)

    _ax.annotate(
        "",
        xy=tuple(_ols),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": DARK, "lw": 2.0, "mutation_scale": 16},
        zorder=4,
    )
    _ax.text(
        _ols[0] + 0.12, _ols[1], r"$y = f_1 - f_2$", fontsize=14, color=DARK, ha="left", va="center", fontweight="bold"
    )

    _ax.scatter(*_nnls, color=B_COLOR, s=60, zorder=7, linewidths=0)
    _ax.text(
        _nnls[0] + 0.14,
        _nnls[1] - 0.07,
        r"NNLS $= \frac{8}{13}f_1$",
        fontsize=10,
        color=B_COLOR,
        ha="left",
        va="bottom",
        fontweight="bold",
    )

    _ax.scatter([0], [0], color=DIM, s=16, zorder=6, linewidths=0)
    _ax.axhline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    _ax.axvline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    _ax.grid(True, color=GRID, lw=0.8, zorder=0)

    _ax.set_xlim(-0.3, 3.7)
    _ax.set_ylim(-0.3, 2.6)
    _ax.set_aspect("equal")
    _ax.tick_params(labelsize=8, length=3, width=0.6)
    _ax.set_xlabel("coefficient of $X_1$", fontsize=10)
    _ax.set_ylabel("coefficient of $X_2$", fontsize=10)
    _ax.set_title(
        r"Coefficient space: $f_1 - f_2$ (OLS) lands outside the cone",
        fontsize=10,
        pad=10,
    )
    plt.tight_layout(pad=0.6)

    mo.mpl.interactive(_fig)
    return


@app.cell
def orthogonal_intro(mo):
    mo.md(r"""
    ### What if the forecasts were orthogonal?

    The instability above is a consequence of the forecasts sharing almost the same
    structure. Suppose instead two analysts produce

    $$g_1 = X_1 + X_2, \qquad g_2 = X_1 - X_2$$

    Because $X_1 \perp X_2$,

    $$\operatorname{Cov}(g_1, g_2)
      = \operatorname{Cov}(X_1 + X_2,\, X_1 - X_2)
      = \operatorname{Var}(X_1) - \operatorname{Var}(X_2) = 0$$

    The forecasts are uncorrelated. $G^\top G$ is diagonal — each weight is estimated
    independently, with $\operatorname{Var}(\hat{w}_i) \approx \sigma^2 / (2n)$ rather
    than the inflated variance from near-singular $F^\top F$.

    Matching the target coefficients now gives

    $$w_1 + w_2 = 2, \quad w_1 - w_2 = 1
      \implies w_1 = \tfrac{3}{2},\; w_2 = \tfrac{1}{2}$$

    Both weights are non-negative: OLS lands *inside* the cone before any constraint
    is imposed, so NNLS agrees with OLS exactly.
    """)
    return


@app.cell
def stability_plot_orthogonal(
    mo,
    n_samples_slider,
    np,
    plt,
    render_weight_panels,
    simulate_weights,
):
    _rng = np.random.default_rng(0)
    _ols_w, _nnls_w = simulate_weights(
        n_samples_slider.value,
        _rng,
        make_F=lambda X1, X2: np.column_stack([X1 + X2, X1 - X2]),
    )
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5))
    render_weight_panels(
        _axes,
        _ols_w,
        _nnls_w,
        xlim=(0.5, 2.5),
        ylim=(-0.5, 1.5),
        pop_ols=(1.5, 0.5),
        ols_label=r"$(3/2,\ 1/2)$ pop. OLS",
        ols_pop_sol=r"pop. solution: $2X_1 + X_2$",
        pop_nnls=(1.5, 0.5),
        nnls_label=r"$(3/2,\ 1/2)$ pop. NNLS",
        nnls_pop_sol=r"pop. solution: $2X_1 + X_2$",
    )
    plt.tight_layout(pad=0.6)
    _controls = mo.vstack([mo.md(f"Sample size $n$: {n_samples_slider}")])
    mo.hstack([_controls, mo.mpl.interactive(_fig)], align="center", gap="0.5rem")
    return


@app.cell
def colspace_plot_orthogonal(
    A1,
    A2,
    B_COLOR,
    CONE,
    DARK,
    DIM,
    GRID,
    R,
    SPINE,
    cone_fill_polygon,
    draw_vector,
    mo,
    np,
    plt,
):
    # f1 = X1 + X2 → coeff vector (1, 1); f2 = X1 - X2 → coeff vector (1, -1)
    # Cov(f1, f2) = Var(X1) - Var(X2) = 0: the forecasts are orthogonal
    _f1 = np.array([1.0, 1.0])
    _f2 = np.array([1.0, -1.0])
    # OLS: w=(3/2, 1/2) → (3/2)*(1,1) + (1/2)*(1,-1) = (2, 1) = y
    # Both weights ≥ 0, so NNLS = OLS; no projection onto the cone boundary is needed
    _target = np.array([2.0, 1.0])

    _theta1 = np.arctan2(_f1[1], _f1[0])  # π/4  (45°)
    _theta2 = np.arctan2(_f2[1], _f2[0])  # −π/4 (−45°)
    _poly_x, _poly_y, _ray1, _ray2 = cone_fill_polygon(_theta1, _theta2, R)

    _fig, _ax = plt.subplots(figsize=(6, 6))
    for _sp in _ax.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)

    _ax.fill(_poly_x, _poly_y, color=CONE, alpha=0.10, zorder=1)
    _ray_kw = {"color": CONE, "lw": 0.9, "ls": "--", "alpha": 0.5, "zorder": 2}
    _ax.plot([0, R * np.cos(_ray1)], [0, R * np.sin(_ray1)], **_ray_kw)
    _ax.plot([0, R * np.cos(_ray2)], [0, R * np.sin(_ray2)], **_ray_kw)

    draw_vector(_ax, _f1, r"$f_1 = (1,1)$", A1, label_offset=0.22)
    draw_vector(_ax, _f2, r"$f_2 = (1,{-1})$", A2, label_offset=0.22)

    # Target arrow — OLS and NNLS land at the same point inside the cone
    _ax.annotate(
        "",
        xy=tuple(_target),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": DARK, "lw": 2.0, "mutation_scale": 16},
        zorder=4,
    )
    _ax.text(
        _target[0] + 0.12,
        _target[1] + 0.06,
        r"$y = \frac{3}{2}f_1 + \frac{1}{2}f_2$",
        fontsize=12,
        color=DARK,
        ha="left",
        va="bottom",
        fontweight="bold",
    )

    _ax.scatter(*_target, color=B_COLOR, s=80, zorder=7, linewidths=0, marker="*")
    _ax.text(
        _target[0] + 0.12,
        _target[1] - 0.10,
        r"OLS $=$ NNLS",
        fontsize=10,
        color=B_COLOR,
        ha="left",
        va="top",
        fontweight="bold",
    )

    _ax.scatter([0], [0], color=DIM, s=16, zorder=6, linewidths=0)
    _ax.axhline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    _ax.axvline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    _ax.grid(True, color=GRID, lw=0.8, zorder=0)

    _ax.set_xlim(-0.4, 3.4)
    _ax.set_ylim(-1.5, 2.3)
    _ax.set_aspect("equal")
    _ax.tick_params(labelsize=8, length=3, width=0.6)
    _ax.set_xlabel("coefficient of $X_1$", fontsize=10)
    _ax.set_ylabel("coefficient of $X_2$", fontsize=10)
    _ax.set_title(
        r"Coefficient space: $y$ lands inside the cone — OLS $=$ NNLS",
        fontsize=10,
        pad=10,
    )
    plt.tight_layout(pad=0.6)

    mo.mpl.interactive(_fig)
    return


@app.cell
def efficiency_intro(mo):
    mo.md(r"""
    ### Making each forecast efficient

    Orthogonality is the property that matters for numerical stability — given orthogonal
    forecasts, OLS correctly recovers the weights $(3/2, 1/2)$ without any instability.
    So why would an analyst go further and also make their forecast efficient?

    The answer is about **interpretability of the weight they receive from the combiner**.

    A forecast $f$ is **efficient** if regressing the target on $f$ alone gives slope 1:

    $$\beta = \frac{\operatorname{Cov}(y,\, f)}{\operatorname{Var}(f)} = 1$$

    When $\beta = 1$, the forecast is already at the right scale — a weight of $w$ from
    the combiner means the combination is using exactly $w$ times your forecast's face
    value. When $\beta \neq 1$, the weight and your actual signal contribution are
    entangled: a weight of $w$ from the combiner represents $w \cdot \beta$ worth of
    signal, so you cannot read off your contribution from the weight alone without also
    knowing your own $\beta$.

    Consider the new analyst's position: they have submitted a forecast to a shared
    combination system, and the system reports back a weight. If their forecast is
    efficient, the weight is immediately meaningful — "the system is using 70% of my
    forecast." If it is not, the weight is a number that requires further context to
    interpret.

    For the orthogonal pair $g_1 = X_1 + X_2$, $g_2 = X_1 - X_2$, the population slopes
    are

    $$\beta_1 = \frac{\operatorname{Cov}(y,\, g_1)}{\operatorname{Var}(g_1)}
              = \frac{3}{2}, \qquad
      \beta_2 = \frac{\operatorname{Cov}(y,\, g_2)}{\operatorname{Var}(g_2)}
              = \frac{1}{2}$$

    Rescaling each by its own $\beta$ gives the efficient versions

    $$g_1^* = \tfrac{3}{2}\,g_1 = \tfrac{3}{2}X_1 + \tfrac{3}{2}X_2, \qquad
      g_2^* = \tfrac{1}{2}\,g_2 = \tfrac{1}{2}X_1 - \tfrac{1}{2}X_2$$

    Scaling preserves orthogonality, so $\operatorname{Cov}(g_1^*,\, g_2^*) = 0$ still
    holds. The combination weights are now both 1: each analyst's forecast enters the
    combination at face value, and the weight the combiner returns has a direct meaning
    — a weight of 1 means "use my forecast as-is," and any deviation from 1 signals
    that the system is either trusting or discounting it relative to face value.
    """)
    return


@app.cell
def stability_plot_efficient(
    mo,
    n_samples_slider,
    np,
    plt,
    render_weight_panels,
    simulate_weights,
):
    _rng = np.random.default_rng(0)
    _ols_w, _nnls_w = simulate_weights(
        n_samples_slider.value,
        _rng,
        make_F=lambda X1, X2: np.column_stack([(3 / 2) * (X1 + X2), (1 / 2) * (X1 - X2)]),
    )
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5))
    render_weight_panels(
        _axes,
        _ols_w,
        _nnls_w,
        xlim=(0.3, 1.7),
        ylim=(-0.5, 2.5),
        pop_ols=(1, 1),
        ols_label=r"$(1,\ 1)$ pop. OLS",
        ols_pop_sol=r"pop. solution: $g_1^* + g_2^*\; (= y)$",
        pop_nnls=(1, 1),
        nnls_label=r"$(1,\ 1)$ pop. NNLS",
        nnls_pop_sol=r"pop. solution: $g_1^* + g_2^*\; (= y)$",
    )
    plt.tight_layout(pad=0.6)
    _controls = mo.vstack([mo.md(f"Sample size $n$: {n_samples_slider}")])
    mo.hstack([_controls, mo.mpl.interactive(_fig)], align="center", gap="0.5rem")
    return


@app.cell
def colspace_plot_efficient(
    A1,
    A2,
    B_COLOR,
    CONE,
    DARK,
    DIM,
    GRID,
    R,
    SPINE,
    cone_fill_polygon,
    draw_vector,
    mo,
    np,
    plt,
):
    # Efficient orthogonal forecasts: g1* = (3/2)(X1+X2), g2* = (1/2)(X1-X2)
    # Same directions as before (45° and −45°), but at the right scales for unit weights
    _g1s = np.array([3 / 2, 3 / 2])  # g1* coefficient vector
    _g2s = np.array([1 / 2, -1 / 2])  # g2* coefficient vector
    _target = np.array([2.0, 1.0])  # y = g1* + g2* exactly

    _theta1 = np.arctan2(_g1s[1], _g1s[0])  # π/4
    _theta2 = np.arctan2(_g2s[1], _g2s[0])  # −π/4
    _poly_x, _poly_y, _ray1, _ray2 = cone_fill_polygon(_theta1, _theta2, R)

    _fig, _ax = plt.subplots(figsize=(6, 6))
    for _sp in _ax.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)

    _ax.fill(_poly_x, _poly_y, color=CONE, alpha=0.10, zorder=1)
    _ray_kw = {"color": CONE, "lw": 0.9, "ls": "--", "alpha": 0.5, "zorder": 2}
    _ax.plot([0, R * np.cos(_ray1)], [0, R * np.sin(_ray1)], **_ray_kw)
    _ax.plot([0, R * np.cos(_ray2)], [0, R * np.sin(_ray2)], **_ray_kw)

    draw_vector(_ax, _g1s, r"$g_1^*$", A1, label_offset=0.20)
    draw_vector(_ax, _g2s, r"$g_2^*$", A2, label_offset=0.18)

    # Tip-to-tail: g1* from origin, then g2* from its tip — lands exactly on y
    _comp_kw = {"mutation_scale": 13, "lw": 1.6, "alpha": 0.40}
    _ax.annotate(
        "",
        xy=tuple(_g1s),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": A1, **_comp_kw},
        zorder=4,
    )
    _ax.annotate(
        "",
        xy=tuple(_target),
        xytext=tuple(_g1s),
        arrowprops={"arrowstyle": "-|>", "color": A2, **_comp_kw},
        zorder=4,
    )
    _ax.scatter(*_g1s, color=DIM, s=22, zorder=6, linewidths=0)

    # Target vector y = g1* + g2* (unit weights)
    _ax.annotate(
        "",
        xy=tuple(_target),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": DARK, "lw": 2.0, "mutation_scale": 16},
        zorder=5,
    )
    _ax.text(
        _target[0] + 0.11,
        _target[1] + 0.06,
        r"$y = g_1^* + g_2^*$",
        fontsize=12,
        color=DARK,
        ha="left",
        va="bottom",
        fontweight="bold",
    )

    _ax.scatter(*_target, color=B_COLOR, s=80, zorder=7, linewidths=0, marker="*")
    _ax.text(
        _target[0] + 0.11,
        _target[1] - 0.10,
        r"OLS $=$ NNLS, weights $= (1,\,1)$",
        fontsize=9,
        color=B_COLOR,
        ha="left",
        va="top",
        fontweight="bold",
    )

    _ax.scatter([0], [0], color=DIM, s=16, zorder=6, linewidths=0)
    _ax.axhline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    _ax.axvline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    _ax.grid(True, color=GRID, lw=0.8, zorder=0)

    _ax.set_xlim(-0.3, 2.7)
    _ax.set_ylim(-0.8, 2.2)
    _ax.set_aspect("equal")
    _ax.tick_params(labelsize=8, length=3, width=0.6)
    _ax.set_xlabel("coefficient of $X_1$", fontsize=10)
    _ax.set_ylabel("coefficient of $X_2$", fontsize=10)
    _ax.set_title(
        r"Efficient forecasts: $g_1^* + g_2^* = y$ exactly — unit weights",
        fontsize=10,
        pad=10,
    )
    plt.tight_layout(pad=0.6)

    mo.mpl.interactive(_fig)
    return


@app.cell
def efficiency_analyst_motivation(mo):
    mo.md(r"""
    From the analyst's perspective, submitting an efficient forecast is a form of
    **calibration before entering the room**. The NNLS combiner sees only the forecast
    values, not the analyst's internal β. Whatever weight it assigns is the only feedback
    the analyst receives. If the forecast is efficient, that weight is immediately
    interpretable: a weight of 0.8 means the combiner is using 80% of the forecast at
    face value, and the non-negativity constraint has a natural reading — the combiner
    will never short a forecast whose direction is already correct.

    If the forecast is *not* efficient, the weight carries no direct meaning without also
    knowing β. A weight of 2 from the combiner might mean "this forecast is trusted" or
    it might be compensating for an under-scaled forecast. The NNLS constraint is still
    applied correctly, but the output is harder to audit or explain.
    """)
    return


@app.cell
def efficient_nonorthogonal_intro(mo):
    mo.md(r"""
    ### Efficient but correlated forecasts

    Efficiency alone does not guarantee unit combination weights. Suppose instead of
    $g_2^* = \tfrac{1}{2}X_1 - \tfrac{1}{2}X_2$, a second analyst submits

    $$h_1 = \tfrac{3}{2}X_1 + \tfrac{3}{2}X_2, \qquad
      h_2 = \tfrac{3}{2}X_1 - \tfrac{1}{2}X_2$$

    Both are still efficient — the univariate slope of $y$ on each is 1:

    $$\beta_1 = \frac{\operatorname{Cov}(y,\, h_1)}{\operatorname{Var}(h_1)}
      = \frac{9/2}{9/2} = 1, \qquad
      \beta_2 = \frac{\operatorname{Cov}(y,\, h_2)}{\operatorname{Var}(h_2)}
      = \frac{5/2}{5/2} = 1$$

    But now they are *correlated*:

    $$\operatorname{Cov}(h_1,\, h_2)
      = \tfrac{3}{2}\cdot\tfrac{3}{2} + \tfrac{3}{2}\cdot(-\tfrac{1}{2})
      = \tfrac{9}{4} - \tfrac{3}{4} = \tfrac{3}{2} \neq 0$$

    Because both forecasts load heavily on $X_1$, their sum $h_1 + h_2 = 3X_1 + X_2$
    over-represents $X_1$ relative to $y = 2X_1 + X_2$. To recover $y$ exactly the
    combiner must use weights $(5/6,\, 1/2)$, not $(1,\, 1)$:

    $$\tfrac{5}{6}\,h_1 + \tfrac{1}{2}\,h_2 = \tfrac{5}{6}\!\left(\tfrac{3}{2}X_1 + \tfrac{3}{2}X_2\right) + \tfrac{1}{2}\!\left(\tfrac{3}{2}X_1 - \tfrac{1}{2}X_2\right) = 2X_1 + X_2 = y$$

    The weights deviate from 1 to compensate for the double-counting of the shared
    $X_1$ component. Efficiency guarantees an interpretable *univariate* slope of 1,
    but only orthogonality guarantees unit *combination* weights.
    """)
    return


@app.cell
def colspace_plot_efficient_nonorthogonal(
    A1,
    A2,
    B_COLOR,
    CONE,
    DARK,
    DIM,
    GRID,
    R,
    SPINE,
    cone_fill_polygon,
    draw_vector,
    mo,
    np,
    plt,
):
    # h1 = (3/2)X1 + (3/2)X2: same as g1*, efficient (slope 1 with y)
    # h2 = (3/2)X1 - (1/2)X2: efficient but NOT orthogonal to h1
    # Cov(h1, h2) = 9/4 - 3/4 = 3/2  (both load on X1)
    # OLS = NNLS = (5/6, 1/2) — not (1, 1)
    _h1 = np.array([3 / 2, 3 / 2])
    _h2 = np.array([3 / 2, -1 / 2])
    _w1, _w2 = 5 / 6, 1 / 2
    _mid = _w1 * _h1          # (5/4, 5/4) — tip of first component arrow
    _target = _mid + _w2 * _h2  # (2, 1) = y

    _theta1 = np.arctan2(_h1[1], _h1[0])   # 45°
    _theta2 = np.arctan2(_h2[1], _h2[0])   # -18.4°
    _poly_x, _poly_y, _ray1, _ray2 = cone_fill_polygon(_theta1, _theta2, R)

    _fig, _ax = plt.subplots(figsize=(6, 6))
    for _sp in _ax.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)

    _ax.fill(_poly_x, _poly_y, color=CONE, alpha=0.10, zorder=1)
    _ray_kw = {"color": CONE, "lw": 0.9, "ls": "--", "alpha": 0.5, "zorder": 2}
    _ax.plot([0, R * np.cos(_ray1)], [0, R * np.sin(_ray1)], **_ray_kw)
    _ax.plot([0, R * np.cos(_ray2)], [0, R * np.sin(_ray2)], **_ray_kw)

    draw_vector(_ax, _h1, r"$h_1$", A1, label_offset=0.20)
    draw_vector(_ax, _h2, r"$h_2$", A2, label_offset=0.20)

    # Tip-to-tail: (5/6)h1 from origin, then (1/2)h2 from its tip
    _comp_kw = {"mutation_scale": 13, "lw": 1.6, "alpha": 0.40}
    _ax.annotate(
        "",
        xy=tuple(_mid),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": A1, **_comp_kw},
        zorder=4,
    )
    _ax.annotate(
        "",
        xy=tuple(_target),
        xytext=tuple(_mid),
        arrowprops={"arrowstyle": "-|>", "color": A2, **_comp_kw},
        zorder=4,
    )
    _ax.scatter(*_mid, color=DIM, s=22, zorder=6, linewidths=0)

    _ax.annotate(
        "",
        xy=tuple(_target),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": DARK, "lw": 2.0, "mutation_scale": 16},
        zorder=5,
    )
    _ax.text(
        _target[0] + 0.11,
        _target[1] + 0.06,
        r"$y = \frac{5}{6}h_1 + \frac{1}{2}h_2$",
        fontsize=12,
        color=DARK,
        ha="left",
        va="bottom",
        fontweight="bold",
    )

    _ax.scatter(*_target, color=B_COLOR, s=80, zorder=7, linewidths=0, marker="*")
    _ax.text(
        _target[0] + 0.11,
        _target[1] - 0.10,
        r"OLS $=$ NNLS, weights $= (5/6,\,1/2)$",
        fontsize=9,
        color=B_COLOR,
        ha="left",
        va="top",
        fontweight="bold",
    )

    _ax.scatter([0], [0], color=DIM, s=16, zorder=6, linewidths=0)
    _ax.axhline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    _ax.axvline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    _ax.grid(True, color=GRID, lw=0.8, zorder=0)

    _ax.set_xlim(-0.3, 2.7)
    _ax.set_ylim(-0.8, 2.2)
    _ax.set_aspect("equal")
    _ax.tick_params(labelsize=8, length=3, width=0.6)
    _ax.set_xlabel("coefficient of $X_1$", fontsize=10)
    _ax.set_ylabel("coefficient of $X_2$", fontsize=10)
    _ax.set_title(
        r"Efficient but correlated: $y$ inside cone — weights $(5/6,\,1/2) \neq (1,\,1)$",
        fontsize=10,
        pad=10,
    )
    plt.tight_layout(pad=0.6)

    mo.mpl.interactive(_fig)
    return


@app.cell
def conclusion(mo):
    mo.md(r"""
    ### Putting it together

    The two scenarios reveal a clean hierarchy:

    | Setting | OLS stable? | OLS in cone? | NNLS benefit |
    |---------|-------------|--------------|--------------|
    | Correlated | No — $\sigma(\hat{w}) \gg 1$ | No | Projects to cone; small bias, large variance reduction |
    | Orthogonal forecasts | Yes — $\sigma(\hat{w}) \approx 1/\sqrt{2n}$ | Yes | No active constraint; OLS $=$ NNLS |

    The practical recipe: **orthogonalise before combining**. A Gram–Schmidt pass (or
    any decorrelation transform such as PCA) removes cross-forecast collinearity, making
    $G^\top G$ well-conditioned. OLS then finds stable, non-negative weights without
    needing the NNLS constraint to rescue it.

    NNLS still provides a useful safety net — if a weight slips negative due to
    finite-sample noise, the constraint clips it to zero. But the constraint is rarely
    active when forecasts are decorrelated, and the fundamental variance problem that
    collinearity creates does not arise in the first place.
    """)
    return


@app.cell
def nnls_algo_intro(mo):
    mo.md(r"""
    ### Solving NNLS

    The NNLS objective is an ordinary quadratic in weight space:

    $$L(w) = \tfrac{1}{2}\lVert y - Fw \rVert^2
      = \tfrac{1}{2}\Bigl(w^\top (F^\top F)\, w - 2(F^\top y)^\top w + \lVert y \rVert^2\Bigr)$$

    In large samples, $\frac{1}{n} F^\top F \to \mathbb{E}[f f^\top]$ and
    $\frac{1}{n} F^\top y \to \mathbb{E}[fy]$. Because $X_1$ and $X_2$ are independent
    with unit variance, these expectations reduce to dot products of the coefficient vectors:

    $$f_1 = 3X_1 + 2X_2, \qquad f_2 = X_1 + X_2, \qquad y = 2X_1 + X_2$$

    $$\frac{F^\top F}{n} \;\longrightarrow\;
      \begin{pmatrix} 3 \cdot 3 + 2 \cdot 2 & 3 \cdot 1 + 2 \cdot 1 \\
                      3 \cdot 1 + 2 \cdot 1 & 1 \cdot 1 + 1 \cdot 1 \end{pmatrix}
      = \begin{pmatrix} 13 & 5 \\ 5 & 2 \end{pmatrix}$$

    $$\frac{F^\top y}{n} \;\longrightarrow\;
      \begin{pmatrix} 3 \cdot 2 + 2 \cdot 1 \\ 1 \cdot 2 + 1 \cdot 1 \end{pmatrix}
      = \begin{pmatrix} 8 \\ 3 \end{pmatrix}$$

    The OLS solution is $w^* = (F^\top F)^{-1} F^\top y = (1,\,-1)$.
    The eigenvalues of $F^\top F$ span a ratio of roughly 200:1, so the level curves of $L$
    are extremely elongated ellipses. Tiny perturbations in the training data shift the OLS
    weights dramatically while barely moving the in-sample fit.

    The non-negativity constraint $w \geq 0$ restricts feasible solutions to the first quadrant.
    Since $w^* = (1, -1)$ violates $w_2 \geq 0$, the constrained minimum must lie on the
    boundary $w_2 = 0$, $w_1 \geq 0$.
    """)
    return


@app.cell
def level_sets_ols_plot(A1, CONE, DIM, GRID, SPINE, loss_grid, mo, np, plt):
    _A = np.array([[3.0, 1.0], [2.0, 1.0]])
    _b = np.array([2.0, 1.0])

    _w1 = np.linspace(-0.5, 2.5, 500)
    _w2 = np.linspace(-3.2, 1.2, 500)
    _W1, _W2 = np.meshgrid(_w1, _w2)
    _L = loss_grid(_A, _b, _W1, _W2)

    _fig, _ax = plt.subplots(figsize=(7, 6))
    for _sp in _ax.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)

    _levels = [0.01, 0.04, 0.125, 0.4, 1.25, 4.0]
    _cs = _ax.contour(_W1, _W2, _L, levels=_levels, colors=DIM, linewidths=0.9, alpha=0.7)
    _ax.clabel(_cs, fmt=lambda v: f"{v:.2g}", fontsize=8, inline_spacing=4)

    _ax.fill_between([0.0, 2.5], 0.0, 1.2, color=CONE, alpha=0.08, zorder=0)
    _ax.axhline(0.0, color=DIM, lw=0.6, alpha=0.5)
    _ax.axvline(0.0, color=DIM, lw=0.6, alpha=0.5)
    _ax.plot([0.0, 2.5], [0.0, 0.0], color=CONE, lw=2.2, zorder=3, label=r"boundary $w_2 = 0,\; w_1 \geq 0$")

    _ax.scatter(1.0, -1.0, color=A1, s=220, zorder=6, marker="*", label=r"OLS $w^* = (1,\,-1)$")

    _ax.grid(True, color=GRID, lw=0.8, zorder=0)
    _ax.set_xlim(-0.5, 2.5)
    _ax.set_ylim(-3.2, 1.2)
    _ax.tick_params(labelsize=8, length=3, width=0.6)
    _ax.set_xlabel("$w_1$", fontsize=12)
    _ax.set_ylabel("$w_2$", fontsize=12)
    _ax.set_title(
        r"Level curves of $\frac{1}{2}\| Aw - b \|^2$: OLS minimum lies outside the feasible region",
        fontsize=10,
        pad=10,
    )
    _ax.legend(fontsize=9)
    plt.tight_layout(pad=0.6)
    mo.mpl.interactive(_fig)
    return


@app.cell
def clamping_md(mo):
    mo.md(r"""
    ### Why not just clamp negative weights to zero?

    A natural first instinct: take the OLS solution $(1, -1)$ and set
    $w_2 \leftarrow \max(0,\, w_2) = 0$, giving the clamped solution $(1, 0)$.

    The level sets reveal why this is suboptimal. Clamping moves straight upward from
    $(1, -1)$ to $(1, 0)$, hitting the constraint boundary but not at its minimum. The
    true NNLS solution also re-optimises $w_1$ to find the lowest point on $w_2 = 0$,
    landing at $(8/13, 0) \approx (0.62, 0)$.

    | Solution | $w$ | $L(w)$ |
    |----------|-----|--------|
    | OLS | $(1,\;-1)$ | $0$ |
    | NNLS | $(8/13,\;0)$ | $1/26 \approx 0.038$ |
    | Naive clamp | $(1,\;0)$ | $1.0$ |

    The clamped loss is 26 times larger than NNLS. Clamping only corrects the sign of the
    violated weight; NNLS re-solves the reduced problem on the active boundary.
    """)
    return


@app.cell
def level_sets_clamp_plot(
    A1,
    B_COLOR,
    CLAMP,
    CONE,
    DIM,
    GRID,
    SPINE,
    loss_grid,
    mo,
    np,
    plt,
):
    _A = np.array([[3.0, 1.0], [2.0, 1.0]])
    _b = np.array([2.0, 1.0])

    _w1b = np.linspace(-0.5, 2.5, 500)
    _w2b = np.linspace(-3.2, 1.2, 500)
    _W1b, _W2b = np.meshgrid(_w1b, _w2b)
    _Lb = loss_grid(_A, _b, _W1b, _W2b)

    _fig2, _ax2 = plt.subplots(figsize=(7, 6))
    for _sp in _ax2.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)

    _ax2.contour(_W1b, _W2b, _Lb, levels=[0.01, 0.04, 0.125, 0.4, 1.25, 4.0], colors=DIM, linewidths=0.7, alpha=0.35)
    _ax2.contour(_W1b, _W2b, _Lb, levels=[1.0 / 26], colors=B_COLOR, linewidths=1.5, linestyles="--", alpha=0.85)
    _ax2.contour(_W1b, _W2b, _Lb, levels=[1.0], colors=CLAMP, linewidths=1.5, linestyles="--", alpha=0.85)

    _ax2.fill_between([0.0, 2.5], 0.0, 1.2, color=CONE, alpha=0.08, zorder=0)
    _ax2.axhline(0.0, color=DIM, lw=0.6, alpha=0.5)
    _ax2.axvline(0.0, color=DIM, lw=0.6, alpha=0.5)
    _ax2.plot([0.0, 2.5], [0.0, 0.0], color=CONE, lw=2.2, zorder=3)

    _ax2.scatter(1.0, -1.0, color=A1, s=220, zorder=6, marker="*", label=r"OLS $(1,\,-1)$,  $L = 0$")
    _ax2.annotate(
        "",
        xy=(1.0, 0.0),
        xytext=(1.0, -1.0),
        arrowprops={"arrowstyle": "-|>", "color": CLAMP, "lw": 1.8, "mutation_scale": 13},
        zorder=5,
    )
    # _ax2.text(1.08, -0.5, r"clamp $w_2 \to 0$", fontsize=9, color=CLAMP, ha="left", va="center")
    _ax2.scatter(1.0, 0.0, color=CLAMP, s=60, zorder=7, marker="D", label=r"clamped $(1,\;0)$,  $L = 1$")
    _ax2.scatter(
        8.0 / 13, 0.0, color=B_COLOR, s=200, zorder=7, marker="*", label=rf"NNLS $({8 / 13:.2f},\;0)$,  $L = 1/26$"
    )

    _ax2.grid(True, color=GRID, lw=0.8, zorder=0)
    _ax2.set_xlim(-0.5, 2.5)
    _ax2.set_ylim(-3.2, 1.2)
    _ax2.tick_params(labelsize=8, length=3, width=0.6)
    _ax2.set_xlabel("$w_1$", fontsize=12)
    _ax2.set_ylabel("$w_2$", fontsize=12)
    _ax2.set_title(
        r"Naive clamp vs NNLS: the constrained optimum lies further along $w_2 = 0$",
        fontsize=10,
        pad=10,
    )
    _ax2.legend(fontsize=9)
    plt.tight_layout(pad=0.6)
    mo.mpl.interactive(_fig2)
    return


@app.cell
def forecast_clamp_md(mo):
    mo.md(r"""
    The same gap shows up in coefficient space. The clamped forecast $(w_1, w_2) = (1, 0)$
    places all weight on $f_1$, so its prediction lands at the tip of the $f_1$ vector
    $(3, 2)$, well past the target. NNLS pulls $w_1$ back to $8/13$, sliding along the
    $f_1$ ray to $(24/13,\, 16/13)$. That point is the orthogonal projection of $y$ onto
    the $f_1$ ray, the closest reachable point in the cone.
    """)
    return


@app.cell
def forecast_clamp_plot(
    A1,
    A2,
    B_COLOR,
    CLAMP,
    CONE,
    DARK,
    DIM,
    GRID,
    R,
    SPINE,
    cone_fill_polygon,
    draw_vector,
    mo,
    np,
    plt,
):
    _f1 = np.array([3.0, 2.0])
    _f2 = np.array([1.0, 1.0])
    _target = np.array([2.0, 1.0])
    _clamped_fc = _f1.copy()  # w=(1,0) → 1·f₁ + 0·f₂ = (3,2)
    _nnls_fc = (8.0 / 13) * _f1  # w=(8/13,0) → (24/13, 16/13)

    _theta1 = np.arctan2(_f1[1], _f1[0])
    _theta2 = np.arctan2(_f2[1], _f2[0])
    _poly_x, _poly_y, _ray1, _ray2 = cone_fill_polygon(_theta1, _theta2, R)

    _fig3, _ax3 = plt.subplots(figsize=(6, 6))
    for _sp in _ax3.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)

    _ax3.fill(_poly_x, _poly_y, color=CONE, alpha=0.10, zorder=1)
    _ray_kw = {"color": CONE, "lw": 0.9, "ls": "--", "alpha": 0.5, "zorder": 2}
    _ax3.plot([0, R * np.cos(_ray1)], [0, R * np.sin(_ray1)], **_ray_kw)
    _ax3.plot([0, R * np.cos(_ray2)], [0, R * np.sin(_ray2)], **_ray_kw)

    draw_vector(_ax3, _f1, r"$f_1$", A1, label_offset=0.22)
    draw_vector(_ax3, _f2, r"$f_2$", A2, label_offset=0.22)

    # Target y
    _ax3.annotate(
        "",
        xy=tuple(_target),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": DARK, "lw": 2.0, "mutation_scale": 16},
        zorder=4,
    )
    _ax3.text(
        _target[0] - 0.16, _target[1] + 0.10, r"$y$", fontsize=13, color=DARK, ha="right", va="bottom", fontweight="bold"
    )

    # Right-angle marker at the NNLS projection
    _f1_unit = _f1 / np.linalg.norm(_f1)
    _perp_dir = _target - _nnls_fc
    _perp_dir = _perp_dir / np.linalg.norm(_perp_dir)
    _sq = 0.08
    _sq_pts = np.array(
        [
            _nnls_fc + _sq * _perp_dir,
            _nnls_fc + _sq * _perp_dir + _sq * _f1_unit,
            _nnls_fc + _sq * _f1_unit,
        ]
    )
    _ax3.plot(
        np.append(_sq_pts[:, 0], _sq_pts[0, 0]),
        np.append(_sq_pts[:, 1], _sq_pts[0, 1]),
        color=DIM,
        lw=0.9,
        alpha=0.7,
        zorder=3,
    )

    # Dotted distance lines from y to each candidate
    _ax3.plot([_target[0], _nnls_fc[0]], [_target[1], _nnls_fc[1]], color=B_COLOR, lw=1.3, ls=":", zorder=3, alpha=0.8)
    _ax3.plot(
        [_target[0], _clamped_fc[0]],
        [_target[1], _clamped_fc[1]],
        color=CLAMP,
        lw=1.3,
        ls=":",
        zorder=3,
        alpha=0.8,
    )

    # Solution markers
    _ax3.scatter(*_clamped_fc, color=CLAMP, s=60, zorder=7, marker="D", label=r"clamped: $(3,\;2) = f_1$")
    _ax3.scatter(*_nnls_fc, color=B_COLOR, s=180, zorder=7, marker="*", label=r"NNLS: $(24/13,\;16/13)$")

    _ax3.scatter([0], [0], color=DIM, s=16, zorder=6, linewidths=0)
    _ax3.axhline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    _ax3.axvline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    _ax3.grid(True, color=GRID, lw=0.8, zorder=0)

    _ax3.set_xlim(-0.4, 3.8)
    _ax3.set_ylim(-0.5, 2.8)
    _ax3.set_aspect("equal")
    _ax3.tick_params(labelsize=8, length=3, width=0.6)
    _ax3.set_xlabel("coefficient of $X_1$", fontsize=10)
    _ax3.set_ylabel("coefficient of $X_2$", fontsize=10)
    _ax3.set_title(
        r"Clamped vs NNLS: $y$ projected onto the nearest cone face",
        fontsize=10,
        pad=10,
    )
    _ax3.legend(fontsize=8, loc="upper left")
    plt.tight_layout(pad=0.6)
    mo.mpl.interactive(_fig3)
    return


@app.cell
def pgd_intro(mo):
    mo.md(r"""
    ### Method 1: Projected Gradient Descent

    The simplest NNLS algorithm alternates two operations at each step:

    1. **Gradient step** — move in the steepest-descent direction

       $\tilde{w} = w_t - \eta \,\nabla L(w_t) = w_t - \eta \,(F^\top F\,w_t - F^\top y)$
    2. **Project** — clip any negative weight back to zero

       $w_{t+1} = \max(\tilde{w},\; 0)$

    Provided the step size satisfies $\eta < 2/\lambda_{\max}(F^\top F) \approx 0.134$,
    the iterates converge to the NNLS solution.

    Starting from the origin $(0, 0)$ — a feasible point, since both weights are
    non-negative — the iterates initially move freely through the interior of the feasible
    region. As they approach the solution, the gradient keeps pushing $w_2$ below zero,
    and the projection clips it back to the boundary. The sequence converges to the NNLS
    solution at $(8/13,\, 0)$.
    """)
    return


@app.cell
def pgd_helpers(np):
    def loss_grid(A, b, W1, W2):
        """Evaluate 0.5*||A @ w - b||^2 at every point of a (W1, W2) meshgrid."""
        W = np.stack([W1, W2], axis=-1)  # (ny, nx, 2)
        resid = W @ A.T - b              # (ny, nx, m)
        return 0.5 * (resid ** 2).sum(axis=-1)

    def projected_gradient_descent(A, b, w0, eta, n_iters):
        """Run projected gradient descent on 0.5*||Aw - b||^2, clipping to w >= 0."""
        w = w0.copy()
        traj = [w.copy()]
        unconstrained = []
        for _ in range(n_iters):
            grad = A.T @ (A @ w - b)
            w_tilde = w - eta * grad
            unconstrained.append(w_tilde.copy())
            w = np.maximum(w_tilde, 0.0)
            traj.append(w.copy())
        return np.array(traj), np.array(unconstrained)

    def gradient_descent(A, b, w0, eta, n_iters):
        """Run unconstrained gradient descent on 0.5*||Aw - b||^2."""
        w = w0.copy()
        traj = [w.copy()]
        for _ in range(n_iters):
            w = w - eta * (A.T @ (A @ w - b))
            traj.append(w.copy())
        return np.array(traj)

    return gradient_descent, loss_grid, projected_gradient_descent


@app.cell
def pgd_params():
    pgd_eta = 0.13
    pgd_n_iters = 40
    return pgd_eta, pgd_n_iters


@app.cell
def gd_plot(
    A1,
    B_COLOR,
    CONE,
    DARK,
    DIM,
    GRID,
    LineCollection,
    LinearSegmentedColormap,
    SPINE,
    gradient_descent,
    loss_grid,
    mo,
    np,
    pgd_eta,
    pgd_n_iters,
    plt,
):
    _A = np.array([[3.0, 1.0], [2.0, 1.0]])
    _b = np.array([2.0, 1.0])

    _traj_gd = gradient_descent(_A, _b, w0=np.array([0.0, 0.0]), eta=pgd_eta, n_iters=pgd_n_iters)

    _w1g = np.linspace(-0.5, 1.5, 400)
    _w2g = np.linspace(-1.5, 1.0, 400)
    _W1, _W2 = np.meshgrid(_w1g, _w2g)
    _L = loss_grid(_A, _b, _W1, _W2)

    _fig_gd, _ax_gd = plt.subplots(figsize=(6.5, 6.5))
    for _sp in _ax_gd.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)

    _ax_gd.contour(_W1, _W2, _L, levels=[0.01, 0.04, 0.125, 0.4, 1.25, 4.0], colors=DIM, linewidths=0.7, alpha=0.35)
    _ax_gd.contour(_W1, _W2, _L, levels=[1.0 / 26], colors=B_COLOR, linewidths=1.3, linestyles="--", alpha=0.7)
    _ax_gd.fill_between([0.0, 1.5], 0.0, 1.0, color=CONE, alpha=0.08, zorder=0)
    _ax_gd.axhline(0.0, color=DIM, lw=0.6, alpha=0.5)
    _ax_gd.axvline(0.0, color=DIM, lw=0.6, alpha=0.5)
    _ax_gd.plot([0.0, 1.5], [0.0, 0.0], color=CONE, lw=2.0, zorder=3)

    _gd_cmap = LinearSegmentedColormap.from_list("gd", ["#bfdbfe", A1])
    _pts_gd = _traj_gd[:, np.newaxis, :]
    _segs_gd = np.concatenate([_pts_gd[:-1], _pts_gd[1:]], axis=1)
    _lc_gd = LineCollection(_segs_gd, cmap=_gd_cmap, linewidth=2.0, zorder=6)
    _lc_gd.set_array(np.linspace(0, 1, len(_segs_gd)))
    _ax_gd.add_collection(_lc_gd)

    _ax_gd.scatter(0.0, 0.0, color=A1, s=220, zorder=7, marker="*", label=r"start $(0,\,0)$")
    _ax_gd.scatter(1.0, -1.0, color=DARK, s=200, zorder=7, marker="*", label=r"OLS $(1,\,-1)$")

    _ax_gd.grid(True, color=GRID, lw=0.8, zorder=0)
    _ax_gd.set_xlim(-0.5, 1.5)
    _ax_gd.set_ylim(-1.5, 1.0)
    _ax_gd.set_xlabel("$w_1$", fontsize=11)
    _ax_gd.set_ylabel("$w_2$", fontsize=11)
    _ax_gd.tick_params(labelsize=8, length=3, width=0.6)
    _ax_gd.set_title(rf"Gradient descent ($\eta = {pgd_eta}$)", fontsize=11, pad=10)
    _ax_gd.legend(fontsize=9)
    plt.tight_layout(pad=0.6)
    mo.mpl.interactive(_fig_gd)
    return


@app.cell
def pgd_plot(
    A1,
    B_COLOR,
    CONE,
    DIM,
    GRID,
    LineCollection,
    LinearSegmentedColormap,
    SPINE,
    loss_grid,
    mo,
    np,
    pgd_eta,
    pgd_n_iters,
    plt,
    projected_gradient_descent,
):
    _A = np.array([[3.0, 1.0], [2.0, 1.0]])
    _b = np.array([2.0, 1.0])

    _traj, _ = projected_gradient_descent(
        _A, _b, w0=np.array([0.0, 0.0]), eta=pgd_eta, n_iters=pgd_n_iters,
    )

    # Level set grid
    _w1g = np.linspace(-0.5, 1.5, 400)
    _w2g = np.linspace(-1.5, 1.0, 400)
    _W1, _W2 = np.meshgrid(_w1g, _w2g)
    _L = loss_grid(_A, _b, _W1, _W2)

    _fig, _ax = plt.subplots(figsize=(6.5, 6.5))
    for _sp in _ax.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)

    _ax.contour(_W1, _W2, _L, levels=[0.01, 0.04, 0.125, 0.4, 1.25, 4.0], colors=DIM, linewidths=0.7, alpha=0.35)
    _ax.contour(_W1, _W2, _L, levels=[1.0 / 26], colors=B_COLOR, linewidths=1.3, linestyles="--", alpha=0.7)
    _ax.fill_between([0.0, 1.5], 0.0, 1.0, color=CONE, alpha=0.08, zorder=0)
    _ax.axhline(0.0, color=DIM, lw=0.6, alpha=0.5)
    _ax.axvline(0.0, color=DIM, lw=0.6, alpha=0.5)
    _ax.plot([0.0, 1.5], [0.0, 0.0], color=CONE, lw=2.0, zorder=3)

    _pgd_cmap = LinearSegmentedColormap.from_list("pgd", ["#ddd6fe", B_COLOR])
    _pts = _traj[:, np.newaxis, :]
    _segs = np.concatenate([_pts[:-1], _pts[1:]], axis=1)
    _lc = LineCollection(_segs, cmap=_pgd_cmap, linewidth=2.0, zorder=6)
    _lc.set_array(np.linspace(0, 1, len(_segs)))
    _ax.add_collection(_lc)

    _ax.scatter(0.0, 0.0, color=A1, s=220, zorder=7, marker="*", label=r"start $(0,\,0)$")
    _ax.scatter(8.0 / 13, 0.0, color=B_COLOR, s=200, zorder=7, marker="*", label=rf"NNLS $({8 / 13:.2f},\;0)$")

    _ax.grid(True, color=GRID, lw=0.8, zorder=0)
    _ax.set_xlim(-0.5, 1.5)
    _ax.set_ylim(-1.5, 1.0)
    _ax.set_xlabel("$w_1$", fontsize=11)
    _ax.set_ylabel("$w_2$", fontsize=11)
    _ax.tick_params(labelsize=8, length=3, width=0.6)
    _ax.set_title(rf"Projected gradient descent ($\eta = {pgd_eta}$)", fontsize=11, pad=10)
    _ax.legend(fontsize=9)
    plt.tight_layout(pad=0.6)

    mo.mpl.interactive(_fig)
    return


@app.cell
def hl_intro(mo):
    mo.md(r"""
    ### Method 2: Lawson–Hanson Active-Set

    Instead of approaching the constraint boundary in small gradient steps, the
    **Lawson–Hanson** algorithm (1974) works with the active set directly. It
    partitions the weight indices at each iteration into:

    - **Passive set** $\mathcal{P}$: weights free to be positive
    - **Zero set** $\mathcal{Z}$: weights clamped to zero

    The KKT conditions for NNLS require that at the optimum every excluded weight
    satisfies $[A^\top(b - Aw)]_j \leq 0$ — moving that weight off zero would only
    increase the loss. The algorithm enforces this one variable at a time:

    1. **Outer step** — compute $z = A^\top(b - Aw)$. If $\max_{j \in \mathcal{Z}} z_j \leq 0$,
       stop. Otherwise add $t = \arg\max_{j \in \mathcal{Z}} z_j$ to $\mathcal{P}$.
    2. **Solve** unconstrained LS on the columns in $\mathcal{P}$.
    3. **Feasibility fix** — if any solution component is negative, interpolate back
       to the constraint boundary and shrink $\mathcal{P}$; repeat from step 2.

    Because each outer step solves an exact subproblem rather than a gradient
    approximation, the algorithm converges in at most $k$ outer steps — one per
    variable — independent of condition number.
    """)
    return


@app.cell
def hl_helpers(np):
    def hanson_lawson_trace(A, b):
        """NNLS via Lawson-Hanson, recording every inner and outer loop event."""
        n = A.shape[1]
        x = np.zeros(n)
        P = []

        def _z():
            return A.T @ (b - A @ x)

        def _loss():
            r = b - A @ x
            return 0.5 * float(r @ r)

        steps = [{"type": "init", "x": x.copy(), "P": list(P), "z": _z(), "loss": _loss()}]

        while True:
            z = _z()
            excluded = [j for j in range(n) if j not in P]
            if not excluded or np.max(z[excluded]) <= 1e-10:
                steps.append({"type": "done", "x": x.copy(), "P": list(P), "z": z, "loss": _loss()})
                break

            t = excluded[int(np.argmax(z[excluded]))]
            P = sorted(P + [t])
            steps.append({"type": "outer", "x": x.copy(), "P": list(P), "added": t, "z": z.copy(), "loss": _loss()})

            while True:
                s_P = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
                s = np.zeros(n)
                for _i, _j in enumerate(P):
                    s[_j] = s_P[_i]

                if np.all(s_P > 1e-10):
                    x = s.copy()
                    steps.append({
                        "type": "accept", "x": x.copy(), "P": list(P),
                        "s": s.copy(), "z": _z(), "loss": _loss(),
                    })
                    break

                infeasible_pidx = [_i for _i in range(len(P)) if s_P[_i] <= 1e-10]
                infeasible_cols = [P[_i] for _i in infeasible_pidx]
                alpha = min(x[P[_i]] / (x[P[_i]] - s_P[_i]) for _i in infeasible_pidx)
                for _i, _j in enumerate(P):
                    x[_j] = x[_j] + alpha * (s_P[_i] - x[_j])
                P = [_j for _j in P if x[_j] > 1e-10]
                steps.append({
                    "type": "interpolate",
                    "x": x.copy(),
                    "P": list(P),
                    "s": s.copy(),
                    "alpha": alpha,
                    "infeasible_cols": infeasible_cols,
                    "z": _z(),
                    "loss": _loss(),
                })

        return steps

    return (hanson_lawson_trace,)


@app.cell
def hl_trace(hanson_lawson_trace, np):
    # Cols 0 and 1 share the pattern (0.9, 0.1, 0, 0), making the 2×2 block
    # det(A^T A) ≈ 0.01. When col 1 enters the passive set, LS produces
    # w₁ = 5, w₀ = −3.5, triggering the inner-loop interpolation step.
    _A_hl = np.array([
        [1., 0.9, 0., 0.],
        [0., 0.1, 1., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ])
    _b_hl = np.array([1., 1., 0.5, 0.8])
    hl_steps = hanson_lawson_trace(_A_hl, _b_hl)
    hl_n_steps = len(hl_steps)
    return hl_n_steps, hl_steps


@app.cell
def hl_step_slider(hl_n_steps, mo):
    hl_slider = mo.ui.slider(0, hl_n_steps - 1, step=1, value=0, label="Step")
    return (hl_slider,)


@app.cell
def hl_plot(
    A1,
    A2,
    B_COLOR,
    CLAMP,
    DIM,
    GRID,
    SPINE,
    hl_n_steps,
    hl_slider,
    hl_steps,
    mo,
    np,
    plt,
):
    from matplotlib.patches import Patch as _Patch

    _step = hl_steps[hl_slider.value]
    _n = len(_step["x"])
    _labels = [f"$w_{{{j + 1}}}$" for j in range(_n)]
    _P = _step["P"]
    _stype = _step["type"]

    if _stype == "init":
        _desc = (
            r"Initial: $x = 0$, $\mathcal{P} = \emptyset$"
            r" — gradient $z$ reveals which weight benefits most from being freed"
        )
    elif _stype == "outer":
        _t = _step["added"]
        _p_str = r"\{" + r",\,".join(f"w_{{{j + 1}}}" for j in _P) + r"\}"
        _desc = (
            rf"Outer — $z[w_{{{_t + 1}}}] = {_step['z'][_t]:.3f}$ largest:"
            rf" add $w_{{{_t + 1}}}$ to $\mathcal{{P}} = {_p_str}$, solve LS on $\mathcal{{P}}$"
        )
    elif _stype == "accept":
        _p_str = r"\{" + r",\,".join(f"w_{{{j + 1}}}" for j in _P) + r"\}"
        _desc = (
            rf"Inner — unconstrained LS on $\mathcal{{P}} = {_p_str}$"
            r" gives all-positive weights: accepted, return to outer"
        )
    elif _stype == "interpolate":
        _ic = _step["infeasible_cols"]
        _ic_str = r"\{" + r",\,".join(f"w_{{{j + 1}}}" for j in _ic) + r"\}"
        _desc = (
            rf"Inner — LS proposal has $w \in {_ic_str} \leq 0$:"
            rf" interpolate by $\alpha = {_step['alpha']:.3f}$, shrink $\mathcal{{P}}$, re-solve"
        )
    else:
        _desc = r"Converged — all excluded gradients $z_j \leq 0$ (KKT satisfied)"

    _fig, (_ax_x, _ax_z) = plt.subplots(1, 2, figsize=(11, 5))

    # Fixed y-limits across all steps, so the axes don't rescale as the slider moves
    _all_x = np.concatenate([_s["x"] for _s in hl_steps] + [_s["s"] for _s in hl_steps if "s" in _s])
    _all_z = np.concatenate([_s["z"] for _s in hl_steps])
    _x_pad = 0.1 * (_all_x.max() - _all_x.min())
    _z_pad = 0.1 * (_all_z.max() - _all_z.min())
    _x_ylim = (_all_x.min() - _x_pad, _all_x.max() + _x_pad)
    _z_ylim = (_all_z.min() - _z_pad, _all_z.max() + _z_pad)

    # Left panel: current weights x
    _x_vals = _step["x"]
    _ax_x.bar(
        np.arange(_n),
        _x_vals,
        color=[B_COLOR if j in _P else DIM for j in range(_n)],
        alpha=0.75,
        zorder=3,
    )
    # Overlay negative LS proposal bars (only visible during interpolate steps)
    if "s" in _step:
        _s_vals = _step["s"]
        for _j in range(_n):
            if _s_vals[_j] < -1e-12:
                _ax_x.bar(_j, _s_vals[_j], color=CLAMP, alpha=0.5, zorder=4)
    _ax_x.axhline(0, color=DIM, lw=0.8, alpha=0.5)
    _ax_x.set_ylim(_x_ylim)
    _ax_x.set_xticks(np.arange(_n))
    _ax_x.set_xticklabels(_labels, fontsize=11)
    _ax_x.set_ylabel("weight", fontsize=10)
    _ax_x.set_title(r"Current weights $x$", fontsize=10)
    _ax_x.grid(True, color=GRID, lw=0.7, axis="y", zorder=0)
    for _sp in _ax_x.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)
    _ax_x.legend(
        handles=[
            _Patch(color=B_COLOR, alpha=0.75, label=r"passive $\mathcal{P}$ (free)"),
            _Patch(color=DIM, alpha=0.75, label=r"zero set $\mathcal{Z}$ (clamped)"),
            _Patch(color=CLAMP, alpha=0.5, label="LS proposal (infeasible)"),
        ],
        fontsize=8,
    )

    # Right panel: gradient z = A^T(b - Ax)
    _z_vals = _step["z"]
    _entering = _step.get("added") if _stype == "outer" else None
    _ax_z.bar(
        np.arange(_n),
        _z_vals,
        color=[A2 if j == _entering else (A1 if j in _P else DIM) for j in range(_n)],
        alpha=0.75,
        zorder=3,
    )
    _ax_z.axhline(0, color=CLAMP, lw=1.3, ls="--", alpha=0.7)
    _ax_z.set_ylim(_z_ylim)
    _ax_z.set_xticks(np.arange(_n))
    _ax_z.set_xticklabels(_labels, fontsize=11)
    _ax_z.set_ylabel(r"$[A^\top(b - Ax)]_j$", fontsize=10)
    _ax_z.set_title(r"Gradient $z = A^\top(b - Ax)$", fontsize=10)
    _ax_z.grid(True, color=GRID, lw=0.7, axis="y", zorder=0)
    for _sp in _ax_z.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)
    _ax_z.legend(
        handles=[
            _Patch(color=A2, alpha=0.75, label=r"entering $\mathcal{P}$ (max $z_j$)"),
            _Patch(color=A1, alpha=0.75, label=r"already in $\mathcal{P}$"),
            _Patch(color=DIM, alpha=0.75, label=r"zero set $\mathcal{Z}$ (candidates)"),
        ],
        fontsize=8,
    )

    plt.tight_layout(pad=0.8)
    _controls = mo.vstack([mo.md(f"**Step {hl_slider.value + 1} of {hl_n_steps}**: {hl_slider}")])
    mo.vstack([
        mo.md(_desc),
        mo.hstack([_controls, mo.mpl.interactive(_fig)], align="center", gap="0.5rem"),
    ])
    return


@app.cell
def hl_trace_3var(hanson_lawson_trace, mo, np):
    # Same near-collinear trick as hl_trace (cols 0 and 1 ≈ (1, 0.9) vs (0, 0.1)),
    # but with the fourth unit-vector column and its row dropped. Still forces
    # w1 back out of the passive set once w2 tries to enter, hitting the
    # interpolation branch with only 3 variables.
    _A_hl3 = np.array([
        [1., 0.9, 0.],
        [0., 0.1, 1.],
        [0., 0., 1.],
    ])
    _b_hl3 = np.array([1., 1., 0.5])
    hl3_steps = hanson_lawson_trace(_A_hl3, _b_hl3)
    hl3_n_steps = len(hl3_steps)
    assert any(_s["type"] == "interpolate" for _s in hl3_steps), (
        "3-variable example failed to trigger the interpolation step"
    )

    _lines = []
    for _i, _s in enumerate(hl3_steps):
        _x_str = ", ".join(f"{v:.3f}" for v in _s["x"])
        if _s["type"] == "outer":
            _lines.append(f"{_i}. outer — add w{_s['added'] + 1}: x = ({_x_str})")
        elif _s["type"] == "interpolate":
            _cols = ", ".join(f"w{j + 1}" for j in _s["infeasible_cols"])
            _lines.append(f"{_i}. interpolate — drop {_cols} (α = {_s['alpha']:.3f}): x = ({_x_str})")
        else:
            _lines.append(f"{_i}. {_s['type']}: x = ({_x_str})")

    mo.md(
        "**3-variable Hanson-Lawson trace** (same collinearity trick as the "
        "4-variable example, one fewer column):\n\n" + "\n".join(f"- {_l}" for _l in _lines)
    )
    return hl3_n_steps, hl3_steps


@app.cell
def hl_step_slider_3var(hl3_n_steps, mo):
    hl3_slider = mo.ui.slider(0, hl3_n_steps - 1, step=1, value=0, label="Step")
    return (hl3_slider,)


@app.cell
def hl_plot_3var(
    A1,
    A2,
    B_COLOR,
    CLAMP,
    DIM,
    GRID,
    SPINE,
    hl3_n_steps,
    hl3_slider,
    hl3_steps,
    mo,
    np,
    plt,
):
    from matplotlib.patches import Patch as _Patch

    _step = hl3_steps[hl3_slider.value]
    _n = len(_step["x"])
    _labels = [f"$w_{{{j + 1}}}$" for j in range(_n)]
    _P = _step["P"]
    _stype = _step["type"]

    if _stype == "init":
        _desc = (
            r"Initial: $x = 0$, $\mathcal{P} = \emptyset$"
            r" — gradient $z$ reveals which weight benefits most from being freed"
        )
    elif _stype == "outer":
        _t = _step["added"]
        _p_str = r"\{" + r",\,".join(f"w_{{{j + 1}}}" for j in _P) + r"\}"
        _desc = (
            rf"Outer — $z[w_{{{_t + 1}}}] = {_step['z'][_t]:.3f}$ largest:"
            rf" add $w_{{{_t + 1}}}$ to $\mathcal{{P}} = {_p_str}$, solve LS on $\mathcal{{P}}$"
        )
    elif _stype == "accept":
        _p_str = r"\{" + r",\,".join(f"w_{{{j + 1}}}" for j in _P) + r"\}"
        _desc = (
            rf"Inner — unconstrained LS on $\mathcal{{P}} = {_p_str}$"
            r" gives all-positive weights: accepted, return to outer"
        )
    elif _stype == "interpolate":
        _ic = _step["infeasible_cols"]
        _ic_str = r"\{" + r",\,".join(f"w_{{{j + 1}}}" for j in _ic) + r"\}"
        _desc = (
            rf"Inner — LS proposal has $w \in {_ic_str} \leq 0$:"
            rf" interpolate by $\alpha = {_step['alpha']:.3f}$, shrink $\mathcal{{P}}$, re-solve"
        )
    else:
        _desc = r"Converged — all excluded gradients $z_j \leq 0$ (KKT satisfied)"

    _fig, (_ax_x, _ax_z) = plt.subplots(1, 2, figsize=(11, 5))

    # Fixed y-limits across all steps, so the axes don't rescale as the slider moves
    _all_x = np.concatenate([_s["x"] for _s in hl3_steps] + [_s["s"] for _s in hl3_steps if "s" in _s])
    _all_z = np.concatenate([_s["z"] for _s in hl3_steps])
    _x_pad = 0.1 * (_all_x.max() - _all_x.min())
    _z_pad = 0.1 * (_all_z.max() - _all_z.min())
    _x_ylim = (_all_x.min() - _x_pad, _all_x.max() + _x_pad)
    _z_ylim = (_all_z.min() - _z_pad, _all_z.max() + _z_pad)

    # Left panel: current weights x
    _x_vals = _step["x"]
    _ax_x.bar(
        np.arange(_n),
        _x_vals,
        color=[B_COLOR if j in _P else DIM for j in range(_n)],
        alpha=0.75,
        zorder=3,
    )
    # Overlay negative LS proposal bars (only visible during interpolate steps)
    if "s" in _step:
        _s_vals = _step["s"]
        for _j in range(_n):
            if _s_vals[_j] < -1e-12:
                _ax_x.bar(_j, _s_vals[_j], color=CLAMP, alpha=0.5, zorder=4)
    _ax_x.axhline(0, color=DIM, lw=0.8, alpha=0.5)
    _ax_x.set_ylim(_x_ylim)
    _ax_x.set_xticks(np.arange(_n))
    _ax_x.set_xticklabels(_labels, fontsize=11)
    _ax_x.set_ylabel("weight", fontsize=10)
    _ax_x.set_title(r"Current weights $x$", fontsize=10)
    _ax_x.grid(True, color=GRID, lw=0.7, axis="y", zorder=0)
    for _sp in _ax_x.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)
    _ax_x.legend(
        handles=[
            _Patch(color=B_COLOR, alpha=0.75, label=r"passive $\mathcal{P}$ (free)"),
            _Patch(color=DIM, alpha=0.75, label=r"zero set $\mathcal{Z}$ (clamped)"),
            _Patch(color=CLAMP, alpha=0.5, label="LS proposal (infeasible)"),
        ],
        fontsize=8,
    )

    # Right panel: gradient z = A^T(b - Ax)
    _z_vals = _step["z"]
    _entering = _step.get("added") if _stype == "outer" else None
    _ax_z.bar(
        np.arange(_n),
        _z_vals,
        color=[A2 if j == _entering else (A1 if j in _P else DIM) for j in range(_n)],
        alpha=0.75,
        zorder=3,
    )
    _ax_z.axhline(0, color=CLAMP, lw=1.3, ls="--", alpha=0.7)
    _ax_z.set_ylim(_z_ylim)
    _ax_z.set_xticks(np.arange(_n))
    _ax_z.set_xticklabels(_labels, fontsize=11)
    _ax_z.set_ylabel(r"$[A^\top(b - Ax)]_j$", fontsize=10)
    _ax_z.set_title(r"Gradient $z = A^\top(b - Ax)$", fontsize=10)
    _ax_z.grid(True, color=GRID, lw=0.7, axis="y", zorder=0)
    for _sp in _ax_z.spines.values():
        _sp.set_edgecolor(SPINE)
        _sp.set_linewidth(0.8)
    _ax_z.legend(
        handles=[
            _Patch(color=A2, alpha=0.75, label=r"entering $\mathcal{P}$ (max $z_j$)"),
            _Patch(color=A1, alpha=0.75, label=r"already in $\mathcal{P}$"),
            _Patch(color=DIM, alpha=0.75, label=r"zero set $\mathcal{Z}$ (candidates)"),
        ],
        fontsize=8,
    )

    plt.tight_layout(pad=0.8)
    _controls = mo.vstack([mo.md(f"**Step {hl3_slider.value + 1} of {hl3_n_steps}**: {hl3_slider}")])
    mo.vstack([
        mo.md(_desc),
        mo.hstack([_controls, mo.mpl.interactive(_fig)], align="center", gap="0.5rem"),
    ])
    return


@app.cell
def cone_sliders(mo, np):
    theta1_slider = mo.ui.slider(0, 2 * np.pi, step=0.05, value=np.pi / 6, label="θ₁")
    theta2_slider = mo.ui.slider(0, 2 * np.pi, step=0.05, value=2 * np.pi / 3, label="θ₂")
    return theta1_slider, theta2_slider


@app.cell
def cone_helpers(np):
    def cone_fill_polygon(theta1, theta2, R, n=300):
        """Fill polygon for the convex cone between two rays, sweeping the shorter arc."""
        t1 = theta1 % (2 * np.pi)
        t2 = theta2 % (2 * np.pi)
        delta_ccw = (t2 - t1) % (2 * np.pi)
        if delta_ccw > np.pi:
            t1, t2 = t2, t1
            delta_ccw = 2 * np.pi - delta_ccw
        arc = np.linspace(t1, t1 + delta_ccw, n)
        poly_x = np.concatenate([[0], R * np.cos(arc), [0]])
        poly_y = np.concatenate([[0], R * np.sin(arc), [0]])
        return poly_x, poly_y, t1, t1 + delta_ccw


    def draw_vector(ax, v, label, color, label_offset=0.22):
        """Draw a 2D vector from the origin with a clean arrow and label."""
        angle = np.arctan2(v[1], v[0])
        ax.annotate(
            "",
            xy=(v[0], v[1]),
            xytext=(0, 0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 2.0,
                "mutation_scale": 16,
            },
            zorder=4,
        )
        ax.text(
            v[0] + label_offset * np.cos(angle),
            v[1] + label_offset * np.sin(angle),
            label,
            fontsize=14,
            color=color,
            ha="center",
            va="center",
            fontweight="bold",
        )

    return cone_fill_polygon, draw_vector


@app.cell
def cone_style():
    lim = 3.5
    R = 1000.0  # large enough that the arc lies far outside the plot limits
    return R, lim


@app.cell
def cone_plot(
    A1,
    A2,
    CONE,
    DIM,
    GRID,
    R,
    SPINE,
    cone_fill_polygon,
    draw_vector,
    lim,
    mo,
    np,
    plt,
    theta1_slider,
    theta2_slider,
):
    theta1 = theta1_slider.value
    theta2 = theta2_slider.value

    a1 = np.array([np.cos(theta1), np.sin(theta1)])
    a2 = np.array([np.cos(theta2), np.sin(theta2)])

    poly_x, poly_y, ray1, ray2 = cone_fill_polygon(theta1, theta2, R)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
        spine.set_linewidth(0.8)

    ax.fill(poly_x, poly_y, color=CONE, alpha=0.10, zorder=1)

    ray_kw = {"color": CONE, "lw": 0.9, "ls": "--", "alpha": 0.5, "zorder": 2}
    ax.plot([0, R * np.cos(ray1)], [0, R * np.sin(ray1)], **ray_kw)
    ax.plot([0, R * np.cos(ray2)], [0, R * np.sin(ray2)], **ray_kw)

    angle_arc = np.linspace(ray1, ray2, 80)
    ax.plot(0.45 * np.cos(angle_arc), 0.45 * np.sin(angle_arc), color=DIM, lw=0.9, alpha=0.7, zorder=3)

    draw_vector(ax, a1, r"$\mathbf{a}_1$", A1)
    draw_vector(ax, a2, r"$\mathbf{a}_2$", A2)

    ax.scatter([0], [0], color=DIM, s=16, zorder=6, linewidths=0)
    ax.axhline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    ax.axvline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8, length=3, width=0.6)

    ax.set_title(
        r"$\mathcal{C} = \{\, \lambda_1\,\mathbf{a}_1 + \lambda_2\,\mathbf{a}_2"
        r"\;\mid\; \lambda_1,\, \lambda_2 \geq 0 \,\}$",
        fontsize=11,
        pad=12,
    )
    plt.tight_layout(pad=0.6)

    controls = mo.vstack(
        [
            mo.md(f"**a₁** angle θ₁: {theta1_slider}"),
            mo.md(f"**a₂** angle θ₂: {theta2_slider}"),
        ]
    )
    mo.hstack([controls, mo.mpl.interactive(fig)], align="center")
    return


@app.cell
def lambda_sliders(mo):
    lambda1_slider = mo.ui.slider(0, 3, step=0.05, value=1.5, label="λ₁")
    lambda2_slider = mo.ui.slider(0, 3, step=0.05, value=1.0, label="λ₂")
    return lambda1_slider, lambda2_slider


@app.cell
def nnls_decomp_plot(
    A1,
    A2,
    B_COLOR,
    CONE,
    DIM,
    GRID,
    R,
    SPINE,
    cone_fill_polygon,
    draw_vector,
    lambda1_slider,
    lambda2_slider,
    lim,
    mo,
    np,
    plt,
):
    _ALPHA_COMP = 0.45

    _theta1 = np.pi / 6
    _theta2 = 2 * np.pi / 3
    _lambda1 = lambda1_slider.value
    _lambda2 = lambda2_slider.value

    _a1 = np.array([np.cos(_theta1), np.sin(_theta1)])
    _a2 = np.array([np.cos(_theta2), np.sin(_theta2)])
    _mid = _lambda1 * _a1
    _b = _mid + _lambda2 * _a2

    _poly_x, _poly_y, _ray1, _ray2 = cone_fill_polygon(_theta1, _theta2, R)

    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))

    for _spine in _ax.spines.values():
        _spine.set_edgecolor(SPINE)
        _spine.set_linewidth(0.8)

    _ax.fill(_poly_x, _poly_y, color=CONE, alpha=0.10, zorder=1)
    _ray_kw = {"color": CONE, "lw": 0.9, "ls": "--", "alpha": 0.5, "zorder": 2}
    _ax.plot([0, R * np.cos(_ray1)], [0, R * np.sin(_ray1)], **_ray_kw)
    _ax.plot([0, R * np.cos(_ray2)], [0, R * np.sin(_ray2)], **_ray_kw)

    draw_vector(_ax, _a1, r"$\mathbf{a}_1$", A1)
    draw_vector(_ax, _a2, r"$\mathbf{a}_2$", A2)

    # Tip-to-tail: λ₁a₁ from origin, then λ₂a₂ from its tip
    _comp_kw = {"mutation_scale": 14, "lw": 1.8, "alpha": _ALPHA_COMP}
    _ax.annotate("", xy=_mid, xytext=(0.0, 0.0), arrowprops={"arrowstyle": "-|>", "color": A1, **_comp_kw}, zorder=4)
    _ax.annotate("", xy=_b, xytext=_mid, arrowprops={"arrowstyle": "-|>", "color": A2, **_comp_kw}, zorder=4)

    _ax.scatter([_mid[0]], [_mid[1]], color=DIM, s=18, zorder=6, linewidths=0)

    # Result vector b
    _ax.annotate(
        "",
        xy=_b,
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": B_COLOR, "lw": 2.0, "mutation_scale": 16},
        zorder=5,
    )
    _b_angle = np.arctan2(_b[1], _b[0])
    _ax.text(
        _b[0] + 0.22 * np.cos(_b_angle),
        _b[1] + 0.22 * np.sin(_b_angle),
        r"$\mathbf{b}$",
        fontsize=14,
        color=B_COLOR,
        ha="center",
        va="center",
        fontweight="bold",
    )

    _ax.scatter([0], [0], color=DIM, s=16, zorder=7, linewidths=0)
    _ax.axhline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    _ax.axvline(0, color=DIM, lw=0.6, alpha=0.5, zorder=1)
    _ax.grid(True, color=GRID, lw=0.8, zorder=0)

    _ax.set_xlim(-lim, lim)
    _ax.set_ylim(-lim, lim)
    _ax.set_aspect("equal")
    _ax.tick_params(labelsize=8, length=3, width=0.6)
    _ax.set_title(
        r"$\mathbf{b} = \lambda_1\,\mathbf{a}_1 + \lambda_2\,\mathbf{a}_2$",
        fontsize=11,
        pad=12,
    )
    plt.tight_layout(pad=0.6)

    _controls = mo.vstack(
        [
            mo.md(f"λ₁: {lambda1_slider}"),
            mo.md(f"λ₂: {lambda2_slider}"),
        ]
    )
    mo.hstack([_controls, mo.mpl.interactive(_fig)], align="center", gap="0.5rem")
    return


if __name__ == "__main__":
    app.run()
