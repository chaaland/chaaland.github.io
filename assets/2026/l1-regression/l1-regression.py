import marimo

__generated_with = "0.21.1"
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
    return IMAGE_DIR, np, plt


@app.cell
def _():
    def make_cartesian_plane(ax):
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")


    def remove_spines(ax):
        ax.spines[["right", "top"]].set_visible(False)

    return make_cartesian_plane, remove_spines


@app.cell
def _(IMAGE_DIR, make_cartesian_plane, np, plt):
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
        if _i == n_points - 1:
            plt.savefig(IMAGE_DIR / f"1d-abs-deviation-{_i:02}.png")

    plt.show()
    return


@app.function
def golden_section_minimize(obj_fn, a, b, n_iters: int = 5):
    assert b > a
    assert n_iters >= 0

    L = b - a
    alpha = (-1 + 5**0.5) / 2
    x1 = b - alpha * L
    x2 = a + alpha * L

    assert a < x1 < x2 < b

    f_x1 = obj_fn(beta=x1)
    f_x2 = obj_fn(beta=x2)

    for i in range(n_iters):
        if f_x1 < f_x2:
            b = x2
            L = b - a

            x2, f_x2 = x1, f_x1

            x1 = b - alpha * L
            f_x1 = obj_fn(beta=x1)
        else:
            a = x1
            L = b - a

            x1, f_x1 = x2, f_x2
            x2 = a + alpha * L

            f_x2 = obj_fn(beta=x2)

        assert a < x1 < x2 < b, f"{a}, {x1}, {x2}, {b}, {i}"

    x_opt = (a + b) / 2
    return x_opt, obj_fn(beta=x_opt)


@app.cell
def _(IMAGE_DIR, make_cartesian_plane, np, plt):
    # Toy residuals: fixed slope, vary only the bias b
    # Objective: f(b) = mean(|b - r_i|), minimized at the median of r
    np.random.seed(7)
    _r = np.array([-3.0, -1.0, 0.5, 2.0, 4.5, 11.0])
    _median = np.median(_r)
    _half = np.abs(_r - _median).max() + 2
    _b_grid = np.linspace(_median - _half, _median + _half, 500)
    _f = np.mean(np.abs(_b_grid[:, np.newaxis] - _r), axis=1)

    _, ax_bias = plt.subplots(figsize=(7, 4))

    ax_bias.plot(_b_grid, _f, color="k", linewidth=2)

    # Mark the data points as ticks on the x-axis
    ax_bias.scatter(_r, np.zeros_like(_r), color="k", zorder=5, s=40, clip_on=False)

    # Mark the median
    _f_at_median = np.mean(np.abs(_median - _r))
    ax_bias.scatter([_median], [_f_at_median], color="tab:red", zorder=5, s=60)
    ax_bias.vlines(_median, 0, _f_at_median, colors="tab:red", linestyles="--", linewidth=1)
    ax_bias.text(_median, -0.4, "median", ha="center", va="top", color="tab:red", fontsize=11)

    make_cartesian_plane(ax_bias)
    ax_bias.set_title(r"$\frac{1}{N}\sum_{i=1}^N |\beta_0 - r_i|$")
    ax_bias.set_xlabel(r"$\beta_0$")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "bias-objective.png")
    plt.show()
    return


@app.function
def knot_scan_1d(x_col, residuals):
    """Exact 1D LAD minimizer by evaluating the objective at every knot.

    Knots z_i = residuals[i] / x_col[i] are the points where the i-th
    absolute-value term changes slope.  The minimum of a sum of piecewise-
    linear functions must occur at one of these N knots, so scanning all of
    them is correct — but costs O(N) knots × O(N) per evaluation = O(N²).
    """
    import numpy as np

    nz = x_col != 0
    knots = residuals[nz] / x_col[nz]
    best_val = np.inf
    best_knot = knots[0]
    for z in knots:
        val = np.mean(np.abs(x_col * z - residuals))
        if val < best_val:
            best_val = val
            best_knot = z
    return best_knot


@app.function
def coordinate_descent_iter(X, y, beta, b, solver=None):
    """One full pass over all weight coordinates, then update the bias.

    Args:
        solver: callable (x_col, residuals) -> float for the 1D sub-problem.
                Defaults to weighted_median_1d.
    """
    import numpy as np

    if solver is None:
        solver = weighted_median_1d

    _, d = X.shape
    history = []

    for k in range(d):
        # Partial residual: remove coordinate k's contribution from prediction
        r = y - b - X @ beta + X[:, k] * beta[k]

        beta[k] = solver(X[:, k], r)
        history.append((beta.copy(), b))

    # Bias closed-form: minimizer of sum_i |b - r_i| is the median
    b = np.median(y - X @ beta)
    history.append((beta.copy(), b))

    return beta, b, history


@app.function
def irls_lad(X, y, n_iters: int = 50, eps: float = 1e-6, beta0=None, b0=None):
    """Iteratively Reweighted Least Squares for L1 regression (LAD).

    At each step, solves a weighted least squares problem where the weights
    are the inverse of the absolute residuals: w_i = 1 / max(|r_i|, eps).

    This works because the L1 loss can be written as a weighted L2 loss:
        sum_i |r_i| = sum_i w_i * r_i^2   where w_i = 1/|r_i|

    Args:
        X: (n, d) design matrix (without bias column)
        y: (n,) target vector
        n_iters: number of IRLS iterations
        eps: small constant to avoid division by zero near zero residuals

    Returns:
        beta: (d,) slope coefficients
        b: scalar bias term
        history: list of (beta, b) snapshots after each iteration
    """
    import numpy as np

    n, d = X.shape
    # Augment X with a bias column for a single unified WLS solve
    X_aug = np.column_stack([X, np.ones(n)])

    theta = np.zeros(d + 1)
    theta[:d] = np.zeros(d) if beta0 is None else np.asarray(beta0, dtype=float)
    theta[d] = np.median(y) if b0 is None else float(b0)
    history = [(theta[:d].copy(), theta[d])]

    for _ in range(n_iters):
        residuals = y - X_aug @ theta
        weights = 1.0 / np.maximum(np.abs(residuals), eps)

        # Weighted least squares: theta = (X^T W X)^{-1} X^T W y
        # Scale rows by weights directly to avoid materializing the NxN diagonal matrix
        Xw = X_aug * weights[:, np.newaxis]
        theta = np.linalg.solve(Xw.T @ X_aug, Xw.T @ y)
        history.append((theta[:d].copy(), theta[d]))

    return theta[:d], theta[d], history


@app.function
def weighted_median_1d(x_col, residuals):
    """Exact 1D LAD minimizer via weighted median (O(N log N)).

    The objective (1/n) Σ |x_col[i]| · |β − residuals[i]/x_col[i]| is
    minimized at the weighted median of knots z_i = r_i/x_i with weights
    w_i = |x_col[i]|.  One sort + a cumulative-weight scan suffices.
    """
    import numpy as np

    nz = x_col != 0
    knots = residuals[nz] / x_col[nz]
    weights = np.abs(x_col[nz])

    order = np.argsort(knots)
    sorted_knots = knots[order]
    sorted_weights = weights[order]

    cumulative = np.cumsum(sorted_weights)
    half = sorted_weights.sum() / 2.0
    idx = np.searchsorted(cumulative, half)
    return sorted_knots[min(idx, len(sorted_knots) - 1)]


@app.function
def coordinate_descent_lad(X, y, n_iters: int = 20, solver=None, beta0=None, b0=None):
    import numpy as np

    if solver is None:
        solver = weighted_median_1d

    _, d = X.shape
    beta = np.zeros(d) if beta0 is None else np.asarray(beta0, dtype=float).copy()
    b = np.median(y) if b0 is None else float(b0)

    history = [(beta.copy(), b)]

    for _ in range(n_iters):
        beta, b, iter_history = coordinate_descent_iter(X, y, beta, b, solver)
        history.extend(iter_history)

    return beta, b, history


@app.function
def lad_minimize(X, y, method="weighted_median", n_iters=20, beta0=None, b0=None, **kwargs):
    """Minimize the LAD (L1) objective: (1/n) Σ |X @ beta + b − y|.

    Args:
        X: (n, d) design matrix (no bias column)
        y: (n,) targets
        method: '``golden_section``'  – coordinate descent, approximate 1D solver (O(N·iters) per coord)
                '``knot_scan``'       – coordinate descent, exact 1D solver (O(N²) per coord)
                '``weighted_median``' – coordinate descent, exact 1D solver (O(N log N) per coord)
                '``irls``'            – iteratively reweighted least squares
        n_iters: number of outer iterations
        beta0: (d,) initial coefficients, or None to use zeros
        b0: initial bias, or None to use median(y)
        **kwargs: forwarded to the backend (e.g. eps= for irls)

    Returns:
        beta: (d,) coefficients
        b: scalar bias
        history: list of (beta, b) snapshots
    """
    if method == "irls":
        return irls_lad(X, y, n_iters=n_iters, beta0=beta0, b0=b0, **kwargs)

    def _golden_solver(x_col, res):
        import numpy as np

        nz = x_col != 0
        knots = res[nz] / x_col[nz]
        a_bound, b_bound = knots.min() - 1e-6, knots.max() + 1e-6

        def obj_fn(beta, x=x_col, r=res):
            return np.mean(np.abs(beta * x - r))

        return golden_section_minimize(obj_fn, a=a_bound, b=b_bound, n_iters=50)[0]

    solver_map = {
        "golden_section": _golden_solver,
        "knot_scan": knot_scan_1d,
        "weighted_median": weighted_median_1d,
    }
    if method not in solver_map:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(solver_map)}")

    return coordinate_descent_lad(X, y, n_iters=n_iters, solver=solver_map[method], beta0=beta0, b0=b0)


@app.cell
def _(np):
    np.random.seed(42)
    x2d = np.linspace(0, 10, 7)
    true_slope, true_intercept = 1.5, 2.0
    y2d = true_slope * x2d + true_intercept + np.random.normal(0, 1.5, 7)
    y2d[3] += 8.0
    return x2d, y2d


@app.cell
def _(np, plt, remove_spines, x2d, y2d):
    _X_aug = np.column_stack([x2d, np.ones_like(x2d)])
    (_ols_slope, _ols_intercept), _, _, _ = np.linalg.lstsq(_X_aug, y2d, rcond=None)
    _lad_slope, _lad_intercept, _ = lad_minimize(x2d.reshape(-1, 1), y2d)
    _lad_slope = _lad_slope[0]
    _x_line = np.array([x2d.min(), x2d.max()])

    _, ax_scatter = plt.subplots(figsize=(6, 4))
    ax_scatter.scatter(x2d, y2d, color="k", zorder=3, s=30)
    ax_scatter.plot(_x_line, _ols_slope * _x_line + _ols_intercept, color="tab:blue", linewidth=2, label="OLS")
    ax_scatter.plot(_x_line, _lad_slope * _x_line + _lad_intercept, color="tab:orange", linewidth=2, label="LAD")
    ax_scatter.set_xlabel("x")
    ax_scatter.set_ylabel("y")
    ax_scatter.legend(frameon=False)
    remove_spines(ax_scatter)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(IMAGE_DIR, np, plt, remove_spines, x2d, y2d):
    # Contour grid
    _slopes = np.linspace(-0.5, 2, 200)
    _intercepts = np.linspace(-4, 12, 200)
    _B, _S = np.meshgrid(_intercepts, _slopes)
    _Z = np.mean(np.abs(_S[:, :, np.newaxis] * x2d + _B[:, :, np.newaxis] - y2d), axis=2)
    _levels = np.linspace(_Z.min(), _Z.min() + 16, 60)

    _, (ax_scatter2, ax_contour) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_scatter2.scatter(x2d, y2d, color="k", zorder=3, s=40)
    ax_scatter2.set_xlabel("x")
    ax_scatter2.set_ylabel("y")
    remove_spines(ax_scatter2)

    ax_contour.contourf(_B, _S, _Z, levels=_levels, cmap="viridis")
    ax_contour.contour(_B, _S, _Z, levels=_levels, colors="white", linewidths=0.4, alpha=0.4)
    ax_contour.set_xlabel(r"$\beta_0$")
    ax_contour.set_ylabel(r"$\beta_1$")
    ax_contour.set_title("L1 loss landscape")

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "loss-landscape.png")
    plt.show()
    return


@app.cell
def _(IMAGE_DIR, np, plt, remove_spines, x2d, y2d):
    _X2d = x2d.reshape(-1, 1)
    _colors = ["tab:cyan", "tab:orange", "tab:pink"]
    _histories = [
        lad_minimize(_X2d, y2d, method="weighted_median", n_iters=10, beta0=[0.0], b0=11.0)[2],
        lad_minimize(_X2d, y2d, method="weighted_median", n_iters=10, beta0=[-0.25], b0=6.0)[2],
        lad_minimize(_X2d, y2d, method="weighted_median", n_iters=10, beta0=[1.0], b0=0.0)[2],
    ]


    def _slope_hist(history):
        return np.array([h[0][0] for h in history])


    def _bias_hist(history):
        return np.array([h[1] for h in history])


    def _loss_hist(s_hist, b_hist):
        return np.array([np.mean(np.abs(s_hist[i] * x2d + b_hist[i] - y2d)) for i in range(len(s_hist))])


    _all_s = np.concatenate([_slope_hist(h) for h in _histories])
    _all_b = np.concatenate([_bias_hist(h) for h in _histories])

    _pad = 8.0
    _slopes = np.linspace(_all_s.min() - _pad, _all_s.max() + _pad, 500)
    _intercepts = np.linspace(_all_b.min() - _pad, _all_b.max() + _pad, 500)
    _B, _S = np.meshgrid(_intercepts, _slopes)
    _Z = np.mean(np.abs(_S[:, :, np.newaxis] * x2d + _B[:, :, np.newaxis] - y2d), axis=2)
    _levels = np.linspace(_Z.min(), _Z.min() + 16, 60)

    _d = _X2d.shape[1]

    _, (ax_loss, ax_traj) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_traj.contourf(_B, _S, _Z, levels=_levels, cmap="viridis")
    ax_traj.contour(_B, _S, _Z, levels=_levels, colors="white", linewidths=0.4, alpha=0.4)

    for _color, _history in zip(_colors, _histories):
        _s = _slope_hist(_history)
        _b = _bias_hist(_history)
        _loss = _loss_hist(_s, _b)
        _iters = np.arange(len(_loss)) / (_d + 1)

        ax_loss.plot(_iters, _loss, color=_color, linewidth=2)
        ax_loss.scatter(_iters, _loss, color=_color, s=20, zorder=5)

        ax_traj.plot(_b, _s, color=_color, linewidth=1.5)
        ax_traj.scatter(_b[1:-1], _s[1:-1], color=_color, s=25, zorder=5)
        ax_traj.scatter(_b[0], _s[0], color=_color, s=80, zorder=6, marker="^")
        ax_traj.scatter(_b[-1], _s[-1], color=_color, s=80, zorder=6, marker="*")


    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("mean absolute deviation")
    ax_loss.set_ylim([0, 9])

    ax_traj.set_xlim([-4, 12])
    ax_traj.set_ylim([-0.5, 2])
    remove_spines(ax_loss)

    ax_traj.set_xlabel(r"$\beta_0$")
    ax_traj.set_ylabel(r"$\beta_1$")
    ax_traj.set_title("Coordinate descent trajectories")

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "cd-trajectory.png")
    plt.show()
    return


@app.cell
def _(IMAGE_DIR, np, plt, remove_spines, x2d, y2d):
    _beta_irls, _b_irls, _irls_history = lad_minimize(x2d.reshape(-1, 1), y2d, method="irls", n_iters=20)

    _s_hist = np.array([h[0][0] for h in _irls_history])
    _b_hist = np.array([h[1] for h in _irls_history])
    _loss_hist = np.array([np.mean(np.abs(_s_hist[i] * x2d + _b_hist[i] - y2d)) for i in range(len(_irls_history))])

    _pad = 2.0
    _slopes = np.linspace(_s_hist.min() - _pad, _s_hist.max() + _pad, 200)
    _intercepts = np.linspace(_b_hist.min() - _pad, _b_hist.max() + _pad, 200)
    _B, _S = np.meshgrid(_intercepts, _slopes)
    _Z = np.mean(np.abs(_S[:, :, np.newaxis] * x2d + _B[:, :, np.newaxis] - y2d), axis=2)
    _levels = np.linspace(_Z.min(), _Z.min() + 10, 50)

    _, (ax_loss_irls, ax_traj_irls) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_loss_irls.plot(_loss_hist, color="k", linewidth=2)
    ax_loss_irls.scatter(range(len(_loss_hist)), _loss_hist, color="k", s=20, zorder=5)
    ax_loss_irls.set_xlabel("iteration")
    ax_loss_irls.set_ylabel("mean absolute deviation")
    ax_loss_irls.set_title("IRLS convergence")
    ax_loss_irls.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
    remove_spines(ax_loss_irls)

    ax_traj_irls.contourf(_B, _S, _Z, levels=_levels, cmap="viridis")
    ax_traj_irls.contour(_B, _S, _Z, levels=_levels, colors="white", linewidths=0.4, alpha=0.4)
    ax_traj_irls.plot(_b_hist, _s_hist, color="white", linewidth=1.5)
    ax_traj_irls.scatter(_b_hist[1:-1], _s_hist[1:-1], color="white", s=25, zorder=5)
    ax_traj_irls.scatter(_b_hist[0], _s_hist[0], color="tab:green", s=70, zorder=6, label="start")
    ax_traj_irls.scatter(_b_hist[-1], _s_hist[-1], color="tab:red", s=70, zorder=6, label="end")
    ax_traj_irls.set_xlabel(r"$\beta_0$")
    ax_traj_irls.set_ylabel(r"$\beta_1$")
    ax_traj_irls.set_title("IRLS trajectory")
    ax_traj_irls.set_xlim(2, 12)
    ax_traj_irls.set_ylim(-0.5, 2)
    ax_traj_irls.legend()

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "irls-trajectory.png")
    plt.show()
    return


@app.cell
def _(IMAGE_DIR, np, plt, remove_spines, x2d, y2d):
    _, _, _cd_history = lad_minimize(x2d.reshape(-1, 1), y2d, method="weighted_median", n_iters=20)
    _, _, _irls_history = lad_minimize(x2d.reshape(-1, 1), y2d, method="irls", n_iters=20)


    def _losses(history):
        s = np.array([h[0][0] for h in history])
        b = np.array([h[1] for h in history])
        return np.array([np.mean(np.abs(s[i] * x2d + b[i] - y2d)) for i in range(len(history))])


    _cd_loss = _losses(_cd_history)
    _irls_loss = _losses(_irls_history)

    # d+1 sub-steps per CD cycle (d feature coords + 1 bias); map to fractional iterations
    _d = len(_cd_history[0][0])
    _cd_iters = np.arange(len(_cd_loss)) / (_d + 1)
    _irls_iters = np.arange(len(_irls_loss), dtype=float)

    _, ax_cmp = plt.subplots(figsize=(7, 4))
    ax_cmp.plot(_cd_iters, _cd_loss, color="tab:blue", linewidth=2, label="coordinate descent (weighted median)")
    ax_cmp.scatter(_cd_iters, _cd_loss, color="tab:blue", s=30, zorder=5)
    ax_cmp.plot(_irls_iters, _irls_loss, color="tab:orange", linewidth=2, label="IRLS")
    ax_cmp.scatter(_irls_iters, _irls_loss, color="tab:orange", s=30, zorder=5)
    ax_cmp.set_xlabel("iteration")
    ax_cmp.set_ylabel("mean absolute deviation")
    ax_cmp.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
    ax_cmp.legend(frameon=False)
    remove_spines(ax_cmp)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "cd-vs-irls.png")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
