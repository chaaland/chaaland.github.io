import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy as scp
    import functools

    from pathlib import Path
    import re
    from collections import Counter

    # IMAGE_DIR = Path("_gradient_flow/images")
    # IMAGE_DIR.mkdir(exist_ok=True)
    return Counter, Path, functools, mo, np, plt, re, scp


@app.cell
def _():
    def make_cartesian_plane(ax):
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")


    def remove_spines(ax):
        ax.spines[["right", "top"]].set_visible(False)

    return (remove_spines,)


@app.cell
def _(np):
    def plot_fading_trajectory(ax, points, alpha0=1.0, decay=0.9, color="tab:red"):
        """Plot a 2D trajectory with fading opacity for each step."""

        points = np.asarray(points)
        N = len(points)

        x_points = points[:, 0][::-1]
        y_points = points[:, 1][::-1]

        # Plot points as circles
        alpha = alpha0
        for i in range(N):
            ax.plot(x_points[i], y_points[i], "o", alpha=alpha, color=color)
            alpha *= decay

        # Plot segments with decaying alpha
        alpha = alpha0
        for i in range(N - 1):
            ax.plot(x_points[i : i + 2], y_points[i : i + 2], linewidth=2, alpha=alpha, color=color)
            alpha *= decay

    def plot_colormap_trajectory(ax, points, colors=("#3fb950", "#f78166"), lw=1.5, alpha=1.0):
        """Plot a 2D trajectory coloring each segment by step index (early=colors[0], late=colors[1])."""
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        points = np.asarray(points)
        N = len(points)
        if N < 2:
            return
        cmap = LinearSegmentedColormap.from_list("traj", list(colors))
        segments = np.stack([points[:-1], points[1:]], axis=1)
        t = np.linspace(0, 1, N - 1)
        lc = LineCollection(segments, cmap=cmap, norm=Normalize(0, 1), linewidth=lw, alpha=alpha)
        lc.set_array(t)
        ax.add_collection(lc)

    return (plot_colormap_trajectory,)


@app.cell
def _(np, scp):
    def generate_ellipse_grid(
        a: float,
        b: float,
        angle: float,
        center: np.ndarray = np.zeros((2,)),
        r_low: float = 0.1,
        r_high: float = 1,
        n_r: int = 100,
        n_theta: int = 50,
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


    def grad_flow_traj(x0, sigma, t):
        dt = t[1] - t[0]
        x_traj = np.array([scp.linalg.expm(-2 * sigma * tau) @ x0 for tau in t])

        return x_traj


    def grad_descent_traj(x0: np.ndarray, sigma: np.ndarray, eta: float, n_steps: int):
        x_traj = [x0]
        for _ in range(n_steps):
            x0 = x0 - eta * (2 * sigma @ x0)
            x_traj.append(x0)

        return np.array(x_traj)


    def make_quad_form(a, b, theta):
        D = np.diag(np.array([1 / a**2, 1 / b**2]))
        V = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        return V @ D @ V.T

    return (
        batch_quad_form,
        generate_ellipse_grid,
        grad_descent_traj,
        grad_flow_traj,
        make_quad_form,
    )


@app.cell
def _(Counter, Path, np, plt, re):
    TEXT_DIR = Path("/Users/chaaland/Documents/chaaland.github.io/assets/shared/txt")


    def zipf(ranks, K, alpha):
        return K * ranks**alpha


    def zipf_nlls_loss(ranks, freqs, K, alpha):
        yhat = zipf(ranks, K, alpha)
        residual = freqs - yhat
        mse = 0.5 * np.mean(np.square(residual))

        return mse


    def plot_zipf_param_contours(empirical_freqs: np.ndarray):
        K_plot = np.linspace(-0.03, 0.07, 150)
        alpha_plot = np.linspace(-1, -0.4, 150)

        K_mesh, Alpha_mesh = np.meshgrid(K_plot, alpha_plot)
        ranks = np.arange(1, empirical_freqs.size + 1)
        Z = np.empty_like(K_mesh)

        for i, alpha in enumerate(alpha_plot):
            for j, c in enumerate(K_plot):
                mse = zipf_nlls_loss(ranks, empirical_freqs, c, alpha)
                Z[i, j] = mse

        fig = plt.figure(figsize=(6, 6))
        plt.contour(K_mesh, Alpha_mesh, Z, levels=10 ** np.linspace(-6, -4, 50))

        plt.title(r"$\frac{1}{2N}\sum_{i=1}^N (f_i - K r_i^{\alpha_i})^2$")
        plt.xlabel(r"$K$", fontsize=14)
        plt.ylabel(r"$\alpha$", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        return plt.gca()


    def word_freqs(topk: int = 25):
        with open(TEXT_DIR / "hamlet.txt", "rt") as f:
            text = " ".join(f.readlines())
            text = re.sub("[^0-9a-zA-Z ]+", "", text)
            all_words = [word.lower() for word in text.split() if len(word) > 0]
            n_words = len(all_words)
            word_freqs = Counter(all_words)
            for word in word_freqs.keys():
                word_freqs[word] /= n_words

        words, freqs = zip(*word_freqs.most_common(topk))
        data = {"words": words, "freqs": freqs}
        return data

    return plot_zipf_param_contours, word_freqs, zipf, zipf_nlls_loss


@app.cell
def quadratic_grad_flow(
    batch_quad_form,
    generate_ellipse_grid,
    grad_flow_traj,
    make_quad_form,
    np,
    plt,
    remove_spines,
):
    _a = 1.0 / np.sqrt(8.0)
    _b = np.sqrt(2.0)
    _theta = np.pi / 6
    _sigma = 0.5 * make_quad_form(a=_a, b=_b, theta=_theta)
    _x0 = np.array([2.0, -1.5])

    _t = np.linspace(0, 4, 500)
    _x_gf = grad_flow_traj(_x0, _sigma, _t)

    _X, _Y = generate_ellipse_grid(a=_a, b=_b, angle=_theta, r_low=0.01, r_high=5, n_r=200, n_theta=200)
    _Z = batch_quad_form(np.stack([_X.ravel(), _Y.ravel()], axis=1).T, _sigma).reshape(_X.shape)

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.contour(_X, _Y, _Z, levels=10, linewidths=0.8)
    _ax.plot(_x_gf[:, 0], _x_gf[:, 1], color="tab:blue", linewidth=2)
    _ax.plot(*_x0, "o", color="tab:green", markersize=8, zorder=5)
    _ax.set_xlim([-4, 4])
    _ax.set_ylim([-4, 4])
    _ax.set_aspect("equal")
    _ax.set_xlabel("$x_1$")
    _ax.set_ylabel("$x_2$")
    _ax.set_title("Gradient flow")
    _ax.legend(frameon=False)
    remove_spines(_ax)
    _fig.tight_layout()
    _ax
    return


@app.cell
def quadratic_central_flow(
    batch_quad_form,
    generate_ellipse_grid,
    grad_descent_traj,
    make_quad_form,
    np,
    plt,
    remove_spines,
):
    _a = 1.0 / np.sqrt(8.0)  # 1/a² = 8  → eigenvalue λ₁ = 8
    _b = np.sqrt(2.0)  # 1/b² = 0.5 → eigenvalue λ₂ = 0.5
    _theta = np.pi / 6  # 30° rotation
    _eta = 1.9 / 8.0  # update factor (1 - η·λ₁) = -0.9

    _sigma = 0.5 * make_quad_form(a=_a, b=_b, theta=_theta)
    _x0 = np.array([2.0, -1.5])
    _x_gd = grad_descent_traj(x0=_x0, sigma=_sigma, eta=_eta, n_steps=30)

    # Central flow: 2-step moving average
    _x_cf = np.array([_x_gd[max(0, k - 1) : k + 1].mean(axis=0) for k in range(len(_x_gd))])

    _X, _Y = generate_ellipse_grid(a=_a, b=_b, angle=_theta, r_low=0.01, r_high=5, n_r=200, n_theta=200)
    _Z = batch_quad_form(np.stack([_X.ravel(), _Y.ravel()], axis=1).T, _sigma).reshape(_X.shape)

    _fig, _ax = plt.subplots(figsize=(5, 5))
    # _ax.contour(_X, _Y, _Z, levels=[0.25, 0.6, 1.4, 3.2, 7.0, 14.0], linewidths=0.8)
    _ax.contour(_X, _Y, _Z, levels=10, linewidths=0.8)
    _ax.plot(_x_gd[:, 0], _x_gd[:, 1], "o-", color="tab:orange", linewidth=1.5, markersize=4, alpha=0.7)
    _ax.plot(_x_cf[:, 0], _x_cf[:, 1], "o-", color="tab:blue", linewidth=2.5, markersize=5)
    _ax.plot(*_x0, "o", color="tab:green", markersize=8, zorder=5)
    _ax.set_xlim([-4, 4])
    _ax.set_ylim([-4, 4])
    _ax.set_aspect("equal")
    _ax.set_xlabel("$x_1$")
    _ax.set_ylabel("$x_2$")
    _ax.set_title("Approx central flow (2-point avg)")
    remove_spines(_ax)
    _fig.tight_layout()
    _ax
    return


@app.cell
def quadratic_loss_comparison(
    batch_quad_form,
    grad_descent_traj,
    grad_flow_traj,
    make_quad_form,
    np,
    plt,
    remove_spines,
):
    _a = 1.0 / np.sqrt(8.0)
    _b = np.sqrt(2.0)
    _theta = np.pi / 6
    _eta = 1.9 / 8.0
    _n_steps = 30

    _sigma = 0.5 * make_quad_form(a=_a, b=_b, theta=_theta)
    _x0 = np.array([2.0, -1.5])

    _x_gd = grad_descent_traj(x0=_x0, sigma=_sigma, eta=_eta, n_steps=_n_steps)
    _x_cf = np.array([_x_gd[max(0, k - 1) : k + 1].mean(axis=0) for k in range(len(_x_gd))])

    # Align gradient flow to discrete step axis: step k ↔ t = k·η
    _t = np.arange(_n_steps + 1) * _eta
    _x_gf = grad_flow_traj(_x0, _sigma, _t)


    def _loss(x_traj):
        return np.array([batch_quad_form(x.reshape(2, 1), _sigma)[0] for x in x_traj])


    _steps = np.arange(_n_steps + 1)
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.semilogy(_steps, _loss(_x_gd), color="tab:orange", linewidth=1.5, alpha=0.7, label="Gradient descent")
    _ax.semilogy(_steps, _loss(_x_cf), color="tab:blue", linewidth=2, label="Central flow (2-pt avg)")
    _ax.semilogy(_steps, _loss(_x_gf), color="tab:purple", linewidth=2, linestyle="--", label="Gradient flow")
    _ax.set_xlabel("Step $k$")
    _ax.set_ylabel("$f(x_k)$")
    _ax.set_title("Loss: gradient descent, central flow, gradient flow")
    _ax.legend(frameon=False)
    remove_spines(_ax)
    _fig.tight_layout()
    _ax
    return


@app.cell
def grad_flow_nlls(
    freqs,
    functools,
    grad_flow_traj_zipf_nlls,
    grad_zipf_nlls,
    mo,
    np,
    plot_zipf_param_contours,
    plt,
):
    def plot_zipf_nlls_grad_flow(freqs, ranks, x0):
        _eta = 8e-5
        grad_fn = functools.partial(grad_zipf_nlls, freqs=freqs, ranks=ranks)

        ax = plot_zipf_param_contours(np.array(freqs))
        x_traj_grad_flow = grad_flow_traj_zipf_nlls(x0, grad_fn, eta=_eta, n_steps=5000)
        ax.plot(x_traj_grad_flow[:, 0], x_traj_grad_flow[:, 1], linewidth=2)

        ax.set_xlim([-0.03, 0.07])
        ax.set_ylim([-1, -0.4])
        plt.tight_layout()

        return ax


    _x0 = np.array([-0.02, -0.9])
    _ax1 = plot_zipf_nlls_grad_flow(freqs, np.arange(1, freqs.size + 1), x0=_x0)
    mo.hstack([_ax1])
    return


@app.cell
def _(np, zipf):
    def grad_flow_traj_zipf_nlls(x0, grad_fn, eta, n_steps=100):
        x_traj = [x0]
        for _ in range(n_steps):
            nabla = grad_fn(K=x0[0], alpha=x0[1])
            nabla /= np.linalg.norm(nabla)
            x0 = x0 - eta * nabla
            x_traj.append(x0)

        return np.array(x_traj)


    def grad_zipf_nlls(freqs, ranks, K, alpha):
        N = len(freqs)
        y_hat = zipf(ranks, K, alpha)
        residual = freqs - y_hat  # (N,)
        dK = -((ranks**alpha) * residual).sum() / N
        dalpha = -(np.log(ranks) * y_hat * residual).sum() / N
        return np.array([dK, dalpha])


    def grad_descent_traj_zipf_nlls(grad_fn, x0: np.ndarray, eta: float, n_steps: int):
        x_traj = [x0]
        for _ in range(n_steps):
            x0 = x0 - eta * grad_fn(K=x0[0], alpha=x0[1])
            x_traj.append(x0)

        return np.array(x_traj)


    def sharpness(hessian):
        eigvals, _ = np.linalg.eigh(hessian)
        return np.max(eigvals)


    def hessian_zipf_nlls(freqs, ranks, K, alpha):
        """
        Returns the 2x2 Hessian matrix of the objective:
            F(K, alpha) = (1/(2N)) * sum (y_i - K r_i^alpha)^2
        """

        N = len(ranks)
        r_alpha = ranks**alpha
        log_r = np.log(ranks)
        y_hat = K * r_alpha
        residual = y_hat - freqs

        # Second derivative w.r.t. K
        d2F_dK2 = (1.0 / N) * np.sum(r_alpha**2)

        # Mixed derivative
        d2F_dKdalpha = (1.0 / N) * np.sum(K * (r_alpha**2) * log_r + residual * r_alpha * log_r)

        # Second derivative w.r.t. alpha
        d2F_dalpha2 = (1.0 / N) * np.sum((K * r_alpha * log_r) ** 2 + residual * K * r_alpha * (log_r**2))

        # Assemble symmetric Hessian
        H = np.array([[d2F_dK2, d2F_dKdalpha], [d2F_dKdalpha, d2F_dalpha2]])

        return H

    return (
        grad_descent_traj_zipf_nlls,
        grad_flow_traj_zipf_nlls,
        grad_zipf_nlls,
        hessian_zipf_nlls,
    )


@app.cell
def _(np, word_freqs):
    N = 100
    data = word_freqs(N)
    freqs = np.array(data["freqs"])
    return (freqs,)


@app.cell
def _(mo):
    eta_cf_slider = mo.ui.slider(1, 100, 2, value=70)
    window_slider = mo.ui.slider(2, 50, 2, value=10)
    sigma_slider = mo.ui.slider(2, 100, 2, value=20)
    n_steps_slider = mo.ui.number(1, 800, step=1, value=450)
    return eta_cf_slider, n_steps_slider, sigma_slider, window_slider


@app.cell
def zipf_nlls_contours(
    eta_cf_slider,
    freqs,
    functools,
    grad_descent_traj_zipf_nlls,
    grad_zipf_nlls,
    mo,
    n_steps_slider,
    np,
    plot_colormap_trajectory,
    plot_zipf_param_contours,
    sigma_slider,
    window_slider,
):
    def windowed_average(traj, window):
        """Trailing moving average of iterates with given (even) window size."""
        n = len(traj)
        avg = np.empty_like(traj)
        for k in range(n):
            start = max(0, k - window + 1)
            avg[k] = traj[start : k + 1].mean(axis=0)
        return avg


    def gaussian_average(traj, sigma):
        """Trailing Gaussian-weighted average; weights decay as exp(-0.5*(lag/sigma)^2)."""
        n = len(traj)
        avg = np.empty_like(traj)
        for k in range(n):
            lags = np.arange(k + 1)[::-1]  # 0, 1, 2, ... lag from current step
            weights = np.exp(-0.5 * (lags / sigma) ** 2)
            weights /= weights.sum()
            avg[k] = (traj[: k + 1] * weights[:, None]).sum(axis=0)
        return avg


    def plot_central_flow_vs_gd(freqs, ranks, eta, window, sigma, n_steps):
        grad_fn = functools.partial(grad_zipf_nlls, freqs=freqs, ranks=ranks)
        x0 = np.array([0.0, -0.8])

        x_gd = grad_descent_traj_zipf_nlls(grad_fn, x0=x0, eta=eta, n_steps=n_steps)
        x_cf_win = windowed_average(x_gd, window)
        x_cf_gauss = gaussian_average(x_gd, sigma)

        ax_contour = plot_zipf_param_contours(np.array(freqs))
        plot_colormap_trajectory(ax_contour, x_gd, colors=("#3fb950", "#f78166"), lw=1.5, alpha=0.5)
        plot_colormap_trajectory(ax_contour, x_cf_win, colors=("#58a6ff", "#a371f7"), lw=2.5)
        # ax_contour.plot(
        #     x_cf_gauss[:, 0],
        #     x_cf_gauss[:, 1],
        #     color="tab:green",
        #     linewidth=2,
        #     linestyle="--",
        # )
        # ax_contour.set_xlim([-0.03, 0.07])
        # ax_contour.set_ylim([-1, -0.4])
        ax_contour.set_xlim([0.03, 0.07])
        ax_contour.set_ylim([-0.75, -0.62])
        ax_contour.legend(frameon=False)

        return ax_contour


    _ranks = np.arange(1, freqs.size + 1)
    _ax_contour = plot_central_flow_vs_gd(
        freqs,
        _ranks,
        eta_cf_slider.value,
        window_slider.value,
        sigma_slider.value,
        n_steps_slider.value,
    )
    mo.hstack(
        [
            _ax_contour,
            mo.vstack(
                [
                    mo.md(rf"$\eta$: {eta_cf_slider}"),
                    mo.md(rf"Window $W$: {window_slider}"),
                    mo.md(rf"Gaussian $\sigma$: {sigma_slider}"),
                    mo.md(rf"Steps: {n_steps_slider}"),
                ]
            ),
        ]
    )
    return


@app.cell
def zipf_nlls_sharpness_contours(
    freqs,
    hessian_zipf_nlls,
    mo,
    np,
    plt,
    remove_spines,
):
    _K_vals = np.linspace(-0.03, 0.07, 100)
    _alpha_vals = np.linspace(-1.0, -0.4, 100)
    _ranks = np.arange(1, freqs.size + 1)

    _K_mesh, _Alpha_mesh = np.meshgrid(_K_vals, _alpha_vals)
    _sharpness_grid = np.empty_like(_K_mesh)

    for _i, _alpha in enumerate(_alpha_vals):
        for _j, _K in enumerate(_K_vals):
            _H = hessian_zipf_nlls(freqs, _ranks, _K, _alpha)
            _eigvals = np.linalg.eigvalsh(_H)
            _sharpness_grid[_i, _j] = _eigvals[-1]

    _eta_ref = 70
    _eos_threshold = 2.0 / _eta_ref

    _fig, _ax = plt.subplots(figsize=(6, 5))
    _pcm = _ax.pcolormesh(
        _K_mesh, _Alpha_mesh, _sharpness_grid,
        cmap="inferno", shading="auto",
        vmin=0, vmax=np.percentile(_sharpness_grid, 98),
    )
    _fig.colorbar(_pcm, ax=_ax, label=r"$\lambda_{\max}(\nabla^2 f)$")
    _ax.contour(
        _K_mesh, _Alpha_mesh, _sharpness_grid,
        levels=[_eos_threshold], colors=["#58a6ff"], linewidths=2, linestyles="--",
    )
    _ax.set_xlabel(r"$K$")
    _ax.set_ylabel(r"$\alpha$")
    _ax.set_title(rf"Sharpness $\lambda_{{\max}}(H)$ — EOS boundary $2/\eta$ at $\eta={_eta_ref}$ (blue dashed)")
    remove_spines(_ax)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def zipf_nlls_loss_comparison(
    freqs,
    functools,
    grad_descent_traj_zipf_nlls,
    grad_flow_traj_zipf_nlls,
    grad_zipf_nlls,
    np,
    plt,
    remove_spines,
    zipf_nlls_loss,
):
    _eta = 70
    _n_steps = 2000
    _x0 = np.array([0.0, -0.8])
    _ranks = np.arange(1, freqs.size + 1)
    _grad_fn = functools.partial(grad_zipf_nlls, freqs=freqs, ranks=_ranks)

    _window = 20
    _x_gd = grad_descent_traj_zipf_nlls(_grad_fn, x0=_x0, eta=_eta, n_steps=_n_steps)
    _x_cf = np.array([_x_gd[max(0, k - _window + 1) : k + 1].mean(axis=0) for k in range(len(_x_gd))])
    _x_gf = grad_flow_traj_zipf_nlls(_x0, _grad_fn, eta=8e-4, n_steps=_n_steps)


    def _loss(x_traj):
        return np.array([zipf_nlls_loss(ranks=_ranks, freqs=freqs, K=x[0], alpha=x[1]) for x in x_traj])


    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.semilogy(_loss(_x_gd), color="tab:red", linewidth=1.5, alpha=0.5, label="Gradient descent")
    _ax.semilogy(_loss(_x_cf), color="tab:blue", linewidth=2, label=f"Central flow (W={_window})")
    _ax.semilogy(_loss(_x_gf), color="tab:purple", linewidth=2, linestyle="--", label="Gradient flow")
    _ax.set_xlabel("Step")
    _ax.set_ylabel("Loss")
    _ax.set_title("Zipf NLLS loss: gradient descent, central flow, gradient flow")
    _ax.legend(frameon=False)
    remove_spines(_ax)
    _fig.tight_layout()
    _ax
    return


@app.cell
def zipf_central_flow_ode(
    freqs,
    grad_zipf_nlls,
    hessian_zipf_nlls,
    np,
    plt,
    remove_spines,
    scp,
    zipf_nlls_loss,
):
    _eta = 70
    _x0 = np.array([0.0, -0.8])
    _ranks = np.arange(1, freqs.size + 1)


    def _grad_sharpness(K, alpha, eps=1e-5):
        def H(K, a):
            return hessian_zipf_nlls(freqs, _ranks, K, a)

        _, eigvecs = np.linalg.eigh(H(K, alpha))
        v = eigvecs[:, -1]

        dH_dK = (H(K + eps, alpha) - H(K - eps, alpha)) / (2 * eps)
        dH_dalpha = (H(K, alpha + eps) - H(K, alpha - eps)) / (2 * eps)

        return np.array([v @ dH_dK @ v, v @ dH_dalpha @ v])


    def _rhs(t, w):
        K, alpha = w
        grad_L = grad_zipf_nlls(freqs, _ranks, K, alpha)
        grad_S = _grad_sharpness(K, alpha)

        denom = grad_S @ grad_S
        sigma2 = max(0.0, 2 * (-grad_L @ grad_S) / denom) if denom > 1e-12 else 0.0

        return -_eta * (grad_L + 0.5 * sigma2 * grad_S)


    _sol = scp.integrate.solve_ivp(_rhs, t_span=(0, 1500), y0=_x0, method="RK45", max_step=1.0)
    _w = _sol.y.T  # shape (n_timepoints, 2)

    _loss_cf_ode = np.array([zipf_nlls_loss(_ranks, freqs, w[0], w[1]) for w in _w])

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.semilogy(_sol.t[:20], _loss_cf_ode[:20], color="tab:blue", linewidth=2, label="Central flow (ODE)")
    _ax.set_xlabel("Step")
    _ax.set_ylabel("Loss")
    _ax.set_title("Zipf NLLS central flow ODE")
    _ax.legend(frameon=False)
    remove_spines(_ax)
    _fig.tight_layout()
    _ax
    return


if __name__ == "__main__":
    app.run()
