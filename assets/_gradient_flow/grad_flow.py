import marimo

__generated_with = "0.18.4"
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
    return make_cartesian_plane, remove_spines


@app.cell
def sharpness_1d(make_cartesian_plane, np, plt):
    def plot_1d_sharpness():
        for _S in [0.5, 1, 2, 5]:
            x = np.linspace(-2, 2, 100)
            y = 0.5 * _S * x**2
            plt.plot(x, y, label=f"S={_S}")

        make_cartesian_plane(plt.gca())
        plt.xlim([-2, 3])
        plt.ylim([-0.1, 4])
        plt.legend(frameon=False, loc="upper right")
        plt.title(r"$y=\frac{S}{2} x^2$")
        return plt.gca()


    plot_1d_sharpness()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Gradient descent on quadratic is

    $$x_{k+1} = (1- S\eta) x_k.$$

    This will fail to converge when

    $$|1 - S\eta| \ge 1.$$

    Since $S,\eta>0$, this occurs when $\eta \ge 2 / S$
    """)
    return


@app.cell
def _(np):
    def plot_fading_trajectory(ax, points, alpha0=1.0, decay=0.9):
        """Plot a 2D trajectory with fading opacity for each step."""

        points = np.asarray(points)
        N = len(points)

        x_points = points[:, 0][::-1]
        y_points = points[:, 1][::-1]

        # Plot points as circles
        alpha = alpha0
        for i in range(N):
            ax.plot(x_points[i], y_points[i], "o", alpha=alpha, color="tab:red")
            alpha *= decay

        # Plot segments with decaying alpha
        alpha = alpha0
        for i in range(N - 1):
            ax.plot(x_points[i : i + 2], y_points[i : i + 2], linewidth=2, alpha=alpha, color="tab:red")
            alpha *= decay
    return (plot_fading_trajectory,)


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
def _(mo, np):
    theta_slider = mo.ui.slider(0, np.pi, 0.1)
    a_slider = mo.ui.slider(1, 4, 0.1)
    b_slider = mo.ui.slider(1, 4, 0.1)
    eta_slider = mo.ui.slider(0.1, 3, 0.1)
    return a_slider, b_slider, eta_slider, theta_slider


@app.cell
def quadratic_minimize(
    a_slider,
    b_slider,
    batch_quad_form,
    eta_slider,
    generate_ellipse_grid,
    grad_descent_traj,
    grad_flow_traj,
    make_quad_form,
    mo,
    np,
    plot_fading_trajectory,
    plt,
    remove_spines,
    theta_slider,
):
    def plot_ellipse_contours(a, b, theta, eta, x0):
        X, Y = generate_ellipse_grid(
            a=a,
            b=b,
            angle=theta,
            r_low=0.01,
            r_high=5,
            n_r=200,
            n_theta=200,
        )

        A = 0.5 * make_quad_form(a=a, b=b, theta=theta)  # want 0.5 x^T A x
        Z = batch_quad_form(np.stack([X.ravel(), Y.ravel()], axis=1).T, A).reshape(X.shape)

        x_traj = grad_flow_traj(x0=x0, sigma=A, t=np.linspace(0, 15, 100))
        x_descent_traj = grad_descent_traj(x0=x0, sigma=A, eta=eta, n_steps=8)

        plt.figure(figsize=(4, 4))
        plt.contour(X, Y, Z, levels=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 3])
        plt.plot(x_traj[:, 0], x_traj[:, 1], linewidth=2)
        plot_fading_trajectory(plt.gca(), x_descent_traj, decay=0.9)

        _t = np.linspace(-5, 5, 100)
        plt.plot(_t, np.tan(theta) * _t, "k--", alpha=0.5)
        plt.plot(_t, np.tan(theta + np.pi / 2) * _t, "k--", alpha=0.5)

        plt.tight_layout()
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.gca().set_aspect("equal", adjustable="box")

        _lambda_a = 1 / a**2
        _lambda_b = 1 / b**2
        plt.title(
            rf"a={a} ($\lambda_a$={_lambda_a:.2f})"
            "\n"
            rf"b={b} ($\lambda_b$={_lambda_b:.2f})"
            "\n"
            rf"$\theta$={theta}"
        )
        return plt.gca()


    def plot_parabaloid_loss(a, b, theta, eta, x0):
        X, Y = generate_ellipse_grid(
            a=a,
            b=b,
            angle=theta,
            r_low=0.01,
            r_high=5,
            n_r=200,
            n_theta=200,
        )
        A = 0.5 * make_quad_form(a=a, b=b, theta=theta)
        Z = batch_quad_form(np.stack([X.ravel(), Y.ravel()], axis=1).T, A).reshape(X.shape)

        x_traj = grad_flow_traj(x0=x0, sigma=A, t=np.linspace(0, 15, 100))
        x_descent_traj = grad_descent_traj(x0=x0, sigma=A, eta=eta, n_steps=8)
        plt.figure(figsize=(4, 4))
        plt.semilogy(batch_quad_form(x_descent_traj.T, A))
        plt.scatter(np.arange(x_descent_traj.shape[0]), batch_quad_form(x_descent_traj.T, A))

        remove_spines(plt.gca())
        plt.title("Loss vs Step")

        return plt.gca()


    _ax1 = plot_ellipse_contours(
        a=a_slider.value, b=b_slider.value, theta=theta_slider.value, eta=eta_slider.value, x0=np.array([-1, 2])
    )
    _ax2 = plot_parabaloid_loss(
        a_slider.value, b=b_slider.value, theta=theta_slider.value, eta=eta_slider.value, x0=np.array([-1, 2])
    )

    _v = mo.vstack(
        [
            mo.md(rf"$\theta$: {theta_slider}"),
            mo.md(rf"$a$: {a_slider}"),
            mo.md(rf"$b$: {b_slider}"),
            mo.md(rf"$\eta$: {eta_slider}"),
        ]
    )
    mo.hstack([_ax1, _ax2, _v])
    return


@app.cell
def _(Counter, Path, np, plt, re):
    TEXT_DIR = Path("txt")


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
def _(np, plt, zipf):
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


    def plot_zipf_nlls_hessian_contours(empirical_freqs: np.ndarray):
        K_plot = np.linspace(-0.03, 0.07, 150)
        alpha_plot = np.linspace(-1, -0.4, 150)

        K_mesh, Alpha_mesh = np.meshgrid(K_plot, alpha_plot)
        ranks = np.arange(1, empirical_freqs.size + 1)
        Z = np.empty_like(K_mesh)

        for i, alpha in enumerate(alpha_plot):
            for j, c in enumerate(K_plot):
                H = hessian_zipf_nlls(ranks, empirical_freqs, c, alpha)
                Z[i, j] = sharpness(H)

        fig = plt.figure(figsize=(6, 6))
        plt.contour(K_mesh, Alpha_mesh, Z, levels=50)

        plt.grid(True)
        plt.tight_layout()
        return plt.gca()
    return (
        grad_descent_traj_zipf_nlls,
        grad_flow_traj_zipf_nlls,
        grad_zipf_nlls,
        hessian_zipf_nlls,
        sharpness,
    )


@app.cell
def _(np, word_freqs):
    N = 100
    data = word_freqs(N)
    freqs = np.array(data["freqs"])
    return (freqs,)


@app.cell
def _(mo):
    eta_2_slider = mo.ui.slider(1, 100, 2)
    return (eta_2_slider,)


@app.cell
def _(
    eta_2_slider,
    freqs,
    functools,
    grad_descent_traj_zipf_nlls,
    grad_zipf_nlls,
    mo,
    np,
    plot_fading_trajectory,
    plot_zipf_param_contours,
):
    def plot_zipf_nlls_grad_descent(freqs, ranks, eta):
        _grad_fn = functools.partial(grad_zipf_nlls, freqs=freqs, ranks=np.arange(1, freqs.size + 1))
        _x0 = np.array([0.0, -0.8])
        _x_descent_traj = grad_descent_traj_zipf_nlls(_grad_fn, x0=_x0, eta=eta, n_steps=500 * 3)
        ax = plot_zipf_param_contours(np.array(freqs))
        plot_fading_trajectory(ax=ax, points=_x_descent_traj, decay=0.997)

        ax.set_xlim([-0.03, 0.07])
        ax.set_ylim([-1, -0.4])

        return ax


    _ax = plot_zipf_nlls_grad_descent(freqs, np.arange(1, freqs.size + 1), eta_2_slider.value)
    mo.hstack([_ax, mo.md(rf"$\eta$: {eta_2_slider}")])
    return


@app.cell
def _(np):
    def ewma(x, beta):
        x = np.asarray(x)
        n = len(x)

        # EWMA kernel — no reverse!
        w = beta ** np.arange(n)

        # Convolution (kernel implicitly flipped)
        y_full = np.convolve(x, w, mode="full")

        # Take the causal part of correct length
        return y_full[:n] / np.cumsum(w)
    return (ewma,)


@app.cell
def _(
    ewma,
    freqs,
    functools,
    grad_descent_traj_zipf_nlls,
    grad_zipf_nlls,
    hessian_zipf_nlls,
    np,
    plt,
    remove_spines,
    sharpness,
    zipf_nlls_loss,
):
    def loss_and_sharpness_plots(freqs, ranks, x0, eta, n_steps=2500):
        grad_fn = functools.partial(grad_zipf_nlls, freqs=freqs, ranks=ranks)
        hessian_fn = functools.partial(hessian_zipf_nlls, ranks=np.arange(1, freqs.size + 1), freqs=freqs)
        x_descent_traj = grad_descent_traj_zipf_nlls(grad_fn, x0=x0, eta=eta, n_steps=n_steps)
        sharpness_vals = np.array([sharpness(hessian_fn(K=c, alpha=alpha)) for c, alpha in x_descent_traj])
        loss_vals = np.array([zipf_nlls_loss(ranks=ranks, freqs=freqs, K=_x[0], alpha=_x[1]) for _x in x_descent_traj])

        plt.figure(figsize=(8, 5))
        plt.subplot(211)
        plt.semilogy(loss_vals, label="Loss")
        plt.plot(ewma(loss_vals, beta=0.999), label=r"$\text{EWMA}_{\beta}$[Loss]")
        plt.title("Loss vs Step", fontsize=14)
        plt.xlim([0, n_steps])
        remove_spines(plt.gca())
        plt.ylabel("Loss", fontsize=14)
        plt.legend(frameon=False)


        plt.subplot(212)
        plt.plot(sharpness_vals)
        plt.gca().axhline(y=2 / eta, color="k", linestyle="--", alpha=0.7, label=rf"$S=2/\eta\approx${2/eta:.2e}")
        plt.xlim([0, n_steps])
        plt.title("Sharpness vs Step", fontsize=14)
        remove_spines(plt.gca())
        plt.tight_layout()
        plt.ylabel("Loss", fontsize=14)
        plt.xlabel("Step", fontsize=14)
        plt.legend(frameon=False)

        return plt.gca()


    loss_and_sharpness_plots(freqs, np.arange(1, freqs.size + 1), x0=np.array([0, -0.8]), eta=70, n_steps=2500)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Loss dynamics are explained by looking at the Taylor expansion. Specfically, denoting $g(x)=\nabla f(x)$, we can perform the second order expansion around the perturbed point $w_t+\delta_t$. Since this would lead to a third order tensor, it's easier to look at the $k^{th}$ coordinate of the gradient approximation centered at $w_t$

    $$g_k(w) \approx g_k(w_t) + Dg_k(w_t)(w-w_t) + \frac{1}{2}(w-w_t)^T \nabla^2 g_k(w_t) (w-w_t), \quad k=1,\ldots, d$$

    Evaluating at the perturbed point $w_t+\delta_t$,

    $$g_k(w_t+\delta_t) \approx g_k(w_t) + Dg_k(w_t)\delta_t + \frac{1}{2}\delta_t^T \nabla^2 g_k(w_t) \delta_t, \quad k=1,\ldots, d$$

    If the oscillation is purely along the top eigenvector of $\nabla^2f$, then $\delta_t = \sigma u$ where $u$ is a unit vector,

    $$\begin{align*}
    g_k(w_t+u) &\approx g_k(w_t) + \sigma Dg_k(w_t)u + \frac{1}{2}u^T \nabla^2 g_k(w_t) u\\
    &= g_k(w_t) + \sigma [\nabla^2 f(w_t)u]_k + \frac{\sigma^2}{2}\sum_{i=1}^d\sum_{j=1}^d \frac{\partial^3 f(w_t)}{\partial x_k \partial x_i \partial x_j} u_i u_j  \\
    &= g_k(w_t) + \sigma [S(w_t)u]_k + \frac{\sigma^2}{2}\left[\frac{\partial}{\partial x_k}\sum_{i=1}^d\sum_{j=1}^d \frac{\partial^2 f(x)}{\partial x_i \partial x_j} u_i u_j\right]_{x=w_t}  \\
    &= g_k(w_t) + \sigma S(w_t)u_k + \frac{\sigma^2}{2}\left[\frac{\partial}{\partial x_k}\left(u^T \nabla^2 f(x) u\right)\right]_{x=w_t}  \\
    &= g_k(w_t) + \sigma S(w_t)u_k + \frac{\sigma^2}{2}\frac{\partial S(w_t)}{\partial x_k}  \\
    \end{align*}
    $$

    So the final gradient approximation is

    $$g(w) \approx g(w_t) + \sigma S(w_t)u + \frac{\sigma^2}{2}\nabla S(w_t) $$
    """)
    return


@app.cell
def _(
    freqs,
    functools,
    grad_flow_traj_zipf_nlls,
    grad_zipf_nlls,
    mo,
    np,
    plot_zipf_param_contours,
    plt,
    remove_spines,
    zipf_nlls_loss,
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


    def plot_zipf_nlls_loss_grad_flow(freqs, ranks, x0):
        _eta = 8e-5

        plt.figure(figsize=(6, 6))

        grad_fn = functools.partial(grad_zipf_nlls, freqs=freqs, ranks=ranks)
        x_traj_grad_flow = grad_flow_traj_zipf_nlls(x0, grad_fn, eta=_eta, n_steps=5000)

        plt.semilogy(
            [
                zipf_nlls_loss(ranks=np.arange(1, freqs.size + 1), freqs=freqs, K=_x[0], alpha=_x[1])
                for _x in x_traj_grad_flow
            ]
        )
        plt.title("Zipf NLLS Loss")
        plt.tight_layout()

        remove_spines(plt.gca())
        return plt.gca()


    _x0 = np.array([-0.02, -0.9])
    _ax1 = plot_zipf_nlls_grad_flow(freqs, np.arange(1, freqs.size + 1), x0=_x0)
    _ax2 = plot_zipf_nlls_loss_grad_flow(freqs, np.arange(1, freqs.size + 1), x0=_x0)

    mo.hstack([_ax1, _ax2])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
