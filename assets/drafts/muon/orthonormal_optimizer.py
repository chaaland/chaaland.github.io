import marimo

__generated_with = "0.17.0"
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
    return Counter, Path, functools, mo, np, plt, re


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
def _(np):
    def orthogonalize(G):
        u, _, v = np.linalg.svd(G, full_matrices=False)

        return u @ v.T


    def newton_schulz_orthogonalization(M, num_steps=5, coefficients=(3.4445, -4.7750, 2.0315)):
        # Check if the matrix is valid (at least 2D)
        if M.ndim < 2:
            raise ValueError("Input matrix M must be at least 2-dimensional.")

        # 1. Normalization (Pre-conditioning)
        # The iteration only converges if the largest singular value is less than ~1.73.
        # It is standard practice to normalize M by its Frobenius norm.
        frob_norm = np.linalg.norm(M, ord="fro")

        # Handle the zero matrix case to prevent division by zero
        if frob_norm == 0:
            return M

        X = M / frob_norm

        m, n = X.shape
        is_transposed = False
        if m > n:
            X = X.T
            is_transposed = True

        a, b, c = coefficients

        # 3. Newton-Schulz Iteration (Quintic Polynomial)
        # The iteration is based on the recurrence: X_k+1 = a*X_k + X_k @ P(X_k^T X_k)
        # where P is a polynomial. We use the symmetric form:
        # X_k+1 = a*X_k + b * (X_k @ X_k.T) @ X_k + c * (X_k @ X_k.T @ X_k @ X_k.T) @ X_k

        for _ in range(num_steps):
            A = X @ X.T  # Gramian
            B = b * A + c * A @ A

            # The update step: X_k+1 = a * X_k + B @ X_k
            # This is the core matrix multiplication-only update.
            X = a * X + B @ X

        if is_transposed:
            X = X.T

        return X
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
    return plot_zipf_param_contours, word_freqs, zipf


@app.cell
def _(np, zipf):
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
    return grad_descent_traj_zipf_nlls, grad_zipf_nlls


@app.cell
def _(np, word_freqs):
    N = 100
    data = word_freqs(N)
    freqs = np.array(data["freqs"])
    return (freqs,)


@app.cell
def _(mo):
    eta_slider = mo.ui.slider(1, 100, 2)
    return (eta_slider,)


@app.cell
def _(
    eta_slider,
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
        plot_fading_trajectory(ax=ax, points=_x_descent_traj, decay=0.999)

        ax.set_xlim([-0.03, 0.07])
        ax.set_ylim([-1, -0.4])

        return ax


    _ax = plot_zipf_nlls_grad_descent(freqs, np.arange(1, freqs.size + 1), eta_slider.value)
    mo.hstack([_ax, mo.md(rf"$\eta$: {eta_slider}")])
    return


@app.cell
def _():
    return


@app.cell
def _(u):
    u
    return


if __name__ == "__main__":
    app.run()
