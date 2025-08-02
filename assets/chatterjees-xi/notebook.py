import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    from corr import pearson_corr, spearman_corr
    return mo, mpl, np, pearson_corr, plt, spearman_corr


@app.cell
def _(mpl, np, plt):
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=14)
    mpl.rcParams["lines.linewidth"] = 3


    def remove_spines(ax):
        ax.spines[["right", "top"]].set_visible(False)


    def make_cartesian_plane(ax):
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")


    def generate_correlated_data(
        n_samples: int, correlation: float, mean_x=0, mean_y=0, std_x=1, std_y=1, random_seed=None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not -1 <= correlation <= 1:
            raise ValueError("Correlation must be between -1 and 1.")

        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate two standard normal distributions
        x = np.random.randn(n_samples)
        z = np.random.randn(n_samples)

        # Construct y such that it has the desired correlation with x
        y = correlation * x + np.sqrt(1 - correlation**2) * z

        # Adjust to desired mean and standard deviation
        x = x * std_x + mean_x
        y = y * std_y + mean_y

        return x, y
    return generate_correlated_data, make_cartesian_plane, remove_spines


@app.cell
def _(mo):
    mo.md(r"""# Pearson's Correlation""")
    return


@app.cell
def _(generate_correlated_data, make_cartesian_plane, mo, plt):
    plt.figure(figsize=(6, 6))
    for _i, rho in enumerate([0.25, 0.5, 0.75, 0.95]):
        _x, _y = generate_correlated_data(n_samples=50, correlation=rho)

        plt.subplot(2, 2, _i + 1)
        plt.title(rf"$\rho$={rho}", loc="left")
        plt.scatter(_x, _y, alpha=0.5)
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])
        make_cartesian_plane(plt.gca())

    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo):
    rho_slider = mo.ui.slider(-3, 3, 0.01)
    intercept_slider = mo.ui.slider(-1, 1, 0.1)
    return


@app.cell
def _(mo, np, pearson_corr, plt, remove_spines):
    plt.figure(figsize=(9, 3))

    plt.subplot(131)
    _xs = np.linspace(-1, 1, 20)
    _ys = _xs**2
    plt.scatter(_xs, _ys)
    remove_spines(plt.gca())
    _rho = pearson_corr(_xs, _ys)
    plt.title(rf"$\rho$={_rho:.2f}")

    plt.subplot(132)
    _xs = np.random.randn(20)
    _ys = 1 / (1 + np.exp(-5 * _xs))
    plt.scatter(_xs, _ys)
    remove_spines(plt.gca())

    _rho = pearson_corr(_xs, _ys)
    plt.title(rf"$\rho$={_rho:.2f}")

    plt.subplot(133)
    _xs = np.linspace(-2 * np.pi, 2 * np.pi, 30)
    _ys = np.sin(_xs)
    plt.scatter(_xs, _ys)
    remove_spines(plt.gca())

    _rho = pearson_corr(_xs, _ys)
    plt.title(rf"$\rho$={_rho:.2f}")
    plt.tight_layout()

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo):
    mo.md(r"""# Spearman's Correlation""")
    return


@app.cell
def _(generate_correlated_data, make_cartesian_plane, mo, plt, spearman_corr):
    plt.figure(figsize=(6, 6))
    for _i, _rho in enumerate([0.25, 0.5, 0.75, 0.95]):
        _x, _y = generate_correlated_data(n_samples=50, correlation=_rho)

        plt.subplot(2, 2, _i + 1)
        _tau = spearman_corr(_x, _y)
        plt.title(rf"$\rho={_rho}/\tau$={_tau:.2f}", loc="left")
        plt.scatter(_x, _y, alpha=0.5)
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])
        make_cartesian_plane(plt.gca())

    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo, np, plt, remove_spines, spearman_corr):
    plt.figure(figsize=(9, 3))

    plt.subplot(131)
    _xs = np.linspace(-1, 1 - 0.001, 20)
    _ys = _xs**2
    plt.scatter(_xs, _ys)
    remove_spines(plt.gca())
    _tau = spearman_corr(_xs, _ys)
    plt.title(rf"$\tau$={_tau:.2f}")

    plt.subplot(132)
    _xs = np.random.randn(20)
    _ys = 1 / (1 + np.exp(-5 * _xs))
    plt.scatter(_xs, _ys)
    remove_spines(plt.gca())
    _tau = spearman_corr(_xs, _ys)
    plt.title(rf"$\tau$={_tau:.2f}")

    plt.subplot(133)
    _xs = np.linspace(-2 * np.pi, 2 * np.pi - 0.0001, 30)
    _ys = np.sin(_xs)
    plt.scatter(_xs, _ys)
    remove_spines(plt.gca())
    _tau = spearman_corr(_xs, _ys)
    plt.title(rf"$\tau$={_tau:.2f}")

    plt.tight_layout()

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(IMAGE_DIR, np, plt, remove_spines):
    def get_linear_data():
        np.random.seed(42)

        m = 3
        b = 2
        n = 20

        x = 2 * np.sort(np.random.rand(n)) - 1
        y = m * x + b

        return x, y


    def get_monotonic_data():
        np.random.seed(42)

        a = 3
        b = 2
        n = 40

        x = 5 * np.sort(np.random.rand(n))
        y = a * np.tanh(x - 1.5) + b

        return x, y


    def get_periodic_data():
        np.random.seed(42)

        a = 1.5
        b = 2
        omega = 1.5
        n = 50

        x = 8 * np.sort(np.random.rand(n))
        y = a * np.cos(omega * x) + b

        return x, y


    def plot_linear_data():
        x, y = get_linear_data()
        plt.figure(figsize=(8, 8))
        plt.scatter(x, y)
        remove_spines(plt.gca())

        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / "linear.png")


    def plot_monotonic_data():
        x, y = get_monotonic_data()

        plt.figure(figsize=(8, 8))
        plt.scatter(x, y)
        remove_spines(plt.gca())
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / "tanh.png")


    def plot_periodic_data():
        x, y = get_periodic_data()
        plt.figure(figsize=(8, 8))
        plt.scatter(x, y)
        remove_spines(plt.gca())

        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / "cosine.png")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
