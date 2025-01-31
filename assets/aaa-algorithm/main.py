from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from interpolate import aaa_iter_, simple_ols
from scipy.special import gamma

mpl.use("Agg")

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True, parents=True)

plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
plt.rc("ytick", labelsize=14)
mpl.rcParams["lines.linewidth"] = 3


def make_cartesian_plane(ax):
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")


def remove_spines(ax):
    ax.spines[["right", "top"]].set_visible(False)


def plot_gamma():
    xs = np.concatenate([np.linspace(i, i + 1, 500, endpoint=False) for i in range(-5, 5)])
    ys = gamma(xs)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys)
    plt.xlim([-5, 5])
    plt.ylim([-6, 6])
    make_cartesian_plane(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "gamma_function.png")


def plot_simple_ols(filename: str):
    xs = np.linspace(-1 + 0.01, 4, 1000)
    ys = np.log1p(xs)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys)
    x_vals = np.linspace(0, 1, 10)
    plt.scatter(x_vals, np.log1p(x_vals))

    degree_to_abs_error = {}
    for m in [1, 2]:
        x_vals = np.linspace(0, 1, 10)
        a, b = simple_ols(np.log1p, x_vals, m=m)
        A = np.vander(xs, m + 1, increasing=True)
        numerator = A @ a
        denominator = A[:, 1:] @ b
        rational = numerator / (1 + denominator)

        plt.plot(xs, rational, "--", label=f"degree={m}", alpha=0.7)
        print(f"{a=}")
        print(f"{b=}")
        degree_to_abs_error[m] = np.abs(rational - ys)

    plt.xlim([-1, 4])
    plt.ylim([-1, 2])
    plt.legend(frameon=False, fontsize=16)
    make_cartesian_plane(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / filename)

    plt.figure(figsize=(8, 8))
    for m, error in degree_to_abs_error.items():
        plt.plot(xs, error, label=f"degree={m}")

    plt.gca().set_yscale("log")
    remove_spines(plt.gca())
    plt.xlim([-1, 4])
    plt.tight_layout()
    plt.legend(frameon=False, fontsize=16)
    plt.savefig(IMAGE_DIR / "ols_logarithm_error.png")


def plot_ols_gamma(f, degree: int, filename: str):
    xs = np.concatenate([np.linspace(i + 0.001, i + 1, 500, endpoint=False) for i in range(-5, 5)])
    ys = gamma(xs)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys)

    a, b = simple_ols(gamma, xs, m=degree)
    A = np.vander(xs, degree + 1, increasing=True)
    numerator = A @ a
    denominator = A[:, 1:] @ b
    rational = numerator / (1 + denominator)

    plt.plot(xs, rational, "--")
    plt.xlim([-5, 5])
    plt.ylim([-6, 6])
    make_cartesian_plane(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / filename)

    return a, b


def plot_aaa_log():
    from interpolate import aaa_inference

    xs = np.linspace(-1 + 0.01, 4, 1000)
    ys = np.log1p(xs)

    z = np.linspace(0, 1, 10)
    N = z.size
    y = np.log1p(z)

    support_mask = np.zeros(N, dtype=bool)
    error = y - np.mean(y)  # (N,)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys)

    for m in [0, 1, 2]:
        max_error_index = np.argmax(np.abs(error)).item()
        w, _, error = aaa_iter_(z, y, max_error_index, support_mask)
        y_hat = aaa_inference(xs, z[support_mask], y[support_mask], w)

        if m == 0:
            continue

        plt.plot(xs, y_hat, "--", label=f"degree={m}")
        plt.xlim([-1, 4])
        plt.ylim([-1, 2])
        print(np.arange(N)[support_mask], w)

    make_cartesian_plane(plt.gca())
    plt.tight_layout()
    plt.legend(frameon=False, fontsize=16)
    plt.savefig(IMAGE_DIR / "aaa_logarithm.png")


def plot_aaa_gamma(f, degree: int):
    xs = np.concat([np.linspace(i + 0.01, i + 1, 25, endpoint=False) for i in range(-3, 5)])
    ys = gamma(xs)

    max_degree = 10
    tol = 1e-9
    z = xs
    M = z.size
    y = gamma(z)

    support_mask = np.zeros(M, dtype=bool)
    error = y - np.mean(y)  # (M,)

    threshold = tol * np.linalg.norm(y, ord=np.inf)
    for m in range(max_degree):
        max_error_index = np.argmax(np.abs(error)).item()
        w, y_hat, error = aaa_iter_(z, y, max_error_index, support_mask)

        plt.figure(figsize=(8, 8))
        plt.plot(xs, ys)
        plt.plot(xs, y_hat)
        plt.scatter(z[support_mask], y[support_mask])
        plt.xlim([-3, 5])
        plt.ylim([-10, 10])
        make_cartesian_plane(plt.gca())
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f"aaa_gamma_degree_{m:02}.png")
        # print(w)

        max_abs_error = np.linalg.norm(error, ord=np.inf)
        print(f"[{m=}] {max_abs_error:.6}")
        if max_abs_error < threshold:
            break


def make_splash_image():
    xs = np.linspace(-0.9 * np.pi / 2, 0.9 * np.pi / 2, 25, endpoint=False)
    ys = np.tan(xs)

    max_degree = 4
    tol = 1e-9
    z = xs
    M = z.size
    y = np.tan(z)

    support_mask = np.zeros(M, dtype=bool)
    error = y - np.mean(y)  # (M,)

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys)

    threshold = tol * np.linalg.norm(y, ord=np.inf)
    for m in range(max_degree):
        max_error_index = np.argmax(np.abs(error)).item()
        w, y_hat, error = aaa_iter_(z, y, max_error_index, support_mask)

        if m > 0:
            plt.plot(xs, y_hat, "--")

        max_abs_error = np.linalg.norm(error, ord=np.inf)
        print(f"[{m=}] {max_abs_error:.6}")
        if max_abs_error < threshold:
            break

    plt.scatter(z[support_mask], y[support_mask])
    plt.xlim([-1, 1])
    plt.ylim([-2, 2])
    make_cartesian_plane(plt.gca())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(IMAGE_DIR / "splash_image.png")


def main():
    make_splash_image()
    # plot_simple_ols("ols_logarithm.png")
    # plot_gamma()

    # for i in [10, 11]:
    #     plot_ols_gamma(gamma, degree=i, filename=f"ols_gamma_degree_{i:02}")

    # TODO caseyh: figure out to inference AAA after it fits
    # plot_aaa_log()
    # plot_aaa_log(np.log, degree=2)
    # plot_aaa_gamma(gamma, degree=2)


if __name__ == "__main__":
    main()
