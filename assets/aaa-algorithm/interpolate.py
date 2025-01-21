from typing import Callable

import numpy as np


def make_cauchy_matrix(z1: np.ndarray, z2: np.ndarray):
    return 1 / (z1[:, None] - z2[None, :])


def make_loewner_matrix(y1: np.ndarray, y2: np.ndarray, z1: np.ndarray, z2: np.ndarray):
    C = make_cauchy_matrix(z1, z2)
    loewner_matrix = y1[:, None] * C - C * y2[None, :]

    return loewner_matrix


def compute_weights(z_tilde, y_tilde, z_support, y_support):
    loewner_matrix = make_loewner_matrix(y_tilde, y_support, z_tilde, z_support)
    _, _, v_tranpose = np.linalg.svd(loewner_matrix, full_matrices=False)
    w = v_tranpose[-1, :]  # smallest singular vector (m,)
    return w


def aaa_iter_(z: np.ndarray, y: np.ndarray, max_error_index: int, support_mask: np.ndarray):
    if z.size != y.size:
        raise ValueError("Expected z and y to be the same size, got `{z.size}` and `{y.size}`.")

    support_mask[max_error_index] = True

    z_support = z[support_mask]
    y_support = y[support_mask]

    z_tilde = z[~support_mask]
    y_tilde = y[~support_mask]

    w = compute_weights(z_tilde, y_tilde, z_support, y_support)

    cauchy_matrix = make_cauchy_matrix(z_tilde, z_support)  # (M-m, m)
    numerator = cauchy_matrix @ (w * y_support)  # (M-m, m) @ (m,) -> (M-m,)
    denominator = cauchy_matrix @ w  # (M-m, m) @ (m,) -> (M-m,)
    rational = y.copy()
    rational[~support_mask] = numerator / denominator  # (M-m,)
    error = rational - y

    return w, rational, error


def aaa(f: Callable, z: np.ndarray, tol: float = 1e-9, max_degree: int = 100):
    M = z.size
    y = f(z)

    support_mask = np.zeros(M, dtype=bool)
    error = y - np.mean(y)  # (M,)

    threshold = tol * np.linalg.norm(y, ord=np.inf)
    for m in range(max_degree):
        max_error_index = np.argmax(np.abs(error)).item()
        w, y_hat, error = aaa_iter_(z, y, max_error_index, support_mask)
        max_abs_error = np.linalg.norm(error, ord=np.inf)
        if max_abs_error < threshold:
            break

    z_support = z[support_mask]
    return w, z_support


def simple_ols(f, z, m) -> tuple[np.ndarray, np.ndarray]:
    y = f(z)  # (M,)
    A = np.vander(z, m + 1, increasing=True)  # (M, m+1)

    # f(x_k) = a_0 + a_1 * x_k + ... + a_m * x_k**m - b_1 * f(x_k) * x_k - ... - b_M * f(x_k) * x_k ** m
    A = np.concat([A, -y[:, None] * A[:, 1:]], axis=1)  # (M, 2*m + 1)

    theta, _, _, _ = np.linalg.lstsq(A, y)

    return theta[: m + 1], theta[m + 1 :]
