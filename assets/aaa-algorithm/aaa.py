import numpy as np


def aaa_iter_(z: np.ndarray, y: np.ndarray, max_error_index: int, support_mask: np.ndarray):
    if z.size != y.size:
        raise ValueError("Expected z and y to be the same size, got `{z.size}` and `{y.size}`.")

    support_mask[max_error_index] = True

    z_support = z[support_mask]
    y_support = y[support_mask]

    z_tilde = z[~support_mask]
    y_tilde = y[~support_mask]

    cauchy_matrix = 1 / (z_tilde[:, None] - z_support[None, :])  # (M-m, m)
    loewner_matrix = y_tilde[:, None] * cauchy_matrix - cauchy_matrix * y_support[None, :]
    _, _, v_tranpose = np.linalg.svd(loewner_matrix, full_matrices=False)
    w = v_tranpose[-1, :]  # smallest singular vector (m,)

    numerator = cauchy_matrix @ (w * z_support)  # (M-m, m) @ (m,) -> (M-m,)
    denominator = cauchy_matrix @ w  # (M-m, m) @ (m,) -> (M-m,)
    rational = y.copy()
    rational[~support_mask] = numerator / denominator  # (M-m,)
    error = rational - y

    return w, rational, error


def aaa(f, z, tol: float = 1e-9, max_degree: int = 100):
    M = z.size
    y = f(z)

    support_mask = np.zeros(M, dtype=bool)
    error = y - np.mean(y)  # (M,)

    for m in range(max_degree):
        y_hat, error = aaa_iter_(support_mask, error)
        if np.linalg.nrom(error, "inf") < tol * np.linalg.norm(y, "inf"):
            break

    return y_hat


def simple_ols(f, z, m) -> tuple[np.ndarray, np.ndarray]:
    y = f(z)  # (M,)
    A = np.vander(z, m + 1, increasing=True)  # (M, m+1)

    # f(x_k) = a_0 + a_1 * x_k + ... + a_m * x_k**m - b_1 * f(x_k) * x_k - ... - b_M * f(x_k) * x_k ** m
    A = np.concat([A, -y[:, None] * A[:, 1:]], axis=1)  # (M, 2*m + 1)

    theta, _, _, _ = np.linalg.lstsq(A, y)

    # breakpoint()
    return theta[: m + 1], theta[m + 1 :]
