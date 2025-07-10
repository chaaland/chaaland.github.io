import math

import numpy as np


def bwd_diff(x, i: int, order: int = 1):
    if i < 0:
        raise ValueError(f"Index must be positive, got {i}")

    if order < 0:
        raise ValueError(f"Backward difference order expected to be >0, got {order}")
    elif order == 0:
        return x[i]
    else:
        return bwd_diff(x, i, order=order - 1) - bwd_diff(x, i - 1, order=order - 1)


def create_matrix(n: int, h: int) -> np.ndarray:
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            A[i, j] = math.perm(i, j) * h**j

    return A


def newton_gregory(x: np.ndarray, x_0: float, ys: np.ndarray, h: float) -> np.ndarray:
    poly_order = len(ys) - 1
    u = (x - x_0) / h

    coeff = 1
    result = np.zeros_like(x)
    for k in range(poly_order + 1):
        result += coeff * bwd_diff(ys, k, order=k)
        coeff *= (u - k) / (k + 1)

    return result
