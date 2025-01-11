import numpy as np


def lagrange_interpolate(x: np.ndarray, pts: list[tuple[int, int]]):
    assert all(len(p) == 2 for p in pts)

    result = 0
    for i, (x_i, y_i) in enumerate(pts):
        num = denom = 1
        for j, (x_j, _) in enumerate(pts):
            if i != j:
                num *= x - x_j
                denom *= x_i - x_j

        ell_i = num / denom
        result += y_i * ell_i

    return result
