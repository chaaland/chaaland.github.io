import numpy as np


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    x_bar = x.sum() / n
    y_bar = y.sum() / n

    x_deviation = x - x_bar
    y_deviation = y - y_bar

    cov_xy = (x_deviation * y_deviation).sum() / n
    var_x = (x_deviation * x_deviation).sum() / n
    var_y = (y_deviation * y_deviation).sum() / n

    rho = cov_xy / (var_x * var_y) ** 0.5

    return rho


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    assert len(x) == len(set(x))
    assert len(y) == len(set(y))

    n = x.size

    rank_x = np.argsort(x)
    rank_y = np.argsort(y)

    d_sq = ((rank_x - rank_y) ** 2).sum()

    corr = 1 - d_sq / (n * (n**2 - 1) / 6)

    return corr


def chaterjees_xi(x: np.ndarray, y: np.ndarray) -> float:
    assert len(x) == len(set(x))
    assert len(y) == len(set(y))

    n = x.size
    rank_x = np.argsort(x)

    y_sorted = y[rank_x]

    xi_corr = 1 - np.abs(y_sorted[:-1] - y_sorted[1:]).sum() / ((n**2 - 1) / 3)

    return xi_corr
