import numpy as np


def compute_rank(x):
    n = len(x)
    rank_x = np.empty_like(x, dtype=int)
    rank_x[np.argsort(x)] = np.arange(1, n + 1)

    return rank_x


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
    rank_x = np.empty_like(x, dtype=int)
    rank_x[np.argsort(x)] = np.arange(1, n + 1)

    rank_y = np.empty_like(y, dtype=int)
    rank_y[np.argsort(y)] = np.arange(1, n + 1)
    rank_diff = rank_x - rank_y
    sq_rank_diff = rank_diff * rank_diff

    corr = 1 - sq_rank_diff.sum() / (n * (n**2 - 1) / 6)

    return corr


def chatterjee_corr(x: np.ndarray, y: np.ndarray) -> float:
    assert len(x) == len(set(x))
    assert len(y) == len(set(y))

    n = x.size
    y_ordered_by_x = y[np.argsort(x)]

    ranks = np.empty_like(x, dtype=int)
    ranks[np.argsort(y_ordered_by_x)] = np.arange(1, n + 1)
    abs_rank_diffs = np.abs(np.diff(ranks))

    xi_corr = 1 - abs_rank_diffs.sum() / ((n**2 - 1) / 3)

    return xi_corr
