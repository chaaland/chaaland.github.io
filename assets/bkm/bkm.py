import math

LOGARITHM_LOOKUP = [math.log(1 + 2.0**-k) for k in range(100)]


def log(x: float, n_iters: int):
    assert n_iters < 100

    log_x = 0
    x_hat = 1
    for k in range(n_iters):
        a_k = 1 + 2**-k
        tmp = x_hat * a_k
        if tmp <= x:
            log_x += LOGARITHM_LOOKUP[k]
            x_hat = tmp

    return log_x


def log_alt(x: float, n_iters: int) -> float:
    log_x = 0
    x_hat = x

    for k in range(n_iters):
        if x_hat > 1:
            x_hat /= 1 + 2**-k
            log_x += LOGARITHM_LOOKUP[k]
        else:
            x_hat *= 1 + 2**-k
            log_x -= LOGARITHM_LOOKUP[k]

    return log_x

def exp(x: float, n_iters: int) -> float:
    pass