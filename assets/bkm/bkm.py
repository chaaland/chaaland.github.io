import math

LOGARITHM_LOOKUP = [math.log(1 + 2.0**-k) for k in range(100)]


def log(x: float, n_iters: int = 30):
    assert n_iters < 30

    log_x = 0
    x_hat = 1
    factor = 1
    for k in range(n_iters):
        tmp = x_hat + x_hat * factor  # x * (1 + 2**-k)
        if tmp <= x:
            log_x += LOGARITHM_LOOKUP[k]
            x_hat = tmp
        factor /= 2
    return log_x


def log_alt(x: float, n_iters: int = 30) -> float:
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


def exp(x: float, n_iters: int = 30) -> float:
    # y = e^x
    log_y = 0
    y_hat = 1

    for k in range(n_iters):
        tmp = log_y + LOGARITHM_LOOKUP[k]
        if tmp < x:
            log_y = tmp
            y_hat = y_hat + y_hat / 2**k  # x * (1 + 2**-k)

    return y_hat
