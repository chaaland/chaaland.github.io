import math

LOGARITHM_LOOKUP = [math.log(1 + 2.0**-k) for k in range(100)]


def log(x: float, n_iters: int = 30):
    assert n_iters < 30

    log_acc = 0
    arg_acc = 1
    factor = 1
    for k in range(n_iters):
        candidate = arg_acc + arg_acc * factor
        if candidate <= x:
            arg_acc = candidate
            log_acc += LOGARITHM_LOOKUP[k]
        factor /= 2
    return log_acc


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
    log_acc = 0
    arg_acc = 1
    factor = 1
    for k in range(n_iters):
        candidate = log_acc + LOGARITHM_LOOKUP[k]
        if candidate < x:
            log_acc = candidate
            arg_acc = arg_acc + arg_acc * factor
        factor /= 2
    return arg_acc
