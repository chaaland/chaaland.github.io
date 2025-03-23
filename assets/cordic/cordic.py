import math

import numpy as np

ARCTAN_LOOKUP = [math.atan(1 / 2**k) for k in range(100)]
ARCTANH_LOOKUP = [math.atanh(1 / 2**k) for k in range(1, 100)]


def cordic_iter(i: int, v: list[float], ccw: bool, scale: bool = False) -> list[float]:
    v_x, v_y = v
    two_factor = 2**-i
    sigma = 1 if ccw else -1

    x_coord = v_x - sigma * two_factor * v_y
    y_coord = sigma * v_x * two_factor + v_y

    if scale:
        k = 1 / (1 + 2 ** (-2 * i)) ** 0.5
        return [k * x_coord, k * y_coord]
    else:
        return [x_coord, y_coord]


def hyperbolic_cordic_iter(i: int, v: list[float], ccw: bool, scale: bool = False) -> list[float]:
    v_x, v_y = v
    two_factor = 2 ** -(i + 1)
    sigma = 1 if ccw else -1

    x_coord = v_x + sigma * two_factor * v_y
    y_coord = sigma * v_x * two_factor + v_y
    if scale:
        k = 1 / (1 - 2 ** (-2 * (i + 1))) ** 0.5
        return [k * x_coord, k * y_coord]
    else:
        return [x_coord, y_coord]


def get_scale_factor(n_iters: int) -> float:
    return math.exp(sum(-0.5 * math.log(1 + 2 ** (-2 * k))) for k in range(n_iters))


def get_hyperbolic_scale_factor(n_iters: int) -> float:
    return math.exp(sum(-0.5 * math.log(1 - 2 ** (-2 * (k + 1)))) for k in range(n_iters))


def cordic(theta: float, n_iters: int = 20) -> tuple[float, float]:
    assert 0 <= theta <= np.pi / 2

    v = [1, 0]
    theta_hat = 0

    for k in range(n_iters):
        if theta_hat == theta:
            gain = get_scale_factor(k)
            cos_theta = gain * v[0]
            sin_theta = gain * v[1]

            return cos_theta, sin_theta

        ccw = theta_hat < theta
        delta_theta = ARCTAN_LOOKUP[k]
        if ccw:
            theta_hat += delta_theta
        else:
            theta_hat -= delta_theta

        v = cordic_iter(k, v, ccw, scale=False)

    gain = get_scale_factor(n_iters)
    cos_theta = gain * v[0]
    sin_theta = gain * v[1]

    return cos_theta, sin_theta


def hyperbolic_cordic(theta: float, n_iters: int = 20) -> tuple[float, float]:
    v = [1, 0]
    theta_hat = 0

    for k in range(n_iters):
        if theta_hat == theta:
            gain = get_hyperbolic_scale_factor(k + 1)
            cosh_theta = gain * v[0]
            sinh_theta = gain * v[1]

            return cosh_theta, sinh_theta

        ccw = theta_hat < theta
        delta_theta = ARCTANH_LOOKUP[k]
        if ccw:
            theta_hat += delta_theta
        else:
            theta_hat -= delta_theta

        v = hyperbolic_cordic_iter(k, v, ccw, scale=False)

    gain = get_hyperbolic_scale_factor(n_iters)
    cosh_theta = gain * v[0]
    sinh_theta = gain * v[1]

    return cosh_theta, sinh_theta
