import numpy as np

ARCTAN_LOOKUP = [np.atan(1 / 2**i) for i in range(100)]
ARCTANH_LOOKUP = [np.atanh(1 / 2**i) for i in range(1, 100)]


def cordic_iter(i: int, v: np.ndarray, ccw: bool, scale: bool = False):
    v_x, v_y = v
    two_factor = 2**-i  # need to make this into bit shifting...

    if ccw:
        sigma = 1
        x_coord = v_x - sigma * two_factor * v_y
        y_coord = sigma * v_x * two_factor + v_y
    else:
        sigma = -1
        x_coord = v_x - sigma * two_factor * v_y
        y_coord = sigma * v_x * two_factor + v_y

    v_next = np.array([x_coord, y_coord])
    if scale:
        k = 1 / (1 + 2 ** (-2 * i)) ** 0.5
        return v_next * k

    return v_next


def hyperbolic_cordic_iter(i: int, v: np.ndarray, ccw: bool, scale: bool = False):
    v_x, v_y = v
    two_factor = 2 ** -(i + 1)  # need to make this into bit shifting...

    if ccw:
        sigma = 1
        x_coord = v_x + sigma * two_factor * v_y
        y_coord = sigma * v_x * two_factor + v_y
    else:
        sigma = -1
        x_coord = v_x + sigma * two_factor * v_y
        y_coord = sigma * v_x * two_factor + v_y

    v_next = np.array([x_coord, y_coord])
    if scale:
        k = 1 / (1 - 2 ** (-2 * (i + 1))) ** 0.5
        return v_next * k

    return v_next


def get_scale_factor(n_iters: int) -> float:
    return np.exp(sum(-0.5 * np.log(1 + 2 ** (-2 * i))) for i in range(n_iters))


def get_hyperbolic_scale_factor(n_iters: int) -> float:
    # TODO caseyh: start at 1?
    return np.exp(sum(-0.5 * np.log(1 - 2 ** (-2 * (i + 1)))) for i in range(n_iters))


def cordic(theta: float) -> tuple[float, float]:
    assert 0 <= theta <= np.pi / 2

    v = np.array([1, 0])
    theta_hat = 0
    n_iters = 20
    for i in range(n_iters):
        cos_theta, sin_theta = v
        if theta_hat == theta:
            return cos_theta, sin_theta

        ccw = theta_hat < theta
        delta_theta = ARCTAN_LOOKUP[i]
        if ccw:
            theta_hat += delta_theta
        else:
            theta_hat -= delta_theta

        v = cordic_iter(i, v, ccw, scale=False)
    v *= get_scale_factor(n_iters)
    return v[0], v[1]


def hyperbolic_cordic(theta: float) -> tuple[float, float]:
    v = np.array([1, 0])
    theta_hat = 0
    n_iters = 20

    # TODO caseyh: should this start at 1?
    for i in range(n_iters):
        cosh_theta, sinh_theta = v
        if theta_hat == theta:
            return cosh_theta, sinh_theta

        ccw = theta_hat < theta
        delta_theta = ARCTANH_LOOKUP[i]
        if ccw:
            theta_hat += delta_theta
        else:
            theta_hat -= delta_theta

        v = hyperbolic_cordic_iter(i + 1, v, ccw, scale=False)

    v *= get_hyperbolic_scale_factor(n_iters)
    return v[0], v[1]
