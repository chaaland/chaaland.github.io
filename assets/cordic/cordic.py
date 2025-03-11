import numpy as np

ARCTAN_LOOKUP = [np.atan(1 / 2**i) for i in range(100)]


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


def get_scale_factor(n_iters: int) -> float:
    return np.exp(sum(-0.5 * np.log(1 + 2 ** (-2 * i))) for i in range(n_iters))


def cordic(theta: float) -> tuple[float, float]:
    # assume first quadrant
    v = np.array([1, 0])
    theta_hat = 0
    for i in range(20):
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
