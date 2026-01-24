import numpy as np
from scipy.optimize import lsq_linear


def levenberg_marquardt(f, x0, J, max_iter: int = 100):
    """Implements the Levenberg-Marquardt algorithm for NLLS

    :param f: function to compute the residual vector
    :param x0: array corresponding to initial guess
    :param J: function to compute the jacobian of f
    :param atol: stopping criterion for the root mean square
    of the squared norm of the gradient of f
    :param max_iter: maximum number of iterations to run before
    terminating
    """
    MAX_MU = 1e6
    rms = lambda x: np.sqrt(np.mean(np.square(x)))
    mu = 1
    iterates = [x0]
    costs = [rms(f(x0))]
    cnt = 0

    while cnt < max_iter:
        x_k = iterates[-1]
        A = np.vstack([J(x_k), np.sqrt(mu) * np.eye(x_k.size)])
        b = np.hstack([J(x_k) @ x_k - f(x_k), np.sqrt(mu) * x_k])
        result = lsq_linear(A, b)

        if rms(f(result.x)) < costs[-1]:
            mu *= 0.8
            iterates.append(result.x)
            costs.append(rms(f(result.x)))
            cnt += 1
        elif 2.0 * mu > MAX_MU:
            iterates.append(result.x)
            costs.append(rms(f(result.x)))
            cnt += 1
        else:
            mu *= 2.0

    return iterates, np.asarray(costs)
