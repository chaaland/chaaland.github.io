import numpy as np
from scipy.optimize import lsq_linear


def gauss_newton(f, x0, J, atol: float = 1e-4, max_iter: int = 100):
    """Implements the Gauss-Newton method for NLLS

    :param f: function to compute the residual vector
    :param x0: array corresponding to initial guess
    :param J: function to compute the jacobian of f
    :param atol: stopping criterion for the root mean square 
    of the squared norm of the gradient of f
    :param max_iter: maximum number of iterations to run before 
    terminating
    """
    iterates = [x0,]
    rms = lambda x: np.sqrt(np.mean(np.square(x)))
    costs = [rms(f(x0)),]
    cnt = 0
    grad_rms = np.inf

    while cnt < max_iter and grad_rms > atol:
        x_k = iterates[-1]
        A = J(x_k)
        b = A @ x_k - f(x_k)
        result = lsq_linear(A, b)
        iterates.append(result.x)
        costs.append(rms(f(result.x)))
        grad_rms = rms(A.T * f(x_k))
        cnt += 1
    
    return iterates, np.asarray(costs)
