import numpy as np
from scipy.optimize import lsq_linear


def gauss_newton(f, x0, J, atol: float = 1e-4, max_iter: int = 100):
    iterates = [x0,]
    mse = lambda x: np.sqrt(np.mean(np.square(x)))
    costs = [mse(f(x0)),]
    cnt = 0
    grad_mse = np.inf

    while cnt < max_iter and grad_mse > atol:
        x_k = iterates[-1]
        A = J(x_k)
        b = A @ x_k - f(x_k)
        result = lsq_linear(A, b)
        iterates.append(result.x)
        costs.append(mse(f(result.x)))
        grad_mse = mse(A.T * f(x_k))
        cnt += 1
    
    return iterates, np.asarray(costs)
