import numpy as np
from scipy.optimize import lsq_linear


def levenberg_marquardt(f, x0, J, max_iter: int = 100):
    mu = 1
    iterates = [x0,]
    costs = [np.mean(np.square(f(x0))),]
    cnt = 0

    while cnt < max_iter:
        x_k = iterates[-1]
        A = np.stack([J(x_k), np.sqrt(mu) * np.eye(x_k.shape)], axis=0)
        b = np.stack([f(x_k) - A.T @ x_k, np.sqrt(mu) * x_k], axis=0)
        result = lsq_linear(A, b)
        iterates.append(result.x)
        costs.append(np.mean(np.square(result.x))
        cnt += 1
        if decrease:
            mu /= 1.5
        else:
            mu *= 2
    
    return iterates, costs




