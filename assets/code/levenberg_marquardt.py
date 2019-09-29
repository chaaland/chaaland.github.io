import numpy as np
from scipy.optimize import lsq_linear


def levenberg_marquardt(f, x0, J, max_iter: int = 100):
    mse = lambda x: np.sqrt(np.mean(np.square(x)))
    mu = 1
    iterates = [x0,]
    costs = [mse(f(x0)),]
    cnt = 0

    while cnt < max_iter:
        x_k = iterates[-1]
        A = np.vstack([J(x_k), np.sqrt(mu) * np.eye(x_k.size)])
        b = np.hstack([J(x_k) @ x_k - f(x_k), np.sqrt(mu) * x_k])
        result = lsq_linear(A, b)

        if mse(f(result.x)) < costs[-1]:
            mu *= 0.8
            iterates.append(result.x)
            costs.append(mse(f(result.x)))
            cnt += 1
        else:
            mu *= 2.0
        print(mu)
    
    return iterates, np.asarray(costs)




