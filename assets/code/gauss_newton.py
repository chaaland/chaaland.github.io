import numpy as np
from scipy.optimize import lsq_linear

def gauss_newton(f, x0, J, max_iter: int = 100):
    iterates = [x0,]
    costs = [np.mean(np.square(f(x0))),]
    cnt = 0

    while cnt < max_iter:
        x_k = iterates[-1]
        A = J(x_k)
        b = f(x_k) - A.T @ x_k
        result = lsq_linear(A, b)
        iterates.append(result.x)
        costs.append(np.mean(np.square(result.x))
        cnt += 1
    
    return iterates, costs




