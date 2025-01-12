import numpy as np


def tanh_sinh_points(n_points, h=0.1):
    """
    Generate the points and weights for tanh-sinh quadrature.

    Parameters:
        n_points: Number of points in each direction from zero
        h: Step size parameter that controls the distribution of points

    Returns:
        points: The quadrature points in [-1, 1]
        weights: The corresponding weights
    """
    # Generate evaluation points
    k = np.arange(-n_points, n_points + 1)
    t = h * k

    # Apply the tanh(sinh()) transformation
    sinh_t = np.sinh(t)
    cosh_t = np.cosh(t)
    tanh_term = np.tanh(np.pi / 2 * sinh_t)

    # Compute the quadrature points (x values)
    x_k = tanh_term

    # Compute the weights using the derivative of the transformation
    cosh_term = np.cosh(np.pi / 2 * sinh_t)
    print(cosh_term)
    w_k = h * np.pi / 2 * (cosh_t / cosh_term) / cosh_term
    print(w_k)

    pos_ones_idxs = x_k == 1
    neg_ones_idxs = x_k == -1

    x_k = np.clip(x_k, np.min(x_k[~neg_ones_idxs]), np.max(x_k[~pos_ones_idxs]))
    # print(x_k)
    # print(np.min(x_k[~neg_ones_idxs]))
    # print(np.max(x_k[~pos_ones_idxs]))
    return x_k, w_k


def tanh_sinh_quadrature(f, n_points=100, h=0.1):
    """
    Compute the integral of f over [-1, 1] using tanh-sinh quadrature.

    Parameters:
        f: Function to integrate
        n_points: Number of points in each direction from zero
        h: Step size parameter

    Returns:
        Approximation of the integral
    """
    points, weights = tanh_sinh_points(n_points, h)
    return np.sum(weights * f(points))
