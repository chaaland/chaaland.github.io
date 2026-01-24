import numpy as np


def tanh_sinh_points(n, h=0.1):
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
    k = np.arange(-n, n)
    t = h * k

    # Apply the tanh(sinh()) transformation
    sinh_t = np.sinh(t)
    cosh_t = np.cosh(t)

    # Compute the quadrature points (x values)
    x_k = np.tanh(np.pi / 2 * sinh_t)

    # Compute the weights using the derivative of the transformation
    cosh_term = np.cosh(np.pi / 2 * sinh_t)
    w_k = h * np.pi / 2 * cosh_t / (cosh_term * cosh_term)

    return x_k, w_k


def tanh_sinh_quadrature(f, n=100, h=0.1):
    """
    Compute the integral of f over [-1, 1] using tanh-sinh quadrature.

    Parameters:
        f: Function to integrate
        n: Number of points in each direction from zero
        h: Step size parameter

    Returns:
        Approximation of the integral
    """
    points, weights = tanh_sinh_points(n, h)
    return np.sum(weights * f(points))


def left_riemann_points(a: int, b: int, n_points: int):
    h = (b - a) / n_points

    x_k = np.linspace(a, b, n_points + 1)
    w_k = h * np.array([1 if i != n_points else 0 for i, _ in enumerate(x_k)])

    return x_k, w_k


def right_riemann_points(a, b, n_points):
    h = (b - a) / n_points
    x_k = np.linspace(a, b, n_points + 1)
    w_k = h * np.array([1 if i != 0 else 0 for i, _ in enumerate(x_k)])

    return x_k, w_k


def riemann_quadrature(f, a=-1, b=1, n=10, side="left", ignore_infs: bool = True):
    match side:
        case "left":
            points, weights = left_riemann_points(a, b, n)
        case "right":
            points, weights = right_riemann_points(a, b, n)
        case _:
            raise ValueError(f"Unsupported value `{side=}`")

    f_vals = f(points)

    if ignore_infs:
        is_finite_mask = np.isfinite(f_vals)
        weights = weights[is_finite_mask]
        f_vals = f_vals[is_finite_mask]

    return np.dot(weights, f_vals)


def trapezoidal_points(a, b, n):
    h = (b - a) / n
    x_k = np.linspace(a, b, n + 1)
    w_k = h / 2 * np.array([2.0 if 1 <= i < n else 1.0 for i, _ in enumerate(x_k)])

    return x_k, w_k


def trapezoidal_quadrature(f, a=-1, b=1, n=10):
    points, weights = trapezoidal_points(a, b, n)
    return np.dot(weights, f(points))
