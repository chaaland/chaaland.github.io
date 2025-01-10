import numpy as np
from matplotlib import pyplot as plt


def plot_rectangular_grid(mode="h"):
    n_x = 10
    n_y = 30

    x = np.linspace(0, 3, n_x)
    y = np.linspace(0, 2 * np.pi, n_y)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(10, 8))
    plt.scatter(X.ravel(), Y.ravel())
    plt.xlabel(r"$r$", fontsize=14)
    plt.ylabel(r"$\theta$", fontsize=14)
    vert_mid = n_x // 2
    horiz_mid = n_y // 2
    if mode.lower() == "h":
        plt.scatter(X[horiz_mid - 2 : horiz_mid + 2, :].ravel(), Y[horiz_mid - 2 : horiz_mid + 2, :].ravel(), color="r")
        plt.tight_layout()
        plt.savefig("../../images/3d-plotting-using-non-rectangular-grids/rectangular-horizontal.png")
    else:
        plt.scatter(X[:, vert_mid - 2 : vert_mid + 2].ravel(), Y[:, vert_mid - 2 : vert_mid + 2].ravel(), color="g")
        plt.tight_layout()
        plt.savefig("../../images/3d-plotting-using-non-rectangular-grids/rectangular-vertical.png")


def plot_ellipse_grid(xlim=None, ylim=None, a=1, b=1, angle=0, center=np.zeros((2, 1)), mode="h"):
    n_r = 10
    n_theta = 30
    r = np.linspace(0, 3, n_r)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    rot_mat = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    R, Theta = np.meshgrid(r, theta)
    scaled_x = (a * R * np.cos(Theta)).reshape((-1, 1))
    scaled_y = (b * R * np.sin(Theta)).reshape((-1, 1))
    plot_grid = rot_mat @ np.hstack([scaled_x, scaled_y]).T + center

    plt.figure(figsize=(10, 8))
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.scatter(plot_grid[0, :], plot_grid[1, :])
    vert_mid = n_r // 2
    horiz_mid = n_theta // 2
    if mode.lower() == "h":
        scaled_x = (a * R * np.cos(Theta))[horiz_mid - 2 : horiz_mid + 2, :].reshape((-1, 1))
        scaled_y = (b * R * np.sin(Theta))[horiz_mid - 2 : horiz_mid + 2, :].reshape((-1, 1))
        c = "r"
    else:
        scaled_x = (a * R * np.cos(Theta))[:, vert_mid - 2 : vert_mid + 2].reshape((-1, 1))
        scaled_y = (b * R * np.sin(Theta))[:, vert_mid - 2 : vert_mid + 2].reshape((-1, 1))
        c = "g"

    plot_grid = rot_mat @ np.hstack([scaled_x, scaled_y]).T + center
    plt.scatter(plot_grid[0, :], plot_grid[1, :], color=c)
    plt.tight_layout()


def plot_sinc_surface():
    x = np.linspace(-6, 6, 400) + 1e-6
    y = np.linspace(-6, 6, 400)

    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.pi * np.sqrt(X**2 + Y**2)) / (np.pi * np.sqrt(X**2 + Y**2))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="hot")
    ax.set_title(r"$\sin\left(\pi * \sqrt{x^2+y^2}\right)/\pi\sqrt{x^2+y^2}$")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)

    r = np.hstack([np.arange(0, 1, 0.1 / 2) + 1e-6, np.linspace(1, 6, 30)])
    theta = np.linspace(0, 2 * np.pi, 100)
    R, Theta = np.meshgrid(r, theta)

    X, Y = R * np.cos(Theta), R * np.sin(Theta)
    Z = np.sin(np.pi * np.sqrt(X**2 + Y**2)) / (np.pi * np.sqrt(X**2 + Y**2))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="hot")
    ax.set_title(r"$\sin\left(\pi * \sqrt{x^2+y^2}\right)/\pi\sqrt{x^2+y^2}$")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)


def plot_gaussian_surface():
    x = np.linspace(-3, 3, 400) + 1e-6
    y = np.linspace(-3, 3, 400)

    correlation = 0.0
    var_x = 1
    var_y = 2
    cov_xy = correlation * np.sqrt(var_x * var_y)
    var_mat = np.asarray([[var_x, cov_xy], [cov_xy, var_y]])

    X, Y = np.meshgrid(x, y)
    xy_stacked = np.vstack([X.reshape((1, -1)), Y.reshape((1, -1))])
    mahalonobis_dist = np.sum(xy_stacked * np.linalg.solve(var_mat, xy_stacked), axis=0).reshape(X.shape)

    Z = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(var_mat)) * np.exp(-0.5 * mahalonobis_dist)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="hot")
    ax.set_title("Gaussian Distribution")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)

    a = np.sqrt(var_x)
    b = np.sqrt(var_y)
    r = np.linspace(0, 5, 50) + 1e-6
    theta = np.linspace(0, 2 * np.pi, 100)
    R, Theta = np.meshgrid(r, theta)
    X, Y = a * R * np.cos(Theta), b * R * np.sin(Theta)

    # plt.figure()
    # plt.scatter(X.ravel(), Y.ravel())
    xy_stacked = np.vstack([X.reshape((1, -1)), Y.reshape((1, -1))])
    mahalonobis_dist = np.sum(xy_stacked * np.linalg.solve(var_mat, xy_stacked), axis=0).reshape(X.shape)
    Z = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(var_mat)) * np.exp(-0.5 * mahalonobis_dist)

    fig = plt.figure(figsize=(10, 10))
    plt.contour(X, Y, Z)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="hot")
    ax.set_xlim([-7, 7])
    ax.set_ylim([-7, 7])
    ax.set_title(r"Gaussian Distribution")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)


def plot_elongated_paraboloid(with_grid=False):
    x = np.linspace(-1, 1, 30)
    y = np.linspace(-1, 1, 30)

    correlation = 0.5
    var_x = 1
    var_y = 4
    X, Y = np.meshgrid(x, y)
    cov_xy = correlation * np.sqrt(var_x * var_y)
    Z = var_x * X**2 + var_y * Y**2 + 2 * cov_xy * X * Y

    sigma = np.asarray([[var_x, cov_xy], [cov_xy, var_y]])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    if with_grid:
        ax.scatter(X.ravel(), Y.ravel(), np.zeros(X.size), c="tab:red")
    ax.plot_surface(X, Y, Z, cmap="hot")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
    ax.set_zlabel(r"$f(x) = x^TAx$", fontsize=14)
    plt.tight_layout()

    r = np.linspace(0, 2, 20)
    theta = np.linspace(0, 2 * np.pi, 50)
    R_mesh, Theta_mesh = np.meshgrid(r, theta)

    eig_vals, eig_vecs = np.linalg.eigh(sigma)
    major_axis = eig_vecs[:, 0]
    angle = np.arctan2(major_axis[1], major_axis[0])
    rot_mat = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    a, b = 1 / np.sqrt(eig_vals[0]), 1 / np.sqrt(eig_vals[1])

    scaled_x = (a * R_mesh * np.cos(Theta_mesh)).ravel()
    scaled_y = (b * R_mesh * np.sin(Theta_mesh)).ravel()
    plot_grid = rot_mat @ np.stack([scaled_x, scaled_y], axis=1).T
    Z = np.sum(plot_grid * (sigma @ plot_grid), axis=0).reshape(R_mesh.shape)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    if with_grid:
        ax.scatter(plot_grid[0, :], plot_grid[1, :], np.zeros(R_mesh.size), c="tab:red")
    ax.plot_surface(plot_grid[0, :].reshape(R_mesh.shape), plot_grid[1, :].reshape(R_mesh.shape), Z, cmap="hot")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
    ax.set_zlabel(r"$f(x) = x^TAx$", fontsize=14)
    plt.tight_layout()


if __name__ == "__main__":
    center = np.asarray([0.25, -0.25]).reshape((-1, 1))
    plot_elongated_paraboloid(with_grid=False)
    plot_elongated_paraboloid(with_grid=True)
    plot_sinc_surface()
    plot_gaussian_surface()
    plot_rectangular_grid(mode="h")
    plot_rectangular_grid(mode="v")
    plot_ellipse_grid([-3, 3], [-3, 3], mode="h")
    plot_ellipse_grid([-3, 3], [-3, 3], mode="v")
    plot_ellipse_grid([-1, 1], [-1, 1], 1 / 3, 1 / 9, mode="h")
    plot_ellipse_grid([-1, 1], [-1, 1], 1 / 3, 1 / 9, mode="v")
    plot_ellipse_grid([-1, 1], [-1, 1], 1 / 3, 1 / 9, -np.pi / 3, mode="h")
    plot_ellipse_grid([-1, 1], [-1, 1], 1 / 3, 1 / 9, -np.pi / 3, mode="v")
    plot_ellipse_grid([-1, 1], [-1, 1], 1 / 3, 1 / 9, -np.pi / 3, center, mode="h")
    plot_ellipse_grid([-1, 1], [-1, 1], 1 / 3, 1 / 9, -np.pi / 3, center, mode="v")

    plt.show()
