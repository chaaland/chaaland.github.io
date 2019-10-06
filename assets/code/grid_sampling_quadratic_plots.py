import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_rectangular_grid(mode="h"):
    n_x = 10
    n_y = 30

    x = np.linspace(0, 3, n_x)
    y = np.linspace(0, 2*np.pi, n_y)
    X, Y = np.meshgrid(x,y)
    
    plt.figure(figsize=(10,8))
    plt.scatter(X.ravel(), Y.ravel())
    vert_mid = n_x // 2
    horiz_mid = n_y // 2
    if mode.lower() == "h":
        plt.scatter(X[horiz_mid-2:horiz_mid+2,:].ravel(), Y[horiz_mid-2:horiz_mid+2,:].ravel(), color="r")
        plt.savefig("../images/grid-sampling-rectangular-horizontal.png")
    else:
        plt.scatter(X[:,vert_mid-2:vert_mid+2].ravel(), Y[:,vert_mid-2:vert_mid+2].ravel(), color="g")
        plt.savefig("../images/grid-sampling-rectangular-vertical.png")

def plot_ellipse_grid(xlim=None, ylim=None, a=1, b=1, angle=0, center=np.zeros((2,1)), mode="h"):
    n_r = 10
    n_theta = 30
    r = np.linspace(0, 3, n_r)
    theta = np.linspace(0, 2*np.pi, n_theta)
    rot_mat = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    R, Theta = np.meshgrid(r, theta)
    scaled_x = (a * R * np.cos(Theta)).reshape((-1, 1))
    scaled_y = (b * R * np.sin(Theta)).reshape((-1, 1))
    plot_grid = rot_mat @ np.hstack([scaled_x, scaled_y]).T + center

    plt.figure(figsize=(10, 8))
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.scatter(plot_grid[0,:], plot_grid[1,:])
    vert_mid = n_r // 2
    horiz_mid = n_theta // 2
    if mode.lower() == "h":
        scaled_x = (a * R * np.cos(Theta))[horiz_mid-2:horiz_mid+2, :].reshape((-1, 1))
        scaled_y = (b * R * np.sin(Theta))[horiz_mid-2:horiz_mid+2, :].reshape((-1, 1))
        c = "r"
    else:
        scaled_x = (a * R * np.cos(Theta))[:, vert_mid-2:vert_mid+2].reshape((-1, 1))
        scaled_y = (b * R * np.sin(Theta))[:, vert_mid-2:vert_mid+2].reshape((-1, 1))
        c = "g"

    plot_grid = rot_mat @ np.hstack([scaled_x, scaled_y]).T + center
    plt.scatter(plot_grid[0,:], plot_grid[1,:], color=c)

def plot_sinc_surface():
    x = np.linspace(-6, 6, 400) + 1e-6
    y = np.linspace(-6, 6, 400)
    
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.pi * np.sqrt(X ** 2 + Y ** 2)) / (np.pi * np.sqrt(X**2 + Y**2))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="hot")
    ax.set_title(r"$\sin(\pi * \sqrt{x^2+y^2})/\sqrt{x^2+y^2}$")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)    

    r = np.linspace(0, 6, 50) + 1e-6
    theta = np.linspace(0, 2 * np.pi, 100)
    R, Theta = np.meshgrid(r, theta)

    X, Y = R * np.cos(Theta), R * np.sin(Theta)
    Z = np.sin(np.pi * np.sqrt(X ** 2 + Y ** 2)) / (np.pi * np.sqrt(X**2 + Y**2))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="hot")
    ax.set_title(r"$\sin(\pi * \sqrt{x^2+y^2})/\sqrt{x^2+y^2}$")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)    

def plot_gaussian_surface():
    x = np.linspace(-3, 3, 400) + 1e-6
    y = np.linspace(-3, 3, 400)
    
    correlation = 0.0
    var_x = 1
    var_y = 9
    cov_xy = correlation * np.sqrt(var_x * var_y)
    var_mat = np.asarray([[var_x, cov_xy], [cov_xy, var_y]])

    X, Y = np.meshgrid(x, y)
    xy_stacked = np.vstack([X.reshape((1,-1)), Y.reshape((1,-1))])
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
    xy_stacked = np.vstack([X.reshape((1,-1)), Y.reshape((1,-1))])
    mahalonobis_dist = np.sum(xy_stacked * np.linalg.solve(var_mat, xy_stacked), axis=0).reshape(X.shape)
    Z = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(var_mat)) * np.exp(-0.5 * mahalonobis_dist)

    fig = plt.figure(figsize=(10, 10))
    plt.contour(X,Y,Z)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="hot")
    ax.set_xlim([-16,16])
    ax.set_ylim([-16,16])
    ax.set_title(r"Gaussian Distribution")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)    

def plot_elongated_paraboloid():
    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)

    correlation = 0.5
    var_x = 1
    var_y = 4
    X, Y = np.meshgrid(x, y)
    cov_xy = correlation * np.sqrt(var_x * var_y)
    Z = var_x*X**2 + var_y*Y**2 + 2 * cov_xy * X * Y

    sigma = np.asarray([[var_x, cov_xy],[cov_xy, var_y]])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="hot")
    ax.set_title(r"Anisotropic Paraboloid")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)    

    r = np.linspace(0, 2, 100)
    theta = np.linspace(0, 2*np.pi,100)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="hot")
    ax.set_title(r"Anisotropic Paraboloid")
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)    

if __name__ == "__main__":
    center = np.asarray([0.5, -0.5]).reshape((-1,1))
    # plot_elongated_paraboloid()
    # plot_sinc_surface()
    # plot_gaussian_surface()
    # plot_rectangular_grid(mode="h")
    # plot_rectangular_grid(mode="v")
    plot_ellipse_grid([-3,3],[-3,3])
    plot_ellipse_grid([-1,1],[-1,1], 1/3, 1/9)
    plot_ellipse_grid([-1,1],[-1,1], 1/3, 1/9, -np.pi/3)
    plot_ellipse_grid([-1,1],[-1,1], 1/3, 1/9, -np.pi/3, center, mode="h")
    plot_ellipse_grid([-1,1],[-1,1], 1/3, 1/9, -np.pi/3, center, mode="v")

    plt.show()