import numpy as np
import matplotlib.pyplot as plt


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
    else:
        plt.scatter(X[:,vert_mid-2:vert_mid+2].ravel(), Y[:,vert_mid-2:vert_mid+2].ravel(), color="g")

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


if __name__ == "__main__":
    center = np.asarray([0.5, -0.5]).reshape((-1,1))
    plot_rectangular_grid(mode="h")
    plot_rectangular_grid(mode="v")
    # plot_ellipse_grid([-3,3],[-3,3])
    # plot_ellipse_grid([-1,1],[-1,1], 1/3, 1/9)
    # plot_ellipse_grid([-1,1],[-1,1], 1/3, 1/9, -np.pi/3)
    plot_ellipse_grid([-1,1],[-1,1], 1/3, 1/9, -np.pi/3, center, mode="h")
    plot_ellipse_grid([-1,1],[-1,1], 1/3, 1/9, -np.pi/3, center, mode="v")

    plt.show()