import numpy as np
import matplotlib.pyplot as plt


def plot_rectangular_grid():
    x = np.linspace(0, 3, 10)
    y = np.linspace(0, 2*np.pi, 30)
    X, Y = np.meshgrid(x,y)
    
    plt.figure(figsize=(10,8))
    plt.scatter(X.ravel(), Y.ravel())
    plt.scatter(X[:,5:7].ravel(), Y[:,5:7].ravel(), color="r")

def plot_radially_symmetric_grid():
    r = np.linspace(0, 3, 10)
    theta = np.linspace(0, 2*np.pi, 30)
    R, Theta = np.meshgrid(r, theta)
    X, Y = R * np.cos(Theta), R * np.sin(Theta)
    
    plt.figure(figsize=(10,8))
    plt.scatter(X.ravel(), Y.ravel())
    plt.scatter(X[:,5:7].ravel(), Y[:,5:7].ravel(), color="r")

def plot_anisotropic_grid():
    r = np.linspace(0, 3, 10)
    theta = np.linspace(0, 2*np.pi, 30)
    R, Theta = np.meshgrid(r, theta)
    a, b = 1/3, 1/9
    X, Y = a * R * np.cos(Theta), b * R * np.sin(Theta)

    plt.figure(figsize=(10,8))
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    
    plt.scatter(X.ravel(), Y.ravel())
    plt.scatter(X[:,5:7].ravel(), Y[:,5:7].ravel(), color="r")

def plot_rotated_anisotropic_grid():
    r = np.linspace(0, 3, 10)
    theta = np.linspace(0, 2*np.pi, 30)
    angle = -np.pi/3
    rot_mat = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    R, Theta = np.meshgrid(r, theta)
    a, b = 1/3, 1/9
    scaled_x = (a * R * np.cos(Theta)).reshape((-1, 1))
    scaled_y = (b * R * np.sin(Theta)).reshape((-1, 1))
    plot_grid = rot_mat @ np.hstack([scaled_x, scaled_y]).T 

    plt.figure(figsize=(10, 8))
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    
    plt.scatter(plot_grid[0,:], plot_grid[1,:])

def plot_shifted_anisotropic_grid():
    r = np.linspace(0, 3, 10)
    theta = np.linspace(0, 2*np.pi, 30)
    angle = -np.pi/3
    rot_mat = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    R, Theta = np.meshgrid(r, theta)
    a, b = 1/3, 1/9
    center = np.asarray([0.5, -0.5]).reshape((-1,1))
    scaled_x = (a * R * np.cos(Theta)).reshape((-1, 1))
    scaled_y = (b * R * np.sin(Theta)).reshape((-1, 1))
    plot_grid = rot_mat @ np.hstack([scaled_x, scaled_y]).T + center

    plt.figure(figsize=(10, 8))
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    
    plt.scatter(plot_grid[0,:], plot_grid[1,:])


if __name__ == "__main__":
    # plot_rectangular_grid()
    # plot_radially_symmetric_grid()
    plot_anisotropic_grid()
    # plot_shifted_anisotropic_grid()
    plt.show()