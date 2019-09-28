def plot_scatter_sampling_grid():
    # x = np.linspace(-3,3, 15)
    # y = np.linspace(-5,5,15)
    # X, Y = np.meshgrid(x,y)
    
    # plt.scatter(X.ravel(), Y.ravel())

    # plt.figure()
    # r = np.linspace(0,3, 10)
    # theta = np.linspace(0,2*np.pi, 20)
    # R, Theta = np.meshgrid(r, theta)
    # X, Y = R * np.cos(Theta), R * np.sin(Theta)
    
    # plt.scatter(X.ravel(), Y.ravel())

    # plt.figure()
    # r = np.linspace(0,3, 10)
    # theta = np.linspace(0,2*np.pi, 30)
    # R, Theta = np.meshgrid(r, theta)
    # a, b = 1/3, 1/9
    # X, Y = a * R * np.cos(Theta), b * R * np.sin(Theta)
    # plt.xlim([-1,1])
    # plt.ylim([-1,1])
    
    # plt.scatter(X.ravel(), Y.ravel())

    plt.figure()
    r = np.linspace(0,3, 10)
    theta = np.linspace(0,2*np.pi, 30)
    angle = -np.pi/3
    rot_mat = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    R, Theta = np.meshgrid(r, theta)
    a, b = 1/3, 1/9
    center = np.asarray([0.5, -0.5]).reshape((-1,1))
    scaled_x = (a * R * np.cos(Theta)).reshape((-1, 1))
    scaled_y = (b * R * np.sin(Theta)).reshape((-1, 1))
    plot_grid = rot_mat @ np.hstack([scaled_x, scaled_y]).T + center
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    
    plt.scatter(plot_grid[0,:], plot_grid[1,:])
    plt.show()