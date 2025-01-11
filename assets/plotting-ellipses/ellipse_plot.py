from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True, parents=True)

GIF_DIR = Path("gifs")
GIF_DIR.mkdir(exist_ok=True, parents=True)


def plot_ellipse_parametric(P: np.ndarray, c: np.ndarray):
    x_points, y_points = ellipse_plot_points(P, c, n_points=100)

    plt.figure()
    plt.plot(x_points, y_points)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.grid(True)


def plot_ellipse_contour(P: np.ndarray, c: np.ndarray, n_points: int = 50):
    x = np.linspace(-2, 3, n_points)
    y = np.linspace(-3, 1, n_points)

    X_mesh, Y_mesh = np.meshgrid(x, y)

    xy_points = np.row_stack([X_mesh.ravel(), Y_mesh.ravel()])  # 2 x n_points^2
    xy_points_centered = xy_points - c

    z = (xy_points_centered * (P @ xy_points_centered)).sum(axis=0)
    Z_mesh = z.reshape(X_mesh.shape)
    plt.figure()
    plt.contour(X_mesh, Y_mesh, Z_mesh, levels=[1])
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.grid(True)


def ellipse_plot_points(P: np.ndarray, c: np.ndarray, n_points: int = 100):
    eig_vals, V = np.linalg.eigh(P)
    D_inv = np.diag(np.reciprocal(eig_vals))

    theta_points = np.linspace(0, 2 * np.pi, n_points)
    xy_unit_circ = np.row_stack([np.cos(theta_points), np.sin(theta_points)])
    xy_points = (V @ np.sqrt(D_inv) @ xy_unit_circ) + c
    x_points = xy_points[0, :]
    y_points = xy_points[1, :]

    return x_points, y_points


def matplotlib_ellipses():
    ellipse = Ellipse(xy=np.array([0, 0]), width=1, height=2, angle=30)

    ax = plt.subplot(111, aspect="equal")
    ellipse.set_alpha(0.1)
    ax.add_artist(ellipse)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])


def plot_concentric_ellipses():
    V = rotation_mat(7 * np.pi / 6)
    P = V @ np.diag(np.asarray([4, 0.5])) @ V.T

    ax = plt.subplot(111)
    # Color map this to plasma
    scales = list(reversed(np.arange(0.1, 1, 0.08)))
    n_ellipses = len(scales)
    for i, elem in enumerate(scales):
        x_points, y_points = ellipse_plot_points(P / elem, c=np.zeros((2, 1)))
        plt.plot(x_points, y_points, color=plt.cm.plasma(i / n_ellipses))
    # Turn off tick labels
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    for direction in ["top", "right", "bottom", "left"]:
        ax.spines[direction].set_visible(False)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "ellipses-concentric.png")


def rotation_mat(theta_val: float) -> np.array:
    return np.array(
        [
            [np.cos(theta_val), -np.sin(theta_val)],
            [np.sin(theta_val), np.cos(theta_val)],
        ]
    )


# Animation code
def animate_ellipse_creation(P: np.ndarray, c: np.ndarray):
    # eig_vals, V = np.linalg.eigh(P)
    # axes_lengths = np.sqrt(np.reciprocal(eig_vals))
    # if axes_lengths[0] > axes_lengths[1]:
    #     major_axis = V[:,0]
    #     major_axis_length = axes_lengths[0]
    #     minor_axis = V[:,1]
    #     minor_axis_length = axes_lengths[1]
    # else:
    #     major_axis = V[:,1]
    #     major_axis_length = axes_lengths[1]
    #     minor_axis = V[:,0]
    #     minor_axis_length = axes_lengths[0]

    # ccw_angle = np.arctan2(major_axis[0], major_axis[1])

    fig, ax = plt.subplots()
    plt.grid(True)
    theta_vals = np.linspace(0, 2 * np.pi, 400)
    x_data, y_data = [], []
    qx, qy = [], []
    u_data, v_data = [], []
    opacities = []
    fig_text = []
    a, b = 2, 0.5
    ccw_angle = np.pi / 3
    center = np.array([-1, 1]).reshape(-1, 1)

    # draw unit circle
    n_frames = 50
    for i in range(n_frames):
        n_thetas = theta_vals.size
        frac = (i + 1) / n_frames
        angles = theta_vals[: int(frac * n_thetas)]
        x_data.append(np.cos(angles))
        y_data.append(np.sin(angles))
        qx.append(np.zeros(2))
        qy.append(np.zeros(2))
        u_data.append(np.asarray([1, 0]))
        v_data.append(np.asarray([0, 1]))
        opacities.append(1)
        fig_text.append(r"$u$")

    # stretch unit circle
    n_frames = 24
    for i in range(n_frames):
        frac = (i + 1) / n_frames
        x_vals = np.cos(theta_vals)
        y_vals = np.sin(theta_vals)
        x_data.append(a**frac * x_vals)
        y_data.append(b**frac * y_vals)
        qx.append(np.zeros(2))
        qy.append(np.zeros(2))
        u_data.append(np.array([a**frac, 0]))
        v_data.append(np.array([0, b**frac]))
        opacities.append(1)
        fig_text.append(r"$D^{-1/2}u$")

    # pause
    n_frames = 12
    for i in range(n_frames):
        x_data.append(x_data[-1])
        y_data.append(y_data[-1])
        qx.append(qx[-1])
        qy.append(qy[-1])
        u_data.append(u_data[-1])
        v_data.append(v_data[-1])
        opacities.append(1)
        fig_text.append(r"$D^{-1/2}u$")

    # rotate ellipse
    n_frames = 30
    for i in range(n_frames):
        frac = (i + 1) / n_frames
        x_vals = a * np.cos(theta_vals)
        y_vals = b * np.sin(theta_vals)
        U = rotation_mat(frac * np.pi / 3)
        xy_vals = U @ np.row_stack([x_vals, y_vals])
        uv_vals = U @ np.diag([a, b])
        x_data.append(xy_vals[0, :])
        y_data.append(xy_vals[1, :])
        u_data.append(uv_vals[0, :])
        v_data.append(uv_vals[1, :])
        qx.append(np.zeros(2))
        qy.append(np.zeros(2))
        opacities.append(1)
        fig_text.append(r"$VD^{-1/2}u$")

    # pause
    n_frames = 12
    for i in range(n_frames):
        x_data.append(x_data[-1])
        y_data.append(y_data[-1])
        qx.append(qx[-1])
        qy.append(qy[-1])
        u_data.append(u_data[-1])
        v_data.append(v_data[-1])
        opacities.append(1)
        fig_text.append(r"$VD^{-1/2}u$")

    # translate
    n_frames = 24
    for i in range(n_frames):
        frac = (i + 1) / n_frames
        x_vals = a * np.cos(theta_vals)
        y_vals = b * np.sin(theta_vals)
        U = rotation_mat(ccw_angle)
        xy_vals = U @ np.row_stack([x_vals, y_vals]) + frac * center
        uv_vals = U @ np.diag([a, b])

        x_data.append(xy_vals[0, :])
        y_data.append(xy_vals[1, :])
        u_data.append(uv_vals[0, :])
        v_data.append(uv_vals[1, :])
        qx.append(frac * center[0])
        qy.append(frac * center[1])
        opacities.append(1)
        fig_text.append(r"$VD^{-1/2}u + c$")

    # pause
    n_frames = 24
    for i in range(n_frames):
        x_data.append(x_data[-1])
        y_data.append(y_data[-1])
        qx.append(qx[-1])
        qy.append(qy[-1])
        u_data.append(u_data[-1])
        v_data.append(v_data[-1])
        opacities.append(1)
        fig_text.append(r"$VD^{-1/2}u + c$")

    # fade out
    n_frames = 24
    for i in range(n_frames):
        x_data.append(x_data[-1])
        y_data.append(y_data[-1])
        qx.append(qx[-1])
        qy.append(qy[-1])
        u_data.append(u_data[-1])
        v_data.append(v_data[-1])
        opacities.append(0.85**i)
        fig_text.append("")

    (ln1,) = plt.plot([], [], linewidth=4)
    qax = ax.quiver(
        np.zeros(2),
        np.zeros(2),
        np.asarray([1, 0]),
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        pivot="tail",
        scale=1,
        units="xy",
    )
    label = ax.text(0.5, 0.1, r"$\{z: z=u, ||u||^2=1\}$", ha="left", va="center", transform=ax.transAxes, fontsize=16)

    def init():
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal", adjustable="box")

    def update(i):
        label.set_text(fig_text[i])
        ln1.set_data(x_data[i], y_data[i])
        ln1.set_alpha(opacities[i])
        qax.set_UVC(u_data[i], v_data[i])
        qax.set_offsets(np.column_stack([qx[i], qy[i]]))
        qax.set_alpha(opacities[i])

    ani = FuncAnimation(fig, update, np.arange(len(u_data)), init_func=init)
    writer = PillowWriter(fps=24)
    ani.save(GIF_DIR / "ellipse-rotation.gif", writer=writer)


if __name__ == "__main__":
    P = np.array([[1, -0.2], [-0.2, 0.4]])
    c = np.array([1, -1]).reshape(-1, 1)

    # plot_ellipse_contour(P, c, 50)
    # plot_ellipse_parametric(P, c)
    # plot_concentric_ellipses()
    matplotlib_ellipses()
    plt.show()
    # animate_ellipse_creation(P, c)
