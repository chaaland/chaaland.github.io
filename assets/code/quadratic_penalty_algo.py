import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import imageio
from matplotlib.animation import FuncAnimation

pjoin = os.path.join


def rotation_matrix(ccw_angle: float):
  return np.asarray(
    [[np.cos(ccw_angle), -np.sin(ccw_angle)],
    [np.sin(ccw_angle), np.cos(ccw_angle)]],
  )

def create_quadratic_form(alpha: float, a: float, b: float):
  U = rotation_matrix(alpha)
  D = np.diag([1/a**2, 1/b**2])
  A = U @ D @ U.T

  return A

def create_elliptic_grid(
  alpha: float, 
  a: float, 
  b: float,
  n_radii: int=40, 
  n_theta: int=100, 
):
  U = rotation_matrix(alpha)

  theta_vals = np.linspace(0, 2 * np.pi, n_theta)
  r_vals = np.linspace(0, 0.4, n_radii)

  R_mesh, Theta_mesh = np.meshgrid(r_vals, theta_vals)
  scaled_x = (a * R_mesh * np.cos(Theta_mesh)).ravel()
  scaled_y = (b * R_mesh * np.sin(Theta_mesh)).ravel()
  coords_to_rotate = np.row_stack([scaled_x, scaled_y])
  plot_grid = U @ coords_to_rotate

  return plot_grid[0, :], plot_grid[1, :]

def evaluate_quadratic_form(
  A: np.ndarray, # (n, n)
  x: np.ndarray, # (n, d)
):
  return np.sum(x * (A @ x), axis=0)

def create_unit_circle_coords(n_theta: int):
  theta = np.linspace(0, 2 * np.pi, n_theta)
  x_vals = np.cos(theta)
  y_vals = np.sin(theta)

  return x_vals, y_vals

def plot_optimisation_problem(alpha: float, a: float, b: float, savefile: str):
  n_theta = 100
  n_radii = 40

  mesh_shape = (n_theta, n_radii)
  A = create_quadratic_form(alpha, a, b)
  x_coords, y_coords = create_elliptic_grid(
    alpha,
    a,
    b,
    n_radii=n_radii,
    n_theta=n_theta,
  )

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection="3d")
  plot_constrained_optimisation_surface(
    ax,
    lambda x: -1.0*evaluate_quadratic_form(A, x),
    x_coords.reshape(mesh_shape),
    y_coords.reshape(mesh_shape),
  )
  plot_unit_norm_constraint(
    ax,
    lambda x: -1.0 * evaluate_quadratic_form(A, x),
  )

  ax.set_xlim([-4, 4])
  ax.set_ylim([-4, 4])
  ax.set_xlabel(r"$x_1$")
  ax.set_ylabel(r"$x_2$")
  ax.set_zlabel(r"$-||Ax||^2$")
  ax.set_title("Spectral Norm Objective")
  plt.tight_layout()

  if savefile:
    plt.savefig(savefile)

def plot_constrained_optimisation_surface(
  ax,
  fn,
  x_mesh: np.ndarray,
  y_mesh: np.ndarray,
):
  xy_coords = np.row_stack([x_mesh.ravel(), y_mesh.ravel()])

  z_mesh = fn(xy_coords).reshape(x_mesh.shape)
  ax.plot_surface(
    x_mesh,
    y_mesh,
    z_mesh, 
    alpha=0.5,
    cmap="magma",
  )

def plot_unit_norm_constraint(ax, fn, n_theta: int=100):
  x_unit_circ, y_unit_circ = create_unit_circle_coords(n_theta)
  xy_unit_circ = np.row_stack([x_unit_circ, y_unit_circ])
  constraint_proj = fn(xy_unit_circ)
  ax.plot(x_unit_circ, y_unit_circ, linestyle="--")
  ax.plot(x_unit_circ, y_unit_circ, constraint_proj, "b", linewidth=2)

def plot_optimisation_level_sets(alpha, a, b, savefile: str):
  n_theta = 100
  n_radii = 40

  mesh_shape = (n_theta, n_radii)
  A = create_quadratic_form(alpha, a, b)
  x_coords, y_coords = create_elliptic_grid(
    alpha,
    a,
    b,
    n_radii=n_radii,
    n_theta=n_theta,
  )
  xy_coords = np.row_stack([x_coords, y_coords])
  z_mesh = -1.0 * evaluate_quadratic_form(A, xy_coords).reshape(mesh_shape)

  # Plot original objective level sets
  fig = plt.figure(figsize=(10, 10))
  lvls = [-1/9, -1/10, -1/12, -1/15, -1/18, -1/27, -1/36, -1/45, -1/81]
  plt.contour(
    x_coords.reshape(mesh_shape),
    y_coords.reshape(mesh_shape),
    z_mesh, 
    levels=lvls,
    cmap="magma",
  )

  x_unit_circ, y_unit_circ = create_unit_circle_coords(n_theta)
  plt.plot(x_unit_circ, y_unit_circ, 'b', label=f"||x||=1")

  plt.xlim([-2, 2])
  plt.ylim([-2, 2])
  plt.xlabel(r"$x_1$")
  plt.ylabel(r"$x_2$")
  plt.title(r"Contours of $-||Ax||^2$")
  plt.legend()
  plt.tight_layout()

  if savefile:
    plt.savefig(savefile)

def plot_relaxed_optimisation_surface(
  alpha: float,
  a: float,
  b: float,
  mu: float,
  azim: int,
):
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection="3d")
  x_vals = np.linspace(-2, 2, 100)
  y_vals = np.linspace(-2, 2, 100)

  A = create_quadratic_form(alpha, a, b)
  X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
  xy_coords = np.row_stack([X_mesh.ravel(), Y_mesh.ravel()])
  constraint_penalty = (np.linalg.norm(xy_coords, axis=0) - 1).reshape(X_mesh.shape)
  objective_penalty = -evaluate_quadratic_form(A, xy_coords).reshape(X_mesh.shape) 
  Z = objective_penalty + mu * np.square(constraint_penalty)

  ax.plot_surface(
    X_mesh,
    Y_mesh,
    Z, 
    alpha=0.8,
    cmap="magma",
  )

  ax.set_ylim(-2, 2)
  ax.set_xlim(-2, 2)
  ax.set_zlim(-0.4, 2)
  ax.set(xlabel=r'$x_1$', ylabel=r'$x_2$',
    title=rf'$-||Ax||^2 + {mu:.3} (||x||-1||)^2$')
  ax.view_init(elev=10., azim=azim)
  plt.tight_layout()
  # Used to return the plot as an image rray
  fig.canvas.draw()       # draw the canvas, cache the renderer
  image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
  image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  return image

def plot_relaxed_optimisation_contours(alpha, a, b, mu):
  # Data for plotting
  x_vals = np.linspace(-1.5, 1.5, 300)
  y_vals = np.linspace(-1.5, 1.5, 300)
  X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)

  A = create_quadratic_form(alpha, a, b)
  X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
  xy_coords = np.row_stack([X_mesh.ravel(), Y_mesh.ravel()])
  constraint_penalty = (np.linalg.norm(xy_coords, axis=0) - 1).reshape(X_mesh.shape)
  objective_penalty = -evaluate_quadratic_form(A, xy_coords).reshape(X_mesh.shape) 
  Z_mesh = objective_penalty + mu * np.square(constraint_penalty)

  fig, ax = plt.subplots()
  lvls = -np.asarray([-0.5, -0.1, 0, 1/1000, 1/500, 1/250, 1/100, 1/50, 0.03,0.04,0.05, 0.1,0.2,0.3, 0.4,0.5,1.0])[::-1]
  cs = ax.contourf(
    X_mesh, 
    Y_mesh, 
    Z_mesh, 
    levels=lvls, 
    cmap="magma",
  )
  ax.set(xlabel=r'$x_1$', ylabel=r'$x_2$',
    title=rf'$-||Ax||^2 + {mu:.3} (||x||-1||)^2$')

  # IMPORTANT ANIMATION CODE HERE
  # Used to keep the limits constant
  ax.set_xlim(-1.5, 1.5)
  ax.set_ylim(-1.5, 1.5)

  # Used to return the plot as an image rray
  fig.canvas.draw()       # draw the canvas, cache the renderer
  image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
  image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  return image

def plot_quadratic_penalty_minimisation_relaxation(alpha: float, a: float, b: float):
  A = create_quadratic_form(alpha, a, b)
  
  from quadratic_penalty import minimize_subject_to_constraints

  x_vals, mu_penalties = minimize_subject_to_constraints(
    objective=lambda x: x.T @ (-A) @ x, 
    equality_constraints= lambda x: np.linalg.norm(x) - 1,
    n=2,
    max_iters=10,
  )

  plt.figure(figsize=(10,10))
  constraint_violation = np.abs(np.linalg.norm(np.array(x_vals), axis=1) - 1)
  plt.semilogy(np.arange(constraint_violation.size)[2:], constraint_violation[2:], "-o")
  plt.xlabel(r"Iteration $k$", fontsize=14)
  plt.ylabel(r"$\log\left(|\, ||x|| - 1|\right) $", fontsize=14)
  plt.title("Constraint violation vs Iteration", fontsize=14)
  plt.tight_layout()
  plt.savefig(pjoin("..", "images", "quadratic-penalty-constraint-violation.png"))

  return x_vals, mu_penalties

def plot_2d_eigval_problem():
  alpha = 2 * np.pi / 3
  a = 3
  b = 9

  # Plot original objective surface
  # fname = pjoin("..", "images", "quadratic-penalty-spectral-norm-objective.png")
  # plot_optimisation_problem(alpha, a, b, savefile=fname)

  # fname = pjoin("..", "images", "quadratic-penalty-spectral-norm-contours.png")
  # plot_optimisation_level_sets(alpha, a, b, savefile=fname)

  # for penalty in [0.1,0.2,0.4,0.6,0.8,1.0]:
  #   fname = pjoin("..", "images", f"quadratic-penalty-spectral-norm-relaxed-objective-{penalty}.png")
  #   plot_relaxed_optimisation_surface(alpha, a, b, mu=penalty, savefile=fname)

  # n_mu = 50
  # mu_vals = np.linspace(0.0, 0.6, n_mu)
  # mu_vals = np.hstack([mu_vals, mu_vals[::-2]])
  # imageio.mimsave(
  #   "../gifs/quadratic-penalty-surface.gif", 
  #   [plot_relaxed_optimisation_surface(alpha, a, b, mu=mu, azim=i/n_mu * 360) for i, mu in enumerate(mu_vals)], 
  #   fps=5)

  # mu_vals = np.linspace(-6, -0.7, 50)
  # mu_vals = np.hstack([mu_vals, mu_vals[::-2]])
  # imageio.mimsave(
  #   '../gifs/quadratic-penalty-contours.gif',
  #   [plot_relaxed_optimisation_contours(alpha, a, b, 2**i) for i in mu_vals],
  #   fps=5,
  # )

  plot_quadratic_penalty_minimisation_relaxation(alpha, a, b)
  
def main():
  plot_2d_eigval_problem()
    
if __name__ == "__main__":
  main()
