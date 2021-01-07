import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab

plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)


def primal_objective(x):
    return x ** 2 + 1


def primal_constraint(x):
    return (x - 2) * (x - 4)


def lagrangian(x, lagrange_multiplier):
    return primal_objective(x) + lagrange_multiplier * primal_constraint(x)


def dual_fn(lagrange_multiplier):
    numerator = (-lagrange_multiplier**3 + 8 * lagrange_multiplier ** 2 + 10 * lagrange_multiplier + 1)
    denominator = (lagrange_multiplier + 1) ** 2
    return  numerator / denominator


def plot_primal(save_dir):
    n_plot_points = 5000
    x_left = -5
    x_right = 5
    x_plot = np.linspace(x_left, x_right, n_plot_points)
    f0 = primal_objective(x_plot)

    feasible_mask = primal_constraint(x_plot) <= 0
    feasible_region = x_plot[feasible_mask]
    feasible_f0 = f0[feasible_mask]

    opt_index = np.argmin(feasible_f0)
    opt_x = feasible_region[opt_index]
    opt_f0 = feasible_f0[opt_index]

    plt.figure(figsize=(10, 8))
    plt.plot(x_plot, f0, linewidth=2)
    plt.plot(feasible_region, np.zeros_like(feasible_region), color='k', linewidth=4)
    plt.scatter(feasible_region[0], 0, color='k', s=50)
    plt.scatter(feasible_region[-1], 0, color='k', s=50)
    plt.plot(feasible_region, feasible_f0, color='xkcd:darkish green', linewidth=4)
    
    plt.scatter(opt_x, opt_f0, s=100)
    plt.text(opt_x - 0.55, opt_f0 + 1, r'$f_0(x^{\star})$', fontsize=18, horizontalalignment='left')

    plt.xlim([x_left, x_right])
    plt.ylim(bottom=-5)
    plt.tight_layout()
    plt.savefig(save_dir / "primal_toy_problem.png")

    
def lagrangian_minimiser(lagrange_multiplier):
    return 3 * lagrange_multiplier / (1 + lagrange_multiplier)


def plot_lagrangian_contours(save_dir):
    n_plot_points = 100
    x_left = 0
    x_right = 5
    x_plot = np.linspace(x_left, x_right, n_plot_points)
    lambda_plot = np.linspace(0, 8, n_plot_points)
    X_mesh, lambda_mesh = np.meshgrid(x_plot, lambda_plot)
    lagrangian_mesh = lagrangian(X_mesh, lambda_mesh)

    fig = plt.figure(figsize=(7, 7))
    x_minimiser = lagrangian_minimiser(lambda_plot)
    plt.contour(X_mesh, lambda_mesh, lagrangian_mesh, cmap='hot', levels=50)
    plt.plot(x_minimiser, lambda_plot, color='xkcd:purple blue', linestyle='--')
    plt.text(1.7, 1.7, r'$(arg\, min_x L(x, \lambda), \lambda)$', fontsize=12, horizontalalignment='left', rotation=70)
    plt.xlabel('x')
    plt.ylabel(r'$\lambda$')
    plt.title(r'$L(x,\lambda) = x^2 + 1 + \lambda(x^2-6x+8)$')
    plt.tight_layout()
    plt.savefig(save_dir / "lagrange_infimum.png")
    

def plot_dual_function(save_dir):
    n_plot_points = 1000
    lambda_left = 0
    lambda_right = 15
    lambda_plot = np.linspace(lambda_left, lambda_right, n_plot_points)
    dual_plot = dual_fn(lambda_plot)
    max_ind = np.argmax(dual_plot)
    lambda_opt = lambda_plot[max_ind]
    dual_opt = dual_plot[max_ind]

    fig = plt.figure(figsize=(7, 7))
    plt.plot(lambda_plot, dual_plot, color='xkcd:purple blue')
    plt.scatter(lambda_opt, dual_opt, s=50)
    plt.text(lambda_opt, dual_opt+0.1, r'$g(\lambda^\star)$', fontsize=12, horizontalalignment='left')

    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$g(\lambda)$')
    plt.xlim([lambda_left, lambda_right])
    plt.ylim([-6, 6])
    plt.tight_layout()
    plt.savefig(save_dir / "dual_fn.png")
    

def plot_relaxed_primal(save_dir):
    n_plot_points = 5000
    x_left = -5
    x_right = 5
    x_plot = np.linspace(x_left, x_right, n_plot_points)
    f0 = primal_objective(x_plot)

    feasible_mask = primal_constraint(x_plot) <= 0
    feasible_region = x_plot[feasible_mask]
    feasible_f0 = f0[feasible_mask]

    opt_index = np.argmin(feasible_f0)
    opt_x = feasible_region[opt_index]
    opt_f0 = feasible_f0[opt_index]

    plt.figure(figsize=(10, 8))
    plt.plot(x_plot, f0, linewidth=2, label=r'$x^2+1$')
    plt.plot(feasible_region, np.zeros_like(feasible_region), color='k', linewidth=4)
    plt.scatter(feasible_region[0], 0, color='k', s=50)
    plt.scatter(feasible_region[-1], 0, color='k', s=50)
    
    for multiplier in [0.5, 1, 2, 4, 8, 16, 32]:
        plt.plot(x_plot, lagrangian(x_plot, multiplier), color='xkcd:purple blue', linestyle='--', alpha=0.5)
        plt.plot(feasible_region, lagrangian(feasible_region, multiplier), color='xkcd:purple blue', linestyle='--')
    plt.plot(feasible_region, lagrangian(feasible_region, multiplier), color='xkcd:purple blue', linestyle='--', label=r'$x^2 + 1 + \lambda_0 (x^2-6x+8)$')

    plt.plot(feasible_region, feasible_f0, color='xkcd:darkish green', linewidth=4, label=r'$x^2 + 1 + \mathbf{1}_\infty\{x^2-6x+8 \leq 0\}$')
    plt.plot([2, 2], [5, 30], color='xkcd:darkish green', linewidth=4)
    plt.plot([4, 4], [17, 30], color='xkcd:darkish green', linewidth=4)

    plt.scatter(opt_x, opt_f0, s=100)
    # plt.text(opt_x - 0.55, opt_f0 + 1, r'$f_0(x^{\star})$', fontsize=18, horizontalalignment='left')

    plt.xlim([x_left, x_right])
    plt.ylim(bottom=-5, top=30)
    plt.legend(prop={'size': 14})
    plt.tight_layout()
    plt.savefig(save_dir / "primal_dual_toy_problem.png")
    

def main(args):
    image_save_dir = Path(f"../../images/{Path(__file__).absolute().parent.name}")
    image_save_dir.mkdir(exist_ok=True, parents=True)
    # plot_primal(image_save_dir)
    # plot_lagrangian_contours(image_save_dir)
    # plot_relaxed_primal(image_save_dir)
    # plot_example()
    # plot_dual_function(image_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    main(args)
