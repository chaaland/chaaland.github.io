"""Generate interactive Plotly figures for gradient flow blog post."""
import numpy as np
import scipy as scp
import plotly.graph_objects as go
from pathlib import Path


def generate_ellipse_grid(
    a: float,
    b: float,
    angle: float,
    center: np.ndarray = None,
    r_low: float = 0.1,
    r_high: float = 1,
    n_r: int = 100,
    n_theta: int = 50,
):
    """Generate a grid of points on an ellipse."""
    if center is None:
        center = np.zeros((2,))

    r = np.linspace(r_low, r_high, n_r)
    theta = np.linspace(0, 2 * np.pi, n_theta)

    r_mesh, theta_mesh = np.meshgrid(r, theta)
    x_mesh = a * (r_mesh * np.cos(theta_mesh))
    y_mesh = b * (r_mesh * np.sin(theta_mesh))

    xy_stacked = np.hstack([x_mesh.reshape((-1, 1)), y_mesh.reshape((-1, 1))]).T

    rot_mat = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    plot_grid = rot_mat @ xy_stacked + center.reshape((2, 1))
    X = plot_grid[0, :].reshape(x_mesh.shape)
    Y = plot_grid[1, :].reshape(y_mesh.shape)

    return X, Y


def batch_quad_form(x: np.ndarray, A: np.ndarray):
    """Compute quadratic form x^T A x for batched x."""
    if A.ndim != 2:
        raise ValueError(f"Expected `A` to be a 2d array, got {A.ndim}")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected `A` to be a square array, got {A.shape}")

    n, _ = A.shape
    if x.shape[0] != n:
        raise ValueError(f"Expected first dimension of `x` to be {n}, got {x.shape[0]}")

    partial_quad = A @ x  # n x m
    return np.sum(x * partial_quad, axis=0)


def make_quad_form(a, b, theta):
    """Create a quadratic form matrix with given semi-axes and rotation."""
    D = np.diag(np.array([1 / a**2, 1 / b**2]))
    V = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    return V @ D @ V.T


def grad_flow_traj(x0, sigma, t):
    """Compute gradient flow trajectory using matrix exponential."""
    x_traj = np.array([scp.linalg.expm(-2 * sigma * tau) @ x0 for tau in t])
    return x_traj


def grad_descent_traj(x0: np.ndarray, sigma: np.ndarray, eta: float, n_steps: int):
    """Compute gradient descent trajectory."""
    x_traj = [x0.copy()]
    x = x0.copy()
    for _ in range(n_steps):
        x = x - eta * (2 * sigma @ x)
        x_traj.append(x.copy())
    return np.array(x_traj)


def create_ellipse_contours_figure():
    """Create interactive figure with ellipse contours, gradient flow, and gradient descent."""

    # Define parameter ranges
    theta_vals = np.linspace(0, np.pi, 7)  # 0 to pi
    a_vals = np.linspace(1, 4, 7)
    b_vals = np.linspace(1, 4, 7)
    eta_vals = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0])

    # Initial point
    x0 = np.array([-1.0, 2.0])

    # Create figure
    fig = go.Figure()

    # Visibility tracking
    trace_visibility = []
    n_traces_per_config = 0  # Will count: contours + flow + descent points + descent lines + axis lines

    # Generate all traces for all parameter combinations
    config_idx = 0
    for theta in theta_vals:
        for a in a_vals:
            for b in b_vals:
                for eta in eta_vals:
                    # Generate grid
                    X, Y = generate_ellipse_grid(
                        a=a, b=b, angle=theta,
                        r_low=0.01, r_high=5,
                        n_r=100, n_theta=100,
                    )

                    # Compute quadratic form
                    A = 0.5 * make_quad_form(a=a, b=b, theta=theta)
                    Z = batch_quad_form(np.stack([X.ravel(), Y.ravel()], axis=1).T, A).reshape(X.shape)

                    # Compute trajectories
                    t = np.linspace(0, 15, 100)
                    x_flow = grad_flow_traj(x0=x0, sigma=A, t=t)
                    x_descent = grad_descent_traj(x0=x0, sigma=A, eta=eta, n_steps=8)

                    # Compute loss values
                    loss_flow = np.array([batch_quad_form(x.reshape(2, 1), A)[0] for x in x_flow])
                    loss_descent = np.array([batch_quad_form(x.reshape(2, 1), A)[0] for x in x_descent])

                    visible = (config_idx == 0)
                    traces_this_config = []

                    # Contour trace
                    contour_trace = go.Contour(
                        x=X[0, :], y=Y[:, 0], z=Z,
                        contours=dict(
                            start=0.01,
                            end=2.0,
                            size=0.15,
                            showlabels=False,
                        ),
                        colorscale='Viridis',
                        showscale=False,
                        visible=visible,
                        hoverinfo='skip',
                    )
                    fig.add_trace(contour_trace)
                    traces_this_config.append(len(fig.data) - 1)

                    # Gradient flow trajectory
                    flow_trace = go.Scatter(
                        x=x_flow[:, 0], y=x_flow[:, 1],
                        mode='lines',
                        line=dict(color='dodgerblue', width=3),
                        name='Gradient Flow',
                        visible=visible,
                        showlegend=(config_idx == 0),
                        legendgroup='flow',
                    )
                    fig.add_trace(flow_trace)
                    traces_this_config.append(len(fig.data) - 1)

                    # Gradient descent trajectory - points with fading opacity
                    n_pts = len(x_descent)
                    alphas = [0.9 ** i for i in range(n_pts)][::-1]

                    for i in range(n_pts):
                        pt_trace = go.Scatter(
                            x=[x_descent[i, 0]], y=[x_descent[i, 1]],
                            mode='markers',
                            marker=dict(color='red', size=10, opacity=alphas[i]),
                            showlegend=False,
                            visible=visible,
                            hoverinfo='skip',
                        )
                        fig.add_trace(pt_trace)
                        traces_this_config.append(len(fig.data) - 1)

                    # Gradient descent lines with fading
                    for i in range(n_pts - 1):
                        line_trace = go.Scatter(
                            x=x_descent[i:i+2, 0], y=x_descent[i:i+2, 1],
                            mode='lines',
                            line=dict(color='red', width=2),
                            opacity=alphas[i],
                            showlegend=(i == 0 and config_idx == 0),
                            name='Gradient Descent' if i == 0 else None,
                            legendgroup='descent',
                            visible=visible,
                            hoverinfo='skip',
                        )
                        fig.add_trace(line_trace)
                        traces_this_config.append(len(fig.data) - 1)

                    # Principal axes
                    t_axis = np.linspace(-5, 5, 100)
                    axis1_trace = go.Scatter(
                        x=t_axis, y=np.tan(theta) * t_axis,
                        mode='lines',
                        line=dict(color='black', width=1, dash='dash'),
                        opacity=0.5,
                        showlegend=False,
                        visible=visible,
                        hoverinfo='skip',
                    )
                    fig.add_trace(axis1_trace)
                    traces_this_config.append(len(fig.data) - 1)

                    axis2_trace = go.Scatter(
                        x=t_axis, y=np.tan(theta + np.pi/2) * t_axis,
                        mode='lines',
                        line=dict(color='black', width=1, dash='dash'),
                        opacity=0.5,
                        showlegend=False,
                        visible=visible,
                        hoverinfo='skip',
                    )
                    fig.add_trace(axis2_trace)
                    traces_this_config.append(len(fig.data) - 1)

                    trace_visibility.append(traces_this_config)

                    if config_idx == 0:
                        n_traces_per_config = len(traces_this_config)

                    config_idx += 1

    total_configs = config_idx
    total_traces = len(fig.data)

    # Create sliders
    def make_visibility_array(active_config):
        vis = [False] * total_traces
        for idx in trace_visibility[active_config]:
            vis[idx] = True
        return vis

    def get_config_index(theta_idx, a_idx, b_idx, eta_idx):
        return (theta_idx * len(a_vals) * len(b_vals) * len(eta_vals) +
                a_idx * len(b_vals) * len(eta_vals) +
                b_idx * len(eta_vals) +
                eta_idx)

    # Create slider steps for theta
    theta_steps = []
    for i, theta in enumerate(theta_vals):
        step = dict(
            method="update",
            args=[{"visible": make_visibility_array(get_config_index(i, 0, 0, 0))}],
            label=f"{theta:.2f}",
        )
        theta_steps.append(step)

    # Create slider steps for a
    a_steps = []
    for i, a in enumerate(a_vals):
        step = dict(
            method="update",
            args=[{"visible": make_visibility_array(get_config_index(0, i, 0, 0))}],
            label=f"{a:.1f}",
        )
        a_steps.append(step)

    # Create slider steps for b
    b_steps = []
    for i, b in enumerate(b_vals):
        step = dict(
            method="update",
            args=[{"visible": make_visibility_array(get_config_index(0, 0, i, 0))}],
            label=f"{b:.1f}",
        )
        b_steps.append(step)

    # Create slider steps for eta
    eta_steps = []
    for i, eta in enumerate(eta_vals):
        step = dict(
            method="update",
            args=[{"visible": make_visibility_array(get_config_index(0, 0, 0, i))}],
            label=f"{eta:.1f}",
        )
        eta_steps.append(step)

    # Update layout
    fig.update_layout(
        title=dict(
            text="Gradient Flow vs Gradient Descent on Quadratic",
            font=dict(size=16),
        ),
        xaxis=dict(
            range=[-4, 4],
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            range=[-4, 4],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        ),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="θ = ", suffix=" rad"),
                pad=dict(t=50),
                steps=theta_steps,
                x=0.0,
                len=0.45,
                y=-0.05,
            ),
            dict(
                active=0,
                currentvalue=dict(prefix="a = "),
                pad=dict(t=50),
                steps=a_steps,
                x=0.55,
                len=0.45,
                y=-0.05,
            ),
            dict(
                active=0,
                currentvalue=dict(prefix="b = "),
                pad=dict(t=50),
                steps=b_steps,
                x=0.0,
                len=0.45,
                y=-0.20,
            ),
            dict(
                active=0,
                currentvalue=dict(prefix="η = "),
                pad=dict(t=50),
                steps=eta_steps,
                x=0.55,
                len=0.45,
                y=-0.20,
            ),
        ],
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        margin=dict(l=50, r=50, t=80, b=150),
        width=700,
        height=700,
    )

    return fig


def create_simple_contours_figure():
    """Create a simpler interactive figure with fewer parameter combinations."""

    # More limited parameter ranges for manageable file size
    theta_vals = np.array([0.0, np.pi/4, np.pi/2])
    a_vals = np.array([1.0, 2.0, 3.0])
    b_vals = np.array([1.0, 2.0, 3.0])
    eta_vals = np.array([0.3, 0.7, 1.0, 1.5, 2.0])

    # Initial point
    x0 = np.array([-1.0, 2.0])

    # Create figure
    fig = go.Figure()

    # Generate contour data for default params
    theta, a, b, eta = 0.0, 2.0, 1.0, 0.7

    X, Y = generate_ellipse_grid(
        a=a, b=b, angle=theta,
        r_low=0.01, r_high=5,
        n_r=100, n_theta=100,
    )

    A = 0.5 * make_quad_form(a=a, b=b, theta=theta)
    Z = batch_quad_form(np.stack([X.ravel(), Y.ravel()], axis=1).T, A).reshape(X.shape)

    # Compute trajectories
    t = np.linspace(0, 15, 100)
    x_flow = grad_flow_traj(x0=x0, sigma=A, t=t)
    x_descent = grad_descent_traj(x0=x0, sigma=A, eta=eta, n_steps=8)

    # Add contour
    fig.add_trace(go.Contour(
        x=X[0, :], y=Y[:, 0], z=Z,
        contours=dict(start=0.01, end=2.0, size=0.15),
        colorscale='Viridis',
        showscale=False,
        hoverinfo='skip',
    ))

    # Add gradient flow
    fig.add_trace(go.Scatter(
        x=x_flow[:, 0], y=x_flow[:, 1],
        mode='lines',
        line=dict(color='dodgerblue', width=3),
        name='Gradient Flow',
    ))

    # Add gradient descent with fading
    n_pts = len(x_descent)
    alphas = [0.9 ** i for i in range(n_pts)][::-1]

    # All descent points
    fig.add_trace(go.Scatter(
        x=x_descent[:, 0], y=x_descent[:, 1],
        mode='markers+lines',
        marker=dict(color='red', size=10),
        line=dict(color='red', width=2),
        name='Gradient Descent',
    ))

    # Principal axes
    t_axis = np.linspace(-5, 5, 100)
    fig.add_trace(go.Scatter(
        x=t_axis, y=np.tan(theta) * t_axis,
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        opacity=0.5,
        showlegend=False,
        hoverinfo='skip',
    ))

    fig.add_trace(go.Scatter(
        x=t_axis, y=np.tan(theta + np.pi/2) * t_axis,
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        opacity=0.5,
        showlegend=False,
        hoverinfo='skip',
    ))

    # Create dropdown menus instead of sliders for simpler interaction
    fig.update_layout(
        title=dict(
            text="Gradient Flow vs Gradient Descent on Quadratic",
            font=dict(size=18),
        ),
        xaxis=dict(
            range=[-4, 4],
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            title="x₁",
        ),
        yaxis=dict(
            range=[-4, 4],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            title="x₂",
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        width=650,
        height=650,
    )

    return fig


def create_loss_vs_step_figure():
    """Create figure showing loss vs step for gradient descent."""

    # Parameters
    x0 = np.array([-1.0, 2.0])
    theta, a, b = 0.0, 2.0, 1.0
    eta_vals = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    A = 0.5 * make_quad_form(a=a, b=b, theta=theta)
    lambda_max = np.max(np.linalg.eigvalsh(2 * A))  # eigenvalue of gradient Hessian

    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, eta in enumerate(eta_vals):
        x_descent = grad_descent_traj(x0=x0, sigma=A, eta=eta, n_steps=20)
        loss_vals = np.array([batch_quad_form(x.reshape(2, 1), A)[0] for x in x_descent])

        # Mark if diverging
        diverging = eta >= 2 / lambda_max
        dash = 'dash' if diverging else 'solid'

        fig.add_trace(go.Scatter(
            x=np.arange(len(loss_vals)),
            y=loss_vals,
            mode='lines+markers',
            name=f'η = {eta}' + (' (unstable)' if diverging else ''),
            line=dict(color=colors[i % len(colors)], width=2, dash=dash),
            marker=dict(size=6),
        ))

    # Add critical eta line annotation
    critical_eta = 2 / lambda_max

    fig.update_layout(
        title=dict(
            text=f"Loss vs Step (a={a}, b={b}, θ=0)<br>Critical η = 2/λ_max = {critical_eta:.2f}",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Step",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title="Loss f(x) = ½ xᵀAx",
            type="log",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        width=700,
        height=500,
    )

    return fig


def create_1d_sharpness_figure():
    """Create figure showing 1D quadratics with different sharpness values."""

    fig = go.Figure()

    x = np.linspace(-2, 2, 100)
    sharpness_vals = [0.5, 1, 2, 5]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, S in enumerate(sharpness_vals):
        y = 0.5 * S * x**2
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=f'S = {S}',
            line=dict(color=colors[i], width=2),
        ))

    fig.update_layout(
        title=dict(
            text="Quadratic Functions with Different Sharpness: f(x) = ½Sx²",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="x",
            range=[-2, 3],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title="f(x)",
            range=[-0.1, 4],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        width=700,
        height=450,
    )

    return fig


def create_interactive_quadratic_figure():
    """Create interactive figure with sliders for all parameters."""

    # Parameter values - keep small for reasonable file size
    theta_vals = np.array([0.0, 0.5, 1.0])
    a_vals = np.array([1.0, 2.0, 3.0])
    b_vals = np.array([1.0, 2.0, 3.0])
    eta_vals = np.array([0.3, 0.7, 1.0, 1.5, 2.0])

    x0 = np.array([-1.0, 2.0])

    fig = go.Figure()

    # Store trace indices for each configuration
    all_traces = []
    trace_idx = 0

    for theta in theta_vals:
        for a in a_vals:
            for b in b_vals:
                for eta in eta_vals:
                    # Generate grid
                    X, Y = generate_ellipse_grid(
                        a=a, b=b, angle=theta,
                        r_low=0.01, r_high=5,
                        n_r=80, n_theta=80,
                    )

                    A = 0.5 * make_quad_form(a=a, b=b, theta=theta)
                    Z = batch_quad_form(np.stack([X.ravel(), Y.ravel()], axis=1).T, A).reshape(X.shape)

                    t = np.linspace(0, 15, 100)
                    x_flow = grad_flow_traj(x0=x0, sigma=A, t=t)
                    x_descent = grad_descent_traj(x0=x0, sigma=A, eta=eta, n_steps=8)

                    visible = (trace_idx == 0)
                    config_traces = []

                    # Contour
                    fig.add_trace(go.Contour(
                        x=X[0, :], y=Y[:, 0], z=Z,
                        contours=dict(start=0.01, end=2.0, size=0.15),
                        colorscale='Viridis',
                        showscale=False,
                        visible=visible,
                        hoverinfo='skip',
                    ))
                    config_traces.append(len(fig.data) - 1)

                    # Gradient flow
                    fig.add_trace(go.Scatter(
                        x=x_flow[:, 0], y=x_flow[:, 1],
                        mode='lines',
                        line=dict(color='dodgerblue', width=3),
                        name='Gradient Flow',
                        visible=visible,
                        showlegend=(trace_idx == 0),
                        legendgroup='flow',
                    ))
                    config_traces.append(len(fig.data) - 1)

                    # Gradient descent
                    fig.add_trace(go.Scatter(
                        x=x_descent[:, 0], y=x_descent[:, 1],
                        mode='markers+lines',
                        marker=dict(color='red', size=8),
                        line=dict(color='red', width=2),
                        name='Gradient Descent',
                        visible=visible,
                        showlegend=(trace_idx == 0),
                        legendgroup='descent',
                    ))
                    config_traces.append(len(fig.data) - 1)

                    # Principal axes
                    t_axis = np.linspace(-5, 5, 50)
                    fig.add_trace(go.Scatter(
                        x=t_axis, y=np.tan(theta) * t_axis if abs(np.cos(theta)) > 1e-6 else np.zeros_like(t_axis),
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dash'),
                        opacity=0.5,
                        showlegend=False,
                        visible=visible,
                        hoverinfo='skip',
                    ))
                    config_traces.append(len(fig.data) - 1)

                    perp_angle = theta + np.pi/2
                    fig.add_trace(go.Scatter(
                        x=t_axis, y=np.tan(perp_angle) * t_axis if abs(np.cos(perp_angle)) > 1e-6 else np.zeros_like(t_axis),
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dash'),
                        opacity=0.5,
                        showlegend=False,
                        visible=visible,
                        hoverinfo='skip',
                    ))
                    config_traces.append(len(fig.data) - 1)

                    all_traces.append({
                        'theta': theta,
                        'a': a,
                        'b': b,
                        'eta': eta,
                        'traces': config_traces,
                    })

                    trace_idx += 1

    total_traces = len(fig.data)
    n_configs = len(all_traces)

    # Create update functions for sliders
    def get_visibility(theta_idx, a_idx, b_idx, eta_idx):
        config_idx = (theta_idx * len(a_vals) * len(b_vals) * len(eta_vals) +
                     a_idx * len(b_vals) * len(eta_vals) +
                     b_idx * len(eta_vals) +
                     eta_idx)
        vis = [False] * total_traces
        for t_idx in all_traces[config_idx]['traces']:
            vis[t_idx] = True
        return vis

    # Build slider steps
    theta_steps = []
    for i, theta in enumerate(theta_vals):
        step = dict(
            method="update",
            args=[{"visible": get_visibility(i, 0, 0, 0)}],
            label=f"{theta:.1f}",
        )
        theta_steps.append(step)

    a_steps = []
    for i, a in enumerate(a_vals):
        step = dict(
            method="update",
            args=[{"visible": get_visibility(0, i, 0, 0)}],
            label=f"{a:.1f}",
        )
        a_steps.append(step)

    b_steps = []
    for i, b in enumerate(b_vals):
        step = dict(
            method="update",
            args=[{"visible": get_visibility(0, 0, i, 0)}],
            label=f"{b:.1f}",
        )
        b_steps.append(step)

    eta_steps = []
    for i, eta in enumerate(eta_vals):
        step = dict(
            method="update",
            args=[{"visible": get_visibility(0, 0, 0, i)}],
            label=f"{eta:.1f}",
        )
        eta_steps.append(step)

    fig.update_layout(
        title=dict(
            text="Gradient Flow vs Gradient Descent on Quadratic",
            font=dict(size=16),
        ),
        xaxis=dict(
            range=[-4, 4],
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            title="x₁",
        ),
        yaxis=dict(
            range=[-4, 4],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            title="x₂",
        ),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="θ = ", suffix=" rad"),
                pad=dict(t=40),
                steps=theta_steps,
                x=0.0,
                len=0.45,
                y=-0.02,
            ),
            dict(
                active=0,
                currentvalue=dict(prefix="a = "),
                pad=dict(t=40),
                steps=a_steps,
                x=0.55,
                len=0.45,
                y=-0.02,
            ),
            dict(
                active=0,
                currentvalue=dict(prefix="b = "),
                pad=dict(t=40),
                steps=b_steps,
                x=0.0,
                len=0.45,
                y=-0.15,
            ),
            dict(
                active=0,
                currentvalue=dict(prefix="η = "),
                pad=dict(t=40),
                steps=eta_steps,
                x=0.55,
                len=0.45,
                y=-0.15,
            ),
        ],
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        margin=dict(l=50, r=50, t=80, b=130),
        width=700,
        height=700,
    )

    return fig


def main():
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)

    print("Generating 1D sharpness figure...")
    fig1 = create_1d_sharpness_figure()
    fig1.write_html(output_dir / "sharpness_1d.html", include_plotlyjs="cdn")

    print("Generating loss vs step figure...")
    fig2 = create_loss_vs_step_figure()
    fig2.write_html(output_dir / "loss_vs_step.html", include_plotlyjs="cdn")

    print("Generating simple contours figure...")
    fig3 = create_simple_contours_figure()
    fig3.write_html(output_dir / "contours_simple.html", include_plotlyjs="cdn")

    print("Generating interactive quadratic figure (this may take a moment)...")
    fig4 = create_interactive_quadratic_figure()
    fig4.write_html(output_dir / "quadratic_interactive.html", include_plotlyjs="cdn")

    print(f"Done! Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
