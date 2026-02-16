"""
Demo script showing how to create interactive Plotly charts with sliders
for embedding in Jekyll blog posts.

Usage:
    python plotly_demo.py

This will generate HTML files in the plots/ directory.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def save_plotly_html(
    fig: go.Figure,
    filename: str,
    output_dir: Path = PLOT_DIR,
) -> None:
    """
    Save a Plotly figure as a standalone HTML file optimized for Jekyll embedding.

    Args:
        fig: Plotly figure object
        filename: Output filename (without .html extension)
        output_dir: Directory to save the HTML file
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "responsive": True,
    }

    fig.write_html(
        output_dir / f"{filename}.html",
        include_plotlyjs="cdn",
        full_html=True,
        config=config,
    )
    print(f"Saved: {output_dir / filename}.html")


def create_sine_wave_with_slider() -> None:
    """
    Create an interactive sine wave plot with a frequency slider.
    Demonstrates Plotly's built-in slider functionality.
    """
    x = np.linspace(0, 4 * np.pi, 500)
    frequencies = np.arange(0.5, 5.1, 0.5)

    fig = go.Figure()

    # Add traces for each frequency (only first one visible initially)
    for i, freq in enumerate(frequencies):
        y = np.sin(freq * x)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"f = {freq:.1f}",
                visible=(i == 0),
                line=dict(width=2),
            )
        )

    # Create slider steps
    steps = []
    for i, freq in enumerate(frequencies):
        step = dict(
            method="update",
            args=[
                {"visible": [j == i for j in range(len(frequencies))]},
                {"title": f"Sine Wave: sin({freq:.1f}x)"},
            ],
            label=f"{freq:.1f}",
        )
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Frequency: ", "suffix": " Hz"},
            pad={"t": 50},
            steps=steps,
        )
    ]

    # Style for dark theme compatibility
    fig.update_layout(
        title="Sine Wave: sin(0.5x)",
        xaxis_title="x",
        yaxis_title="y",
        sliders=sliders,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa"),
        xaxis=dict(
            gridcolor="rgba(128,128,128,0.2)",
            zerolinecolor="rgba(128,128,128,0.4)",
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.2)",
            zerolinecolor="rgba(128,128,128,0.4)",
        ),
        margin=dict(l=60, r=40, t=80, b=60),
    )

    save_plotly_html(fig, "sine_wave")


def create_gaussian_with_sliders() -> None:
    """
    Create an interactive Gaussian distribution plot with mean and std sliders.
    Demonstrates multiple sliders controlling different parameters.
    """
    x = np.linspace(-10, 10, 500)

    means = np.arange(-3, 3.5, 0.5)
    stds = np.arange(0.5, 3.1, 0.5)

    fig = go.Figure()

    # Create traces for all combinations
    trace_idx = 0
    for mu in means:
        for sigma in stds:
            y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"μ={mu:.1f}, σ={sigma:.1f}",
                    visible=(trace_idx == 0),
                    line=dict(width=2),
                )
            )
            trace_idx += 1

    n_means = len(means)
    n_stds = len(stds)

    # Create steps for mean slider
    mean_steps = []
    for i, mu in enumerate(means):
        visibility = [False] * (n_means * n_stds)
        # Show first std for this mean
        visibility[i * n_stds] = True
        step = dict(
            method="update",
            args=[{"visible": visibility}],
            label=f"{mu:.1f}",
        )
        mean_steps.append(step)

    # Create steps for std slider
    std_steps = []
    for j, sigma in enumerate(stds):
        visibility = [False] * (n_means * n_stds)
        # Show first mean for this std
        visibility[j] = True
        step = dict(
            method="update",
            args=[{"visible": visibility}],
            label=f"{sigma:.1f}",
        )
        std_steps.append(step)

    sliders = [
        dict(
            active=int(n_means // 2),
            currentvalue={"prefix": "Mean (μ): "},
            pad={"t": 50, "b": 10},
            steps=mean_steps,
            x=0.0,
            len=0.45,
        ),
        dict(
            active=0,
            currentvalue={"prefix": "Std Dev (σ): "},
            pad={"t": 50, "b": 10},
            steps=std_steps,
            x=0.55,
            len=0.45,
        ),
    ]

    fig.update_layout(
        title="Gaussian Distribution",
        xaxis_title="x",
        yaxis_title="Probability Density",
        sliders=sliders,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa"),
        xaxis=dict(
            gridcolor="rgba(128,128,128,0.2)",
            zerolinecolor="rgba(128,128,128,0.4)",
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.2)",
            zerolinecolor="rgba(128,128,128,0.4)",
        ),
        margin=dict(l=60, r=40, t=80, b=100),
    )

    save_plotly_html(fig, "gaussian")


def create_3d_surface_with_animation() -> None:
    """
    Create an animated 3D surface plot.
    Demonstrates animation frames with play/pause buttons.
    """
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)

    # Create frames for animation
    frames = []
    n_frames = 30
    for k in range(n_frames):
        t = k / n_frames * 2 * np.pi
        Z = np.sin(np.sqrt(X**2 + Y**2) - t)
        frames.append(
            go.Frame(
                data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis", showscale=False)],
                name=str(k),
            )
        )

    # Initial surface
    Z0 = np.sin(np.sqrt(X**2 + Y**2))
    fig = go.Figure(
        data=[go.Surface(x=X, y=Y, z=Z0, colorscale="Viridis", showscale=False)],
        frames=frames,
    )

    # Animation controls
    fig.update_layout(
        title="Animated Ripple Surface",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa"),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=50, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ],
        margin=dict(l=0, r=0, t=50, b=50),
    )

    save_plotly_html(fig, "animated_surface")


def create_dropdown_comparison() -> None:
    """
    Create a plot with dropdown menu to switch between different functions.
    Demonstrates dropdown widget functionality.
    """
    x = np.linspace(-2 * np.pi, 2 * np.pi, 500)

    functions = {
        "sin(x)": np.sin(x),
        "cos(x)": np.cos(x),
        "tan(x)": np.clip(np.tan(x), -10, 10),
        "sin(x) + cos(x)": np.sin(x) + np.cos(x),
        "sin²(x)": np.sin(x) ** 2,
    }

    fig = go.Figure()

    for i, (name, y) in enumerate(functions.items()):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                visible=(i == 0),
                line=dict(width=2),
            )
        )

    # Create dropdown menu
    buttons = []
    for i, name in enumerate(functions.keys()):
        visibility = [j == i for j in range(len(functions))]
        buttons.append(
            dict(
                label=name,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Function: {name}"},
                ],
            )
        )

    fig.update_layout(
        title="Function: sin(x)",
        xaxis_title="x",
        yaxis_title="y",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa"),
        xaxis=dict(
            gridcolor="rgba(128,128,128,0.2)",
            zerolinecolor="rgba(128,128,128,0.4)",
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.2)",
            zerolinecolor="rgba(128,128,128,0.4)",
            range=[-2, 2],
        ),
        margin=dict(l=60, r=40, t=100, b=60),
    )

    save_plotly_html(fig, "function_dropdown")


if __name__ == "__main__":
    print("Generating interactive Plotly charts...")
    print("-" * 50)

    create_sine_wave_with_slider()
    create_gaussian_with_sliders()
    create_3d_surface_with_animation()
    create_dropdown_comparison()

    print("-" * 50)
    print("Done! Charts saved to:", PLOT_DIR)
