---
title: "Interactive Plots with Plotly"
categories:
  - Programming
date: 2025-01-04 19:00:00 +0000
tags:
  - Visualization
  - Plotly
  - Python
toc: true
excerpt: "A demonstration of interactive Plotly visualizations embedded in blog posts."
---

This post demonstrates how to embed interactive Plotly visualizations in Jekyll blog posts. All plots below are fully interactive - try hovering, zooming, and using the controls!

## Sine Wave with Frequency Slider

The simplest interactive element is a slider that controls a single parameter. Here we vary the frequency of a sine wave from 0.5 to 5 Hz.

{% include plotly_figure.html
   src="/assets/examples/plots/sine_wave.html"
   height="550px"
   caption="Drag the slider to change the frequency of the sine wave."
%}

The key insight is that Plotly pre-computes traces for each slider position, then toggles visibility. This keeps the interactivity client-side with no server required.

## Gaussian Distribution with Multiple Sliders

For visualizations with multiple parameters, we can add additional sliders. This Gaussian distribution plot lets you independently control both the mean ($\mu$) and standard deviation ($\sigma$).

{% include plotly_figure.html
   src="/assets/examples/plots/gaussian.html"
   height="600px"
   caption="Adjust the mean and standard deviation to see how they affect the shape of the distribution."
%}

## Dropdown Menu for Function Selection

When you have discrete options rather than continuous parameters, a dropdown menu works well. This example lets you switch between different trigonometric functions.

{% include plotly_figure.html
   src="/assets/examples/plots/function_dropdown.html"
   height="550px"
   caption="Select a function from the dropdown menu."
%}

## Animated 3D Surface

Plotly also supports animations with play/pause controls. This rippling surface demonstrates a traveling wave in 3D.

{% include plotly_figure.html
   src="/assets/examples/plots/animated_surface.html"
   height="600px"
   caption="Click Play to animate the ripple. You can also rotate the 3D view by dragging."
%}

## How It Works

These interactive plots are created using Plotly in Python and exported as standalone HTML files:

```python
import numpy as np
import plotly.graph_objects as go

# Create the figure
x = np.linspace(0, 4 * np.pi, 500)
frequencies = [0.5, 1.0, 1.5, 2.0]

fig = go.Figure()

# Add a trace for each frequency value
for i, freq in enumerate(frequencies):
    fig.add_trace(go.Scatter(
        x=x,
        y=np.sin(freq * x),
        visible=(i == 0),  # Only first trace visible initially
    ))

# Create slider steps
steps = []
for i, freq in enumerate(frequencies):
    step = dict(
        method="update",
        args=[{"visible": [j == i for j in range(len(frequencies))]}],
        label=f"{freq}",
    )
    steps.append(step)

fig.update_layout(sliders=[dict(steps=steps)])

# Export as standalone HTML
fig.write_html("my_plot.html", include_plotlyjs="cdn")
```

The HTML file is then embedded in the post using a simple Jekyll include:

```liquid
{% raw %}{% include plotly_figure.html
   src="/assets/examples/plots/sine_wave.html"
   height="550px"
   caption="Your caption here"
%}{% endraw %}
```

This approach keeps the blog statically hosted on GitHub Pages while still providing rich interactivity.
