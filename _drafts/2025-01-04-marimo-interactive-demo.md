---
title: "Interactive Notebooks with Marimo"
categories:
  - Programming
date: 2025-01-04 20:00:00 +0000
tags:
  - Visualization
  - Marimo
  - Python
  - Interactive
toc: true
excerpt: "A demonstration of interactive Marimo notebooks running entirely in your browser via WebAssembly."
---

This post demonstrates how to embed fully interactive **Marimo notebooks** in Jekyll blog posts. Unlike static plots, these notebooks run Python directly in your browser using WebAssembly - no server required!

## Full Interactive Notebook

The notebook below includes three interactive demonstrations:

1. **Fourier Series** - See how summing sine waves approximates a square wave
2. **Damped Harmonic Oscillator** - Explore how damping and frequency affect oscillations
3. **2D Gaussian Distribution** - Visualize how standard deviations shape the distribution

Try adjusting the sliders and watch the plots update in real-time!

{% include marimo_notebook.html
   src="/assets/examples/notebooks/index.html"
   height="900px"
   caption="Interactive Marimo notebook running Python in your browser via WebAssembly."
%}

## How It Works

Marimo notebooks can be exported to WebAssembly (WASM), allowing them to run entirely client-side in the browser. This is powered by [Pyodide](https://pyodide.org/), which compiles Python to WebAssembly.

### Creating the Notebook

First, create a Marimo notebook with interactive elements:

```python
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt

# Create a slider UI element
n_terms = mo.ui.slider(start=1, stop=25, value=1, label="Number of terms")

# Use the slider value reactively
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = fourier_series(x, n_terms.value)

plt.plot(x, y)
plt.show()
```

### Exporting to WASM

Export the notebook using the Marimo CLI:

```bash
marimo export html-wasm notebook.py -o output_dir/ --mode run
```

This creates:
- `index.html` - The notebook entry point
- `assets/` - Required WASM runtime and dependencies

### Embedding in Jekyll

Use the `marimo_notebook.html` include to embed the notebook:

```liquid
{% raw %}{% include marimo_notebook.html
   src="/assets/examples/notebooks/index.html"
   height="900px"
   caption="Your caption here"
%}{% endraw %}
```

## Marimo vs Plotly

| Feature | Marimo WASM | Plotly |
|---------|-------------|--------|
| Interactivity | Full Python reactivity | Pre-computed traces |
| File size | Larger (~MB for WASM runtime) | Smaller (~KB per plot) |
| Libraries | Any pure Python package | Plotly only |
| Load time | Slower (loads Python runtime) | Instant |
| Best for | Complex notebooks, education | Simple interactive plots |

Use **Plotly** for quick, lightweight interactive charts. Use **Marimo WASM** when you need full Python interactivity with arbitrary libraries.

## Browser Support

Marimo WASM notebooks work in modern browsers:
- Chrome (recommended)
- Firefox
- Safari
- Edge

Note: The first load may take a few seconds as the Python runtime initializes.
