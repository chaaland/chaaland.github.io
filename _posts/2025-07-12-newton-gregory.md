---
title: "Newton-Gregory Interpolation"
categories:
  - Mathematics
date:   2025-07-12 12:30:00 +0100
mathjax: true
tags:
  - Numerical methods
  - Interpolation
toc: true
# classes: wide
excerpt: "Interpolate equally-spaced data efficiently and discover its connection to Taylor series."
header: 
  overlay_image: assets/2025/newton-gregory/images/splash_image.png
  overlay_filter: 0.6
---

This post explores the Newton-Gregory interpolation method, an efficient algorithm for polynomial interpolation that's particularly useful when dealing with equally-spaced data points.
We'll also see how it relates to Taylor polynomials.

## Polynomial Interpolation Recap

As discussed in a [previous post]({{ site.baseurl }}/algorithms/lagrange-interpolation), we can find a polynomial that passes through a given set of points using various approaches. For a quadratic polynomial:

$$y = a_2 x^2 + a_1 x + a_0.$$

Given 3 points $$(x_0, y_0), (x_1, y_1), (x_2, y_2)$$, we can solve for the coefficients using the system:

$$
\begin{bmatrix}
y_0\\
y_1\\
y_2\\
\end{bmatrix}
=
\begin{bmatrix}
x_0^2 & x_0 & 1\\
x_1^2 & x_1 & 1\\
x_2^2 & x_2 & 1\\
\end{bmatrix}
\begin{bmatrix}
a_2\\
a_1\\
a_0\\
\end{bmatrix}
$$

The problem with this approach is that the Vandermonde matrix can be poorly conditioned.
It also requires $$O(n^3)$$ algorithmic complexity to solve which becomes relevant for higher order polynomials.

Alternatively, using the Lagrange basis functions:

$$
\begin{align*}
y &= a_0 {(x-x_1)(x-x_2) \over (x_0 - x_1)(x_0 - x_2)} + a_1 {(x-x_0)(x-x_2) \over (x_1 - x_0)(x_1 - x_2)} + a_2 {(x-x_0)(x-x_1) \over (x_2 - x_0)(x_2 - x_1)}\\
\quad\\
&= a_0 \ell_0(x) + a_1 \ell_1(x) + a_2 \ell_2(x)\\
\end{align*}
$$

By evaluating the polynomial at $$x_0, x_1,$$ and $$x_2$$, we get the trivial system of equations

$$
\begin{bmatrix}
y_0\\
y_1\\
y_2\\
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1\\
\end{bmatrix}
\begin{bmatrix}
a_0\\
a_1\\
a_2\\
\end{bmatrix}
$$

The strength of Lagrange interpolation is that finding the coefficients is trivial.
You don't need to implement complicated algorithms like Gaussian elimination or QR factorisation, just compute each coefficient directly.

A drawback of Lagrange interpolation however, is that we have no way of updating the coefficients.
Say we were interpolating a regularly spaced time series and a new fourth point comes in.
We'd then need a cubic polynomial, but with Lagrange, there's no way to reuse the quadratic polynomial coefficients.
We have no choice but to resolve for all of the coefficients.

## Newton-Gregory Interpolation

Newton-Gregory is an alternative approach to polynomial interpolation that addresses some of the short comings of the previous two approaches.

### Quadratic Case

The subject of this post is yet another parameterisation of the interpolating polynomial, this time given by

$$ y = a_0 + a_1 (x-x_0) + a_2(x-x_0)(x-x_1)$$

Enforcing that this polynomial pass through the points $$(x_0, y_0), (x_1, y_1), (x_2, y_2)$$ we get the system of equations

$$
\begin{bmatrix}
y_0\\
y_1\\
y_2\\
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 0\\
1 & x_1-x_0 & 0\\
1 & x_2-x_0 & (x_2-x_0)(x_2 -x_1)\\
\end{bmatrix}
\begin{bmatrix}
a_0\\
a_1\\
a_2\\
\end{bmatrix}.
$$

If we make the further assumption that the points are evenly spaced with spacing $$h$$ (as they might be when interpolating points from a lookup table), we get the following linear system

$$
\begin{bmatrix}
y_0\\
y_1\\
y_2\\
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 0\\
1 & h & 0\\
1 & 2h & 2h^2\\
\end{bmatrix}
\begin{bmatrix}
a_0\\
a_1\\
a_2\\
\end{bmatrix}
$$

This triangular system makes [forward substitution](https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution) trivial, unlike the dense systems produced by the Vandermonde matrix

- From the first equation $$a_0 = y_0$$
- From the second equation $$y_1 = a_0 + ha_1 \implies a_1 = (y_1 - y_0) / h$$
- From the final equation, substituting in $$a_0$$ and $$a_1$$, $$y_2 = a_0 + 2 h a_1 + 2h^2 a_2 \implies a_2 = {y_2 - 2y_1 + y_0 \over 2h^2}$$

To make the pattern clearer, we can define _backward difference operators_.
The first order backward difference operator is

$$\Delta y_i = y_i - y_{i-1}.$$

We can also define a second difference operator as the backward difference operator applied to the backward difference

$$\Delta^2 y_i = \Delta (\Delta y_i) = \Delta (y_i - y_{i-1}) = y_i - 2y_{i-1} + y_{i-2}$$

Using this notation we can rewrite our coefficients succinctly as

$$
\begin{align*}
a_0 &= y_0\\
a_1 &= {1 \over h}\Delta y_1\\
a_2 &= {1 \over 2h^2}\Delta^2 y_2\\
\end{align*}
$$

Substituting our coefficients back into the polynomial,

$$
\begin{align*}
y &= y_0 + {y_1 - y_0 \over h}(x-x_0) + {y_2 - 2y_1 + y_0 \over 2h^2}(x - x_0)(x - x_1)\\
&= y_0 + {\Delta y_1 \over h}(x-x_0) + {\Delta^2 y_2 \over 2h^2}(x - x_0)(x - x_0 - h)\\
&= y_0 +{x-x_0 \over h} \Delta y_1 + {x - x_0 \over h} {x - x_0 - h \over h}{\Delta^2 y_2 \over 2}\\
\end{align*}
$$

Defining the normalised distance from $$x_0$$ as

$$ u = {x-x_0 \over h},$$

the polynomial simplifies to

$$
y= y_0 + u \Delta y_1  + {u(u-1) \over 2} \Delta^2 y_2.
$$

### General Case

The pattern for the system of equations might not be entirely obvious from the quadratic case, so let's consider the case of a fourth degree polynomial

$$
\begin{align*}
y = &a_0 + a_1 (x-x_0) + a_2(x-x_0)(x-x_1) \\
&+ a_3(x-x_0)(x-x_1)(x-x_2) + a_4(x-x_0)(x-x_1)(x-x_2)(x-x_3)
\end{align*}
$$

Plugging in evenly spaced points $$x_i = x_0 + i\cdot h$$ and their corresponding $$y$$-values, $$y_i$$, we get the following system of equations

$$

\begin{bmatrix}
y_0\\
y_1\\
y_2\\
y_3\\
y_4
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
1 & h & 0 & 0 & 0\\
1 & 2h & 2h^2 & 0 & 0\\
1 & 3h & 3 \cdot 2h^2 & 3\cdot 2 \cdot 1 h^3 & 0\\
1 & 4h & 4\cdot 3 h^2 & 4\cdot 3 \cdot 2 h^3 & 4 \cdot 3 \cdot 2 \cdot 1 h^4\\
\end{bmatrix}
\begin{bmatrix}
a_0\\
a_1\\
a_2\\
a_3\\
a_4
\end{bmatrix}.
$$

The entries of the matrix have the simple formula (assuming 1 based indexing)

$$
a_{ij} =
\begin{cases}
0 & i < j\\
{(i-1)! \over (i-j)!}h^{j-1} & i \ge j
\end{cases}
$$

We can easily create this matrix with just a few lines of python

{% highlight python %}
import math
import numpy as np

def create_matrix(n, h):
    A = np.empty((n, n))

    # zero based index rather 1 based in the previous formula
    for i in range(n):
        for j in range(i + 1):
            A[i, j] = math.perm(i, j) * h**j
    return A
{% endhighlight %}

Continuing with forward substitution, we can easily find the coefficient of the cubic term is

$$
\begin{align*}
a_3 &= {y_3 - 3y_2 +3y_1 - y_0 \over 6h^3}\\
&= {1 \over 3!h^3}\Delta^3 y_3
\end{align*}
$$

The general form of the polynomial coefficients is

$$a_n = {1 \over n!h^n} \Delta^n y_n.$$

Plugging back in, we get the Newton-Gregory interpolating polynomial

$$y = y_0 + u \Delta y_1  + {u(u-1) \over 2} \Delta^2 y_2 + {u(u-1)(u-2) \over 3!} \Delta^3 y_3 + \cdots$$

where the general term is

$${u(u-1)(u-2)\cdots(u-n+1) \over n!} \Delta^n y_n$$

and the $$k$$-th order difference is $$\Delta^k y_n = \Delta^{k-1}y_n - \Delta^{k-1} y_{n-1}$$.
The following code shows an implementation of the Newton-Gregory method.

{% highlight python %}
import math

import numpy as np

def bwd_diff(x, i: int, order: int = 1):
    if i < 0:
        raise ValueError(f"Index must be positive, got {i}")

    if order < 0:
        raise ValueError(f"Backward difference order expected to be >0, got {order}")
    elif order == 0:
        return x[i]
    else:
        return bwd_diff(x, i, order=order - 1) - bwd_diff(x, i - 1, order=order - 1)

def newton_gregory(x: np.ndarray, x_0: float, ys: np.ndarray, h: float) -> np.ndarray:
    poly_order = len(ys) - 1
    u = (x - x_0) / h

    coeff = 1
    result = np.zeros_like(x)
    for k in range(poly_order + 1):
        result += coeff * bwd_diff(ys, k, order=k)
        coeff *= (u - k) / (k + 1)

    return result
{% endhighlight python %}

## Relation to Taylor Polynomials

Recall the $$n$$-th degree Taylor expansion of a continuous function $$f$$ about a point $$x_0$$ is

$$
\begin{align*}
f(x) &= f(x_0) + f'(x_0)(x-x_0) + {f''(x_0)\over 2}(x-x_0)^2 + \cdots + {f^{(n)}(x_0)\over n!}(x-x_0)^n\\
&= \sum_{k=0}^n {f^{(k)}(x_0) \over k!}(x-x_0)^k .
\end{align*}
$$

It turns out that Newton-Gregory interpolation is actually the discrete analog of Taylor polynomials!

Returning back to one of our expressions for the Newton-Gregory polynomial, we have

$$
y = y_0 + {\Delta y_1 \over h}(x-x_0) + {\Delta^2 y_2 \over 2h^2}(x - x_0)(x - x_0 - h) + \cdots
$$

It is clear from the definition of the derivative that

$$
\lim_{h\rightarrow 0} {\Delta y_1 \over h}(x-x_0) = f'(x_0)(x-x_0)
$$

Using the backward difference formulation of the derivative we have

$$
f'(x_0) = \lim_{h\rightarrow 0} {f(x)- f(x-h) \over h}.
$$

Since the second derivative is simply the derivative of the first derivative

$$
f''(x_0) = \lim_{h\rightarrow 0} {f'(x_0)- f'(x_0 - h) \over h}.
$$

By applying the backward difference approximation to the derivatives, we have

$$
\begin{align*}
f''(x_0) &= \lim_{h\rightarrow 0} {1 \over h}\left({f(x_0) - f(x_0-h) \over h} - {f(x_0-h) - f(x_0 - 2h) \over h}\right)\\
&= \lim_{h\rightarrow 0} {f(x_0) - 2f(x_0-h) + f(x_0-2h) \over h^2}.
\end{align*}
$$

From this formulation of the second derivative, we can see that

$$
\lim_{h\rightarrow 0} {\Delta^2 y_2 \over 2h^2}(x - x_0)(x - x_0 - h) = {1 \over 2}f''(x_0)(x - x_0)^2.
$$

Continuing with the higher order terms, we would see that in the limit as $$h$$ approaches 0, the Newton-Gregory formula becomes exactly the Taylor polynomial.

Though Taylor polynomials use derivatives and Newton-Gregory uses finite differences, in the limit as the spacing tends to 0, differences become derivatives!

The widget below shows how the Newton-Gregory polynomial approaches the Taylor approximation of $$\log(1+x)$$ as $$h$$ decreases towards 0.

<style>
.ng-widget {
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 12px;
  background: #161b22;
  margin: 1rem auto 1.5rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  max-width: 700px;
}

.ng-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  align-items: center;
  margin-bottom: 10px;
}

.ng-controls label {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.8rem;
  color: #c9d1d9;
  line-height: 1.1;
}

.ng-controls input[type="range"] {
  width: 100px;
  height: 6px;
  accent-color: #58a6ff;
}

.ng-readout {
  font-variant-numeric: tabular-nums;
  color: #8b949e;
  font-size: 0.85rem;
  min-width: 3em;
}

.ng-plot {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 4px;
  background: #0d1117;
  border: 1px solid #30363d;
}

.ng-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 16px;
  margin-top: 10px;
  font-size: 0.82rem;
  color: #8b949e;
}

.ng-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.ng-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}

.ng-line {
  width: 20px;
  height: 3px;
  display: inline-block;
}

.ng-info {
  font-size: 0.8rem;
  color: #8b949e;
  margin-top: 8px;
  padding: 8px 12px;
  background: rgba(88, 166, 255, 0.05);
  border-radius: 4px;
  border-left: 3px solid #58a6ff;
}

.ng-button {
  padding: 5px 10px;
  border: 1px solid #30363d;
  border-radius: 6px;
  background: #0d1117;
  color: #c9d1d9;
  font-size: 0.8rem;
  font-family: inherit;
  cursor: pointer;
  transition: all 0.2s ease;
}

.ng-button:hover {
  background: #21262d;
  border-color: #58a6ff;
}

@media (max-width: 600px) {
  .ng-controls input[type="range"] {
    width: 80px;
  }
}
</style>

<div class="ng-widget" id="ng-widget">
  <div class="ng-controls">
    <label>
      h (spacing)
      <input type="range" min="0.01" max="0.2" value="0.10" step="0.01" data-param="h">
      <span class="ng-readout" data-readout="h">0.10</span>
    </label>
    <label>
      N (degree)
      <input type="range" min="2" max="6" value="3" step="1" data-param="n">
      <span class="ng-readout" data-readout="n">3</span>
    </label>
    <button type="button" class="ng-button" id="ng-reset">Reset</button>
  </div>
  <svg class="ng-plot" id="ng-svg" viewBox="0 0 600 400" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="600" height="400" fill="#0d1117"></rect>
    <g id="ng-grid"></g>
    <g id="ng-axes"></g>
    <path id="ng-true-path" fill="none" stroke="#58a6ff" stroke-width="3.5" stroke-linecap="round"></path>
    <path id="ng-taylor-path" fill="none" stroke="#f78166" stroke-width="3" stroke-linecap="round" stroke-dasharray="6,4"></path>
    <path id="ng-interp-path" fill="none" stroke="#7ee787" stroke-width="3" stroke-linecap="round" stroke-dasharray="10,5"></path>
    <g id="ng-sample-points"></g>
  </svg>
  <div class="ng-legend">
    <span class="ng-chip"><span class="ng-line" style="background:#58a6ff"></span>log(1+x)</span>
    <span class="ng-chip"><span class="ng-line" style="background:#f78166; background-image: repeating-linear-gradient(90deg, #f78166 0, #f78166 4px, transparent 4px, transparent 7px);"></span>Taylor</span>
    <span class="ng-chip"><span class="ng-line" style="background:#7ee787; background-image: repeating-linear-gradient(90deg, #7ee787 0, #7ee787 8px, transparent 8px, transparent 12px);"></span>Newton-Gregory</span>
    <span class="ng-chip"><span class="ng-dot" style="background:#7ee787"></span>Sample points</span>
  </div>
  <div class="ng-info" id="ng-info">
    As h → 0, Newton-Gregory converges to the Taylor polynomial
  </div>
</div>

## Conclusion

The Newton-Gregory method is useful when

- data is evenly spaced such as data from a numerical approximation table or time series sampled with a regular period
- new data points are arriving in an online fashion. Newton-Gregory allows you to solve for just one new coefficient without needing to resolve for the others.
- the Vandermonde matrix is ill-conditioned

In addition, Newton-Gregory interpolation provides a connection between discrete and continuous functions.

<script>
(function() {
  // Layout constants
  const W = 600, H = 400;
  const margin = { left: 50, right: 30, top: 30, bottom: 40 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  // Coordinate bounds
  const xMin = -0.25, xMax = 1.05;
  const yMin = -0.35, yMax = 0.85;

  // Get DOM elements
  const svg = document.getElementById('ng-svg');
  const gridG = document.getElementById('ng-grid');
  const axesG = document.getElementById('ng-axes');
  const truePath = document.getElementById('ng-true-path');
  const taylorPath = document.getElementById('ng-taylor-path');
  const interpPath = document.getElementById('ng-interp-path');
  const samplePointsG = document.getElementById('ng-sample-points');
  const infoDiv = document.getElementById('ng-info');

  // Coordinate transforms
  function toSvgX(x) { return margin.left + (x - xMin) / (xMax - xMin) * plotW; }
  function toSvgY(y) { return margin.top + (yMax - y) / (yMax - yMin) * plotH; }

  // Backward difference operator (recursive with memoization)
  function bwdDiff(ys, i, order, memo) {
    const key = `${i},${order}`;
    if (memo[key] !== undefined) return memo[key];

    let result;
    if (order === 0) {
      result = ys[i];
    } else if (i < order) {
      result = 0;
    } else {
      result = bwdDiff(ys, i, order - 1, memo) - bwdDiff(ys, i - 1, order - 1, memo);
    }
    memo[key] = result;
    return result;
  }

  // Newton-Gregory interpolation
  function newtonGregory(xs, x0, ys, h) {
    const n = ys.length;
    const result = new Array(xs.length);
    const memo = {};

    for (let j = 0; j < xs.length; j++) {
      const u = (xs[j] - x0) / h;
      let coeff = 1;
      let sum = 0;

      for (let k = 0; k < n; k++) {
        sum += coeff * bwdDiff(ys, k, k, memo);
        coeff *= (u - k) / (k + 1);
      }
      result[j] = sum;
    }
    return result;
  }

  // Taylor series for log(1+x): x - x²/2 + x³/3 - ...
  function taylorLog1p(xs, degree) {
    const result = new Array(xs.length);

    for (let j = 0; j < xs.length; j++) {
      const x = xs[j];
      let sum = 0;
      let sign = 1;
      let xPow = x;

      for (let k = 1; k <= degree; k++) {
        sum += sign * xPow / k;
        sign *= -1;
        xPow *= x;
      }
      result[j] = sum;
    }
    return result;
  }

  // Generate evenly spaced array
  function linspace(start, end, n) {
    const step = (end - start) / (n - 1);
    return Array.from({length: n}, (_, i) => start + i * step);
  }

  // log(1+x)
  function log1p(x) {
    return Math.log(1 + x);
  }

  // Convert array of [x, y] to SVG path
  function toPath(xs, ys) {
    let d = '';
    for (let i = 0; i < xs.length; i++) {
      const sx = toSvgX(xs[i]);
      const sy = toSvgY(ys[i]);
      // Skip points outside bounds
      if (ys[i] < yMin - 0.5 || ys[i] > yMax + 0.5) {
        if (d && d[d.length - 1] !== ' ') d += ' ';
        continue;
      }
      if (!d || d[d.length - 1] === ' ') {
        d += `M ${sx} ${sy}`;
      } else {
        d += ` L ${sx} ${sy}`;
      }
    }
    return d;
  }

  // Draw grid
  function drawGrid() {
    gridG.innerHTML = '';

    // Vertical lines
    for (let v = 0; v <= 1; v += 0.25) {
      const sx = toSvgX(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', sx); line.setAttribute('y1', margin.top);
      line.setAttribute('x2', sx); line.setAttribute('y2', H - margin.bottom);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }

    // Horizontal lines
    for (let v = -0.25; v <= 0.75; v += 0.25) {
      const sy = toSvgY(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', margin.left); line.setAttribute('y1', sy);
      line.setAttribute('x2', W - margin.right); line.setAttribute('y2', sy);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
  }

  // Draw axes
  function drawAxes() {
    axesG.innerHTML = '';

    // X-axis (at y=0)
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin.left); xAxis.setAttribute('y1', toSvgY(0));
    xAxis.setAttribute('x2', W - margin.right); xAxis.setAttribute('y2', toSvgY(0));
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(xAxis);

    // Y-axis (at x=0)
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', toSvgX(0)); yAxis.setAttribute('y1', margin.top);
    yAxis.setAttribute('x2', toSvgX(0)); yAxis.setAttribute('y2', H - margin.bottom);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(yAxis);

    // X-axis labels
    for (let v = 0; v <= 1; v += 0.5) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', toSvgX(v)); label.setAttribute('y', H - margin.bottom + 20);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '12');
      label.setAttribute('text-anchor', 'middle');
      label.textContent = v.toFixed(1);
      axesG.appendChild(label);
    }

    // Y-axis labels
    for (let v = 0; v <= 0.5; v += 0.25) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - 10); label.setAttribute('y', toSvgY(v) + 4);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '12');
      label.setAttribute('text-anchor', 'end');
      label.textContent = v.toFixed(2);
      axesG.appendChild(label);
    }

    // Axis titles
    const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xLabel.setAttribute('x', W / 2); xLabel.setAttribute('y', H - 5);
    xLabel.setAttribute('fill', '#8b949e'); xLabel.setAttribute('font-size', '13');
    xLabel.setAttribute('text-anchor', 'middle');
    xLabel.textContent = 'x';
    axesG.appendChild(xLabel);

    const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yLabel.setAttribute('x', 8); yLabel.setAttribute('y', H / 2);
    yLabel.setAttribute('fill', '#8b949e'); yLabel.setAttribute('font-size', '13');
    yLabel.setAttribute('text-anchor', 'middle');
    yLabel.setAttribute('transform', `rotate(-90, 8, ${H/2})`);
    yLabel.textContent = 'y';
    axesG.appendChild(yLabel);
  }

  // Draw sample points
  function drawSamplePoints(xPts, yPts) {
    samplePointsG.innerHTML = '';
    for (let i = 0; i < xPts.length; i++) {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', toSvgX(xPts[i]));
      circle.setAttribute('cy', toSvgY(yPts[i]));
      circle.setAttribute('r', 5);
      circle.setAttribute('fill', '#7ee787');
      circle.setAttribute('stroke', '#0d1117');
      circle.setAttribute('stroke-width', 1.5);
      samplePointsG.appendChild(circle);
    }
  }

  // Main update function
  function update() {
    const h = parseFloat(document.querySelector('[data-param="h"]').value);
    const N = parseInt(document.querySelector('[data-param="n"]').value);

    // Update readouts
    document.querySelector('[data-readout="h"]').textContent = h.toFixed(2);
    document.querySelector('[data-readout="n"]').textContent = N;

    // Generate x values for plotting
    const xs = linspace(-0.2, 1.0, 200);

    // True log(1+x)
    const ysTrue = xs.map(log1p);

    // Sample points for Newton-Gregory
    const x0 = 0;
    const nPoints = N + 1;  // N+1 points for degree N polynomial
    const xPts = Array.from({length: nPoints}, (_, i) => x0 + i * h);
    const yPts = xPts.map(log1p);

    // Newton-Gregory interpolation
    const ysNG = newtonGregory(xs, x0, yPts, h);

    // Taylor approximation (degree N)
    const ysTaylor = taylorLog1p(xs, N);

    // Update paths
    truePath.setAttribute('d', toPath(xs, ysTrue));
    taylorPath.setAttribute('d', toPath(xs, ysTaylor));
    interpPath.setAttribute('d', toPath(xs, ysNG));

    // Draw sample points
    drawSamplePoints(xPts, yPts);

    // Update info
    infoDiv.innerHTML = `N = ${N} (degree ${N} polynomial) &nbsp;|&nbsp; h = ${h.toFixed(2)} &nbsp;|&nbsp; ${nPoints} sample points`;
  }

  // Initialize
  drawGrid();
  drawAxes();
  update();

  // Event listeners for sliders
  document.querySelectorAll('.ng-widget input[type="range"]').forEach(input => {
    input.addEventListener('input', update);
  });

  // Reset button
  document.getElementById('ng-reset').addEventListener('click', () => {
    document.querySelector('[data-param="h"]').value = 0.10;
    document.querySelector('[data-param="n"]').value = 3;
    update();
  });
})();
</script>