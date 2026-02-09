---
title: "Least Absolute Deviation"
categories:
  - Optimization
date: 2026-01-09 19:00:00 +0000
mathjax: true
tags:
  - Optimization
toc: true
classes: wide
excerpt: ""
---

The most common method of fitting a linear model to data is ordinary least squares.
In ordinary least squares we want to solve the following optimization problem

$$
\begin{equation}
\underset{\beta}{\text{minimize}} \quad \sum_{i=1}^N (\beta^T x_i - y_i)^2
\end{equation}.
$$

Oftentimes this will be written in matrix form as

$$
\begin{equation}
\underset{\beta}{\text{minimize}} \quad ||X\beta - y||_2^2
\end{equation}
$$

where $$X\in \mathbf{R}^{N\times d}$$ and $$y\in \mathbf{R}^{d}$$.
In this form, least squares has a particularly nice closed form solution<sup>[1](#footnote1)</sup>

$$\beta^\star = (X^TX)^{-1}X^Ty.$$

However, least squares is strongly affected by outlier data points.
A more robust fitting procedure is _least absolute deviations_ where we instead solve the optimization problem

$$
\begin{equation}
\underset{\beta}{\text{minimize}} \quad \sum_{i=1}^N |\beta^T x_i - y_i|
\end{equation}.
$$

Or written in matrix form

$$
\begin{equation}
\underset{\beta}{\text{minimize}} \quad ||X\beta - y||_1
\end{equation}.
$$

However, unlike least squares, there is no succinct closed form solution.
The next section will introduce a heuristic algorithm for solving this problem

# Coordinate Descent

Since the objective is the composition of an affine function with the absolute value function (which is convex), the least absolute deviation objective is convex.
This means any local minimum we find will be a global minimum!

In typical applications, the $$\beta$$ we're solving for can be very high dimensional and hard to visualize.
What if we could instead reduce the multi-dimensional optimization down to a series of 1D optimization problems?

This is the idea of coordinate descent.
Instead of optimizing over all $$\beta_1,\ldots,\beta_d$$ jointly, we cycle through each variable holding the other fixed and optimize over just one variable

$$
\begin{equation}
\underset{\beta_k}{\text{minimize}} \quad \sum_{i=1}^N |\beta_k x_i - r_i|
\end{equation}
$$

where $$r_i = y_i - \sum_{j\ne k} \beta_j x_{ij}$$

# Golden-Section

<div class="widget-container" id="golden-ratio-widget">
  <div class="widget-controls">
    <button type="button" class="widget-button" id="golden-ratio-reset">Reset</button>
  </div>
  <svg class="widget-plot" id="golden-ratio-svg" viewBox="0 0 600 150" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="600" height="150" fill="#0d1117"></rect>
    <!-- Bar background -->
    <rect id="gr-bar-bg" x="80" y="50" width="440" height="30" fill="#1c2128" rx="2"></rect>
    <!-- Left segment -->
    <rect id="gr-left-segment" x="80" y="50" width="220" height="30" fill="#58a6ff" rx="2"></rect>
    <!-- Right segment -->
    <rect id="gr-right-segment" x="300" y="50" width="220" height="30" fill="#3fb950" rx="2"></rect>
    <!-- Draggable point -->
    <circle id="gr-point" cx="300" cy="65" r="8" fill="#58a6ff" cursor="pointer"></circle>
    <!-- Ratio labels -->
    <g id="gr-labels"></g>
  </svg>
  <div class="widget-info" id="golden-ratio-info">
    <div class="widget-info-row">
      <span class="widget-info-label">Whole / Longer:</span>
      <span class="widget-info-value" id="gr-ratio1">1.000</span>
    </div>
    <div class="widget-info-row">
      <span class="widget-info-label">Longer / Shorter:</span>
      <span class="widget-info-value" id="gr-ratio2">∞</span>
    </div>
  </div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure X: Interactive golden ratio demonstration. Drag the middle point to adjust the position. The ratios highlight in gold when within 5% of φ ≈ 1.618.</figcaption>
</div>

Now that we have a simple 1D problem, we can graph an example to see what the objective might look like.
In Figure 1 we see an objective with just one term

<figure>
    <a href="/assets/2026/golden-section/images/1d-abs-deviation-00.png"><img src="/assets/2026/golden-section/images/1d-abs-deviation-00.png"></a>
    <figcaption>Figure 1</figcaption>
</figure>

Intuitively,

- maybe just focus on the golden section algorithm??
- setup the least absolute deviation problem and contrast it with OLS
- start with one variable case of just the slope
- introduce the golden section method
- propose alternate method just using the kinks
- frame as an iterative weighted least squares problem
- multi-variable case with coordinate descent

## Footnotes

<a name="footnote1">1</a>: This formula only holds if $$X^TX$$ is invertible. More specifically, when $$X$$ is skinny (i.e. $$N>d$$) and full rank (i.e. $$\mathbf{rank}(X)=d$$)
