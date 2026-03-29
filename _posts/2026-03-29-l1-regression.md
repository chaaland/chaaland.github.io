---
title: "Robust Regression Without Gradients"
categories:
  - Optimization
date: 2026-03-29 00:00:00 +0000
mathjax: true
tags:
  - Optimization
  - Regression
  - Machine Learning
toc: true
classes: wide
excerpt: "L1 regression is more robust to outliers than least squares, but harder to solve. We walk through four algorithms, each addressing a limitation of the previous one."
---

## The Problem: Fitting a Line with the Right Loss

When fitting a line to noisy data $$(x_1, y_1),\ldots, (x_N, y_N)$$, the most common approach is to use ordinary least squares (OLS).
The goal is to find a slope $$\beta_1$$ and intercept $$\beta_0$$ that minimizes the average squared distance between the line and the data.
Stated mathematically, it is the optimization problem 

$$
\underset{\beta_1,\beta_0}{\text{minimize}}\quad \frac{1}{N} \sum_{i=1}^N (\beta_1 x_i + \beta_0 - y_i)^2
$$

Figure 1 shows some example data along with a contour plot of the objective function.

<style>
#ols-widget { max-width: 100%; }
.ols-panels { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }
.ols-panel { flex: 1; min-width: 280px; }
.ols-panel-title { text-align: center; font-size: 0.85rem; color: #c9d1d9; margin-top: 6px; font-style: italic; }
@media (max-width: 600px) { .ols-panels { flex-direction: column; align-items: center; } }
</style>

<div class="widget-container" id="ols-widget">
  <div class="widget-controls">
    <label>
      β₁
      <input type="range" min="-0.5" max="3.5" value="0.5" step="0.05" id="ols-slope-slider">
      <span class="widget-readout" id="ols-slope-readout">0.50</span>
    </label>
    <label>
      β₀
      <input type="range" min="-4" max="14" value="10" step="0.25" id="ols-intercept-slider">
      <span class="widget-readout" id="ols-intercept-readout">10.00</span>
    </label>
    <button type="button" class="widget-button" id="ols-minimize-btn">Snap to minimum</button>
  </div>
  <div class="ols-panels">
    <div class="ols-panel">
      <svg class="widget-plot" id="ols-scatter-svg" viewBox="0 0 320 280" preserveAspectRatio="xMidYMid meet">
        <defs>
          <clipPath id="ols-scatter-clip">
            <rect x="50" y="25" width="250" height="215"></rect>
          </clipPath>
        </defs>
        <rect x="0" y="0" width="320" height="280" fill="#0d1117"></rect>
        <g id="ols-scatter-line" clip-path="url(#ols-scatter-clip)"></g>
        <g id="ols-scatter-points" clip-path="url(#ols-scatter-clip)"></g>
        <g id="ols-scatter-axes"></g>
      </svg>
      <div class="ols-panel-title">Data with outlier</div>
    </div>
    <div class="ols-panel">
      <svg class="widget-plot" id="ols-contour-svg" viewBox="0 0 320 280" preserveAspectRatio="xMidYMid meet">
        <defs>
          <clipPath id="ols-contour-clip">
            <rect x="50" y="25" width="250" height="215"></rect>
          </clipPath>
        </defs>
        <rect x="0" y="0" width="320" height="280" fill="#0d1117"></rect>
        <g id="ols-contour-heatmap" clip-path="url(#ols-contour-clip)"></g>
        <g id="ols-contour-marker" clip-path="url(#ols-contour-clip)"></g>
        <g id="ols-contour-axes"></g>
      </svg>
      <div class="ols-panel-title">OLS loss landscape</div>
    </div>
  </div>
  <div class="widget-info" id="ols-info">β₁ = 0.50 | β₀ = 10.00 | OLS loss = —</div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 1: OLS loss landscape in (β₀, β₁) space. Darker regions indicate lower loss.</figcaption>
</div>

OLS is nice because the objective is differentiable making it amenable to gradient based optimization methods.
OLS also has a nice closed form solution in the form of the normal equations.<sup>[1](#footnote1)</sup>

One drawback is that it is very sensitive to outliers.
This is because when the difference between the predicted value and the observed value is large, as in the case of an outlier, squaring this difference makes it even larger, leading to an overemphasis on extreme points.

An alternative approach to least squares is least absolute deviation (LAD), which is given mathematically by 

$$
\underset{\beta_1, \beta_0}{\text{minimize}}\quad \frac{1}{N} \sum_{i=1}^N |\beta_1 x_i + \beta_0 - y_i|.
$$

Figure 2 shows the same data points with the loss landscape of the LAD objective.

<style>
#lad-widget { max-width: 100%; }
.lad-panels { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }
.lad-panel { flex: 1; min-width: 280px; }
.lad-panel-title { text-align: center; font-size: 0.85rem; color: #c9d1d9; margin-top: 6px; font-style: italic; }
@media (max-width: 600px) { .lad-panels { flex-direction: column; align-items: center; } }
</style>

<div class="widget-container" id="lad-widget">
  <div class="widget-controls">
    <label>
      β₁
      <input type="range" min="-0.5" max="3.5" value="0.5" step="0.05" id="lad-slope-slider">
      <span class="widget-readout" id="lad-slope-readout">0.50</span>
    </label>
    <label>
      β₀
      <input type="range" min="-4" max="14" value="10" step="0.25" id="lad-intercept-slider">
      <span class="widget-readout" id="lad-intercept-readout">10.00</span>
    </label>
    <button type="button" class="widget-button" id="lad-minimize-btn">Snap to minimum</button>
  </div>
  <div class="lad-panels">
    <div class="lad-panel">
      <svg class="widget-plot" id="lad-scatter-svg" viewBox="0 0 320 280" preserveAspectRatio="xMidYMid meet">
        <defs>
          <clipPath id="lad-scatter-clip">
            <rect x="50" y="25" width="250" height="215"></rect>
          </clipPath>
        </defs>
        <rect x="0" y="0" width="320" height="280" fill="#0d1117"></rect>
        <g id="lad-scatter-line" clip-path="url(#lad-scatter-clip)"></g>
        <g id="lad-scatter-points" clip-path="url(#lad-scatter-clip)"></g>
        <g id="lad-scatter-axes"></g>
      </svg>
      <div class="lad-panel-title">Data with outlier</div>
    </div>
    <div class="lad-panel">
      <svg class="widget-plot" id="lad-contour-svg" viewBox="0 0 320 280" preserveAspectRatio="xMidYMid meet">
        <defs>
          <clipPath id="lad-contour-clip">
            <rect x="50" y="25" width="250" height="215"></rect>
          </clipPath>
        </defs>
        <rect x="0" y="0" width="320" height="280" fill="#0d1117"></rect>
        <g id="lad-contour-heatmap" clip-path="url(#lad-contour-clip)"></g>
        <g id="lad-contour-marker" clip-path="url(#lad-contour-clip)"></g>
        <g id="lad-contour-axes"></g>
      </svg>
      <div class="lad-panel-title">LAD loss landscape</div>
    </div>
  </div>
  <div class="widget-info" id="lad-info">β₁ = 0.50 | β₀ = 10.00 | LAD loss = —</div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 2: LAD loss landscape in (β₀, β₁) space. Darker regions indicate lower loss.</figcaption>
</div>

The LAD objective has the desirable property of being robust to outliers since large errors are not over-penalized.
However, it also lacks a closed form solution unlike OLS.
Furthermore, the lack of smoothness makes gradient based methods more subtle.
In this post, we'll see four algorithms for solving the least absolute deviation problem with each one addressing a concrete limitation of the previous.

## Method 1: Coordinate Descent with Golden Section Search

Method 1 is coordinate descent: we alternate between optimizing over slope $$\beta_1$$ with intercept $$\beta_0$$ held fixed, and optimizing over $$\beta_0$$ with $$\beta_1$$ held fixed.
Each 1D subproblem has an efficient solution — golden section search for the slope update and the sample median for the intercept update.

To see why, suppose we already know the optimal $$\beta_0^\star$$.
Our problem then reduces to the 1D

$$
\underset{\beta_1}{\text{minimize}}\quad \frac{1}{N} \sum_{i=1}^N |\beta_1 x_i + (\beta_0^\star - y_i)|.
$$

It's easy to see that the objective is not differentiable at $$\beta_1 = (y_i - \beta_0^\star) / x_i,\ i=1,\ldots,N$$.
Plotting an example objective function, we can clearly see the $$N$$ non-differentiable points

<figure>
  <a href="/assets/2026/l1-regression/images/1d-abs-deviation-06.png">
    <img src="/assets/2026/l1-regression/images/1d-abs-deviation-06.png" alt="1D absolute deviation objective showing N non-differentiable kink points">
  </a>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 3: 1D absolute deviation objective as a function of slope, with kinks at each data point.</figcaption>
</figure>

Recall that in binary search, we have three points or indices $$\ell_1 < \ell_2 < \ell_3$$.
We then determine whether the solution lies in $$[\ell_1, \ell_2]$$ or in $$[\ell_2, \ell_3]$$ and update the three indices accordingly to lie within the new interval.

The golden section algorithm is similar in spirit.
Instead of using three points to divide the interval into disjoint subsets, golden section search uses 4 points $$\ell_1 < \ell_2 < \ell_3 < \ell_4$$ to divide the solution interval into two _overlapping_ subsets $$[\ell_1, \ell_3]$$ or $$[\ell_2, \ell_4]$$ and determining in which the solution must lie.

The golden section method does this in a particularly clever way that avoids extra evaluations of the objective.
See [Golden Section Search for Robust Regression](/optimization/golden-section/) for a full derivation and implementation. 

However, we started by assuming we knew $$\beta_0^\star$$, but in practice, this is a value we need to compute.
A rough argument is that using $${d \over dx} \lvert x\rvert = \mathbf{sign}(x)$$<sup>[2](#footnote2)</sup> and denoting $$r_i = y_i - \beta_1 x_i$$, leads to the derivative of the objective

$$
\frac{dL}{d\beta_0}  = \frac{1}{N}\sum_{i=1}^N \mathbf{sign}(\beta_0 - r_i).
$$

Setting this to zero, we see that the terms in the summation need to cancel each other out.
This occurs when there are an equal number of residuals greater or equal to $$\beta_0^\star$$ as there are less or equal to $$\beta_0^\star$$.
This happens exactly when $$\beta_0^\star$$ is a median of $$r_1,\ldots,r_N$$.
For odd $$N$$, this value is unique and for even $$N$$ it can be any value between the two middle values when sorted as shown in Figure 4.

<figure>
  <a href="/assets/2026/l1-regression/images/bias-objective.png">
    <img src="/assets/2026/l1-regression/images/bias-objective.png" alt="1D LAD objective as a function of bias, showing a piecewise-linear curve minimized at the median of the residuals">
  </a>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 4: LAD objective as a function of bias for fixed slope. The piecewise-linear curve is minimized at any median of the residuals.</figcaption>
</figure>

Figure 5 illustrates coordinate descent trajectories using the above methods in the inner optimization loop.

<figure>
  <img src="/assets/2026/l1-regression/images/cd-trajectory.png" alt="Coordinate descent trajectories on LAD problem" style="max-width: 100%;">
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 5: Coordinate descent trajectories on a simple least absolute deviation problem. Trajectories can stall at a non-optimal point when no axis-aligned descent direction exists, even though a diagonal descent direction does.</figcaption>
</figure>

An important thing to note about coordinate descent is that it does not necessarily lead to the global minimum, even for convex problems!
When the objective is not differentiable, coordinate descent can get "stuck" since it's only optimizing one variable at a time.
Looking at the convergence points in Figure 5, we can see that moving horizontally or vertically are not descent directions yet there _is_ a descent direction when moving in both coordinates simultaneously.

### Multidimensional LAD

The more general form of linear regression fits a hyperplane to the data.
The objective function is

$$
\underset{\beta, \beta_0}{\text{minimize}}\quad \frac{1}{N} \sum_{i=1}^N |\beta^T x^{(i)} + \beta_0 - y^{(i)}|,
$$

where $$\beta\in \mathbf{R}^d$$ collects all slope parameters (the scalar $$\beta_1$$ from the 2D case becomes the first component of $$\beta$$).
We can turn this problem into a series of 1D problems via coordinate descent as well.
We loop over each parameter: apply golden section when minimizing over any $$\beta_k$$, and compute the median residual when minimizing over $$\beta_0$$.

## Method 2: Knot Scan

In the inner loop of coordinate descent we relied on golden section but looking at the objective in Figure 3, it should be clear that a minimum will always exist at a kink.

This leads to an even simpler algorithm than golden section!
Simply enumerate all of the non-differentiable points $$z_i = r_i / x^{(i)}_k, i=1,\ldots,N$$ and evaluate the objective function at each.
This method has the distinct advantage of being exact as opposed to golden section search which can get arbitrarily close to the minimum without reaching it.

The following code implements this algorithm, taking the feature column $$x_k$$ and partial residuals $$r$$ as inputs and returning the optimal $$\beta_k$$.

<details>
<summary>
Click for code
</summary>

{% highlight python %}
def knot_scan_1d(x_k, r):
    # x_k: (N,)  feature values for coordinate k across all samples
    # r:   (N,)  partial residuals y - sum_{j != k} beta_j * x_j - b
    nz = x_k != 0          # (N,) boolean mask; skip flat terms
    z = r[nz] / x_k[nz]   # (M,) knot locations, M <= N
    best_val = np.inf
    beta_k = z[0]
    for z_i in z:
        val = np.mean(np.abs(x_k * z_i - r))
        if val < best_val:
            best_val = val
            beta_k = z_i

    return beta_k
{% endhighlight %}

</details>
<br>

However, since the cost of enumerating the knots is $$\mathcal{O}(N)$$ and the cost of evaluating the objective at each knot is $$\mathcal{O}(N)$$, the entire algorithm $$O(N^2)$$.
For very large datasets with hundreds of millions of points, this can be costly.

## Method 3: Weighted Median

To come up with an improved method consider rewriting the 1D optimization problem as

$$
\underset{\beta_k}{\text{minimize}}\quad \frac{1}{N} \sum_{i=1}^N \lvert x^{(i)}_k \rvert \cdot \lvert \beta_k  - r_i / x^{(i)}_k \rvert.
$$

This problem is equivalent to

$$
\underset{\beta_k}{\text{minimize}}\quad \frac{1}{N} \sum_{i=1}^N w_i \cdot \lvert \beta_k  - c_i \rvert
$$

which is just a _weighted_ version of the optimization solving for the bias.
Using the same argument as we used for the bias, the solution is the weighted median!
We can enumerate the kinks in $$\mathcal{O}(N)$$ just as before and then sort them in $$\mathcal{O}(N \log N)$$ giving us a much better runtime than $$\mathcal{O}(N^2)$$.

The following code implements this, replacing the linear scan over knots with a single sort and cumulative weight traversal.

<details>
<summary>
Click for code
</summary>

{% highlight python %}
def weighted_median_1d(x_k, r):
    # x_k: (N,)  feature values for coordinate k across all samples
    # r:   (N,)  partial residuals y - sum_{j != k} beta_j * x_j - b
    nz = x_k != 0
    z = r[nz] / x_k[nz]        # (M,) knot locations, M <= N
    w = np.abs(x_k[nz])        # (M,) weights

    order = np.argsort(z)
    z_sorted = z[order]         # (M,) knots in ascending order
    w_sorted = w[order]         # (M,) corresponding weights

    cumulative = np.cumsum(w_sorted)         # (M,)
    half = w_sorted.sum() / 2.0              # scalar threshold
    idx = np.searchsorted(cumulative, half)
    return z_sorted[min(idx, len(z_sorted) - 1)]  # scalar optimal beta_k
{% endhighlight %}

</details>
<br>

The inherent limitation is the outer loop of coordinate descent.
We're only able to make progress by moving on one axis of coordinate space.
In practical applications there may be faster paths to the minimum if we admit more trajectories than just the piecewise constant ones.

## Method 4: IRLS

Returning to the LAD objective,

$$
\underset{\beta, \beta_0}{\text{minimize}}\quad \frac{1}{N} \sum_{i=1}^N \lvert \beta^T x^{(i)} + \beta_0 - y^{(i)}\rvert,
$$

we can perform a clever rewrite by dividing and multiplying each term in the summation by $$ \lvert  \beta^T x^{(i)} + \beta_0 - y^{(i)}\rvert$$  

$$
\underset{\beta, \beta_0}{\text{minimize}}\quad \frac{1}{N} \sum_{i=1}^N \frac{1}{\lvert  \beta^T x^{(i)} + \beta_0 - y^{(i)}\rvert} \cdot (\beta^T x^{(i)} + \beta_0  - y^{(i)})^2.
$$

If we treat the absolute value term as a constant (i.e. not dependent on $$\beta$$ and $$\beta_0$$) we would have

$$
\underset{\beta, \beta_0}{\text{minimize}}\quad \frac{1}{N} \sum_{i=1}^N w_i \cdot (\beta^T x^{(i)} + \beta_0 - y^{(i)})^2
$$

which is simply a _weighted_ least squares problem and like OLS, also has a simple closed form solution.<sup>[3](#footnote3)</sup>
Note that the weights are inversely proportional to the error.
This means samples with large absolute deviation are given _less_ weight.

However, since our weights actually _do_ depend on the parameters we're optimizing over, we need to use an iterative approach.

1. Initialize parameters
2. Compute $$w_1,\ldots,w_N$$
3. Solve the associated weighted least squares problem
4. If not converged, go back to step 2

This algorithm is called Iteratively Reweighted Least Squares (IRLS).
Its implementation is surprisingly compact as shown in the code below.

<details>
<summary>
Click for code
</summary>

{% highlight python %}
def irls_lad(X, y, n_iters=50, eps=1e-6):
    n, d = X.shape
    X_aug = np.column_stack([X, np.ones(n)])

    beta = np.zeros(d + 1)

    for _ in range(n_iters):
        residuals = y - X_aug @ beta
        weights = 1.0 / np.maximum(np.abs(residuals), eps)

        Xw = X_aug * weights[:, np.newaxis]
        beta = np.linalg.solve(Xw.T @ X_aug, Xw.T @ y)

    return beta[:d], beta[d]
{% endhighlight %}

</details>
<br>

Figure 6 shows the IRLS trajectory on the same LAD minimization problem as before.

<figure>
  <a href="/assets/2026/l1-regression/images/irls-trajectory.png">
    <img src="/assets/2026/l1-regression/images/irls-trajectory.png" alt="IRLS parameter trajectory converging to the LAD optimum">
  </a>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 6: IRLS trajectory on the LAD loss landscape. Parameters converge in a small number of iterations without the axis-aligned staircase pattern of coordinate descent.</figcaption>
</figure>

We can see that the loss function decreases rapidly after just the first iteration which demonstrates the value of being able to take steps in non-axis aligned directions.
Figure 7 compares the convergence of coordinate descent and IRLS side by side.

<figure>
  <a href="/assets/2026/l1-regression/images/cd-vs-irls.png">
    <img src="/assets/2026/l1-regression/images/cd-vs-irls.png" alt="Side-by-side comparison of coordinate descent and IRLS convergence on the LAD problem">
  </a>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 7: Coordinate descent vs IRLS convergence on the same LAD problem. IRLS reaches the optimum in far fewer iterations by updating all parameters simultaneously.</figcaption>
</figure>

There are two things worth noting about this graph
1. IRLS converges extremely quickly compared to coordinate descent
2. IRLS converges to the true minimum. Coordinate descent converged to a point where it could not make any progress in an axis-aligned direction and stalled before reaching the minimum.

## Summary

In this post we've seen 4 different methods for solving the least absolute deviation minimization problem.

The clear winner for speed of convergence is Iteratively Reweighted Least Squares.
However, in cases where you don't care as much about speed and just need something that works (e.g. you don't have access to/don't want to take a dependence on a linear algebra package), coordinate descent with weighted median is a simple, easy to implement second choice.

## Footnotes

<a name="footnote1">1</a>: In the general form, OLS minimizes $$\|Ax - b\|^2$$ over $$x \in \mathbf{R}^d$$, where $$A \in \mathbf{R}^{N \times d}$$ is the design matrix and $$b \in \mathbf{R}^N$$ is the target vector. Setting the gradient $$2A^T(Ax - b) = 0$$ gives the normal equations $$A^TAx = A^Tb$$, with closed-form solution $$x = (A^TA)^{-1}A^Tb$$ whenever $$A^TA$$ is invertible.

<a name="footnote2">2</a>: Strictly, $$\lvert x \rvert$$ is not differentiable at $$x = 0$$. The identity $${d \over dx}\lvert x\rvert = \mathbf{sign}(x)$$ uses the convention $$\mathbf{sign}(0) = 0$$, but the true derivative is undefined there. The correct generalization is the subgradient: $$\partial \lvert x \rvert = \mathbf{sign}(x)$$ for $$x \neq 0$$, and $$\partial \lvert x \rvert = [-1, 1]$$ at $$x = 0$$.

<a name="footnote3">3</a>: Weighted least squares minimizes $$\|W^{1/2}(Ax - b)\|^2 = (Ax-b)^T W (Ax-b)$$ where $$W = \mathbf{diag}(w_1,\ldots,w_N)$$. Setting the gradient to zero gives $$A^TWAx = A^TWb$$, with closed-form solution $$x = (A^TWA)^{-1}A^TWb$$.

{% include widget-scripts.html %}
<script>
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';

  // Same dataset as OLS widget
  var xs = [0.000, 1.667, 3.333, 5.000, 6.667, 8.333, 10.000];
  var ys = [2.745, 4.293, 7.972, 19.785, 14.369, 15.651, 16.296];

  function ladLoss(m, b) {
    var s = 0;
    for (var i = 0; i < xs.length; i++) { s += Math.abs(m * xs[i] + b - ys[i]); }
    return s / xs.length;
  }

  function el(tag, attrs) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs.hasOwnProperty(k)) e.setAttribute(k, attrs[k]);
    return e;
  }
  function txt(content, attrs) { var e = el('text', attrs); e.textContent = content; return e; }

  // Viridis colormap (5 key stops)
  var VIR = [[68,1,84],[59,82,139],[33,144,140],[93,201,99],[253,231,37]];
  function viridis(t) {
    t = Math.max(0, Math.min(1, t));
    var s = t * 4, lo = Math.floor(s), hi = Math.min(lo + 1, 4), f = s - lo;
    return 'rgb(' +
      Math.round(VIR[lo][0] + f * (VIR[hi][0] - VIR[lo][0])) + ',' +
      Math.round(VIR[lo][1] + f * (VIR[hi][1] - VIR[lo][1])) + ',' +
      Math.round(VIR[lo][2] + f * (VIR[hi][2] - VIR[lo][2])) + ')';
  }

  // Shared layout constants
  var W = 320, H = 280;
  var PD = { t: 25, r: 20, b: 40, l: 50 };
  var PW = W - PD.l - PD.r;   // 250
  var PH = H - PD.t - PD.b;   // 215

  // Scatter coordinate domain
  var SX0 = -0.5, SX1 = 10.5, SY0 = -2, SY1 = 23;
  function sx(x) { return PD.l + (x - SX0) / (SX1 - SX0) * PW; }
  function sy(y) { return PD.t + (SY1 - y) / (SY1 - SY0) * PH; }

  // Contour domain: intercept on x-axis, slope on y-axis
  var M0 = -0.5, M1 = 3.5, B0 = -4.0, B1 = 14.0;
  function cx(b) { return PD.l + (b - B0) / (B1 - B0) * PW; }
  function cy(m) { return PD.t + (M1 - m) / (M1 - M0) * PH; }

  // Pre-compute 120x120 heatmap losses once
  var GN = 120, cw = PW / GN, ch = PH / GN;
  var losses = new Array(GN * GN);
  var lMin = Infinity;
  for (var gi = 0; gi < GN; gi++) {
    for (var gj = 0; gj < GN; gj++) {
      var lv = ladLoss(M0 + (gj + 0.5) / GN * (M1 - M0), B0 + (gi + 0.5) / GN * (B1 - B0));
      losses[gi * GN + gj] = lv;
      if (lv < lMin) lMin = lv;
    }
  }
  var lRange = 8; // colormap spans 8 loss units above minimum

  function drawHeatmap() {
    var g = document.getElementById('lad-contour-heatmap');
    for (var gi = 0; gi < GN; gi++) {
      for (var gj = 0; gj < GN; gj++) {
        g.appendChild(el('rect', {
          x: PD.l + gi * cw,
          y: PD.t + (GN - 1 - gj) * ch,
          width: cw + 0.5, height: ch + 0.5,
          fill: viridis(Math.min(1, (losses[gi * GN + gj] - lMin) / lRange))
        }));
      }
    }
  }

  function drawAxes(gId, xVals, xFmt, xToSvg, yVals, yFmt, yToSvg, xLabel, yLabel) {
    var g = document.getElementById(gId); g.innerHTML = '';
    g.appendChild(el('line', { x1: PD.l, y1: H - PD.b, x2: W - PD.r, y2: H - PD.b, stroke: '#8b949e', 'stroke-width': 1 }));
    g.appendChild(el('line', { x1: PD.l, y1: PD.t,     x2: PD.l,     y2: H - PD.b, stroke: '#8b949e', 'stroke-width': 1 }));
    xVals.forEach(function (v) {
      var px = xToSvg(v);
      g.appendChild(el('line', { x1: px, y1: H - PD.b, x2: px, y2: H - PD.b + 4, stroke: '#8b949e', 'stroke-width': 1 }));
      g.appendChild(txt(xFmt(v), { x: px, y: H - PD.b + 15, 'text-anchor': 'middle', fill: '#8b949e', 'font-size': 10, 'font-family': 'monospace' }));
    });
    yVals.forEach(function (v) {
      var py = yToSvg(v);
      g.appendChild(el('line', { x1: PD.l - 4, y1: py, x2: PD.l, y2: py, stroke: '#8b949e', 'stroke-width': 1 }));
      g.appendChild(txt(yFmt(v), { x: PD.l - 6, y: py + 4, 'text-anchor': 'end', fill: '#8b949e', 'font-size': 10, 'font-family': 'monospace' }));
    });
    g.appendChild(txt(xLabel, { x: PD.l + PW / 2, y: H - 5, 'text-anchor': 'middle', fill: '#8b949e', 'font-size': 11, 'font-family': 'monospace' }));
    var mid = PD.t + PH / 2;
    g.appendChild(txt(yLabel, { x: 12, y: mid, 'text-anchor': 'middle', fill: '#8b949e', 'font-size': 11, 'font-family': 'monospace', transform: 'rotate(-90,12,' + mid + ')' }));
  }

  function drawScatterAxes() {
    drawAxes('lad-scatter-axes',
      [0, 2, 4, 6, 8, 10], function (v) { return v; }, sx,
      [0, 5, 10, 15, 20],  function (v) { return v; }, sy,
      'x', 'y');
  }

  function drawContourAxes() {
    drawAxes('lad-contour-axes',
      [0, 4, 8, 12], function (v) { return v; }, cx,
      [0, 1, 2, 3], function (v) { return v; }, cy,
      'β₀', 'β₁');
  }

  function drawPoints() {
    var g = document.getElementById('lad-scatter-points'); g.innerHTML = '';
    for (var i = 0; i < xs.length; i++) {
      g.appendChild(el('circle', { cx: sx(xs[i]), cy: sy(ys[i]), r: 4, fill: '#c9d1d9', stroke: '#0d1117', 'stroke-width': 1 }));
    }
  }

  function updateLine(m, b) {
    var g = document.getElementById('lad-scatter-line'); g.innerHTML = '';
    g.appendChild(el('line', { x1: sx(SX0), y1: sy(m * SX0 + b), x2: sx(SX1), y2: sy(m * SX1 + b), stroke: '#58a6ff', 'stroke-width': 2 }));
  }

  function updateMarker(m, b) {
    var g = document.getElementById('lad-contour-marker'); g.innerHTML = '';
    var px = cx(b), py = cy(m), r = 6;
    g.appendChild(el('line', { x1: px - r, y1: py,     x2: px + r, y2: py,     stroke: 'white', 'stroke-width': 1.5 }));
    g.appendChild(el('line', { x1: px,     y1: py - r, x2: px,     y2: py + r, stroke: 'white', 'stroke-width': 1.5 }));
    g.appendChild(el('circle', { cx: px, cy: py, r: r, fill: 'none', stroke: 'white', 'stroke-width': 1.5 }));
  }

  function render() {
    var m = parseFloat(document.getElementById('lad-slope-slider').value);
    var b = parseFloat(document.getElementById('lad-intercept-slider').value);
    document.getElementById('lad-slope-readout').textContent = m.toFixed(2);
    document.getElementById('lad-intercept-readout').textContent = b.toFixed(2);
    updateLine(m, b);
    updateMarker(m, b);
    document.getElementById('lad-info').textContent =
      'β₁ = ' + m.toFixed(2) + ' | β₀ = ' + b.toFixed(2) + ' | LAD loss = ' + ladLoss(m, b).toFixed(2);
  }

  drawHeatmap();
  drawScatterAxes();
  drawContourAxes();
  drawPoints();
  render();

  document.getElementById('lad-slope-slider').addEventListener('input', render);
  document.getElementById('lad-intercept-slider').addEventListener('input', render);

  // Find argmin in precomputed grid
  var lMinIdx = 0;
  for (var k = 1; k < GN * GN; k++) { if (losses[k] < losses[lMinIdx]) lMinIdx = k; }
  var mLAD = M0 + (Math.floor(lMinIdx / GN) + 0.5) / GN * (M1 - M0);
  var bLAD = B0 + (lMinIdx % GN + 0.5) / GN * (B1 - B0);

  document.getElementById('lad-minimize-btn').addEventListener('click', function () {
    document.getElementById('lad-slope-slider').value = mLAD;
    document.getElementById('lad-intercept-slider').value = bLAD;
    render();
  });
})();

(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';

  // Dataset: numpy seed 42, x=linspace(0,10,7), y=1.5x+2+N(0,1.5), y[3]+=8
  var xs = [0.000, 1.667, 3.333, 5.000, 6.667, 8.333, 10.000];
  var ys = [2.745, 4.293, 7.972, 19.785, 14.369, 15.651, 16.296];

  function olsLoss(m, b) {
    var s = 0;
    for (var i = 0; i < xs.length; i++) { var r = m * xs[i] + b - ys[i]; s += r * r; }
    return s / xs.length;
  }

  function el(tag, attrs) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs.hasOwnProperty(k)) e.setAttribute(k, attrs[k]);
    return e;
  }
  function txt(content, attrs) { var e = el('text', attrs); e.textContent = content; return e; }

  // Viridis colormap (5 key stops)
  var VIR = [[68,1,84],[59,82,139],[33,144,140],[93,201,99],[253,231,37]];
  function viridis(t) {
    t = Math.max(0, Math.min(1, t));
    var s = t * 4, lo = Math.floor(s), hi = Math.min(lo + 1, 4), f = s - lo;
    return 'rgb(' +
      Math.round(VIR[lo][0] + f * (VIR[hi][0] - VIR[lo][0])) + ',' +
      Math.round(VIR[lo][1] + f * (VIR[hi][1] - VIR[lo][1])) + ',' +
      Math.round(VIR[lo][2] + f * (VIR[hi][2] - VIR[lo][2])) + ')';
  }

  // Shared layout constants
  var W = 320, H = 280;
  var PD = { t: 25, r: 20, b: 40, l: 50 };
  var PW = W - PD.l - PD.r;   // 250
  var PH = H - PD.t - PD.b;   // 215

  // Scatter coordinate domain
  var SX0 = -0.5, SX1 = 10.5, SY0 = -2, SY1 = 23;
  function sx(x) { return PD.l + (x - SX0) / (SX1 - SX0) * PW; }
  function sy(y) { return PD.t + (SY1 - y) / (SY1 - SY0) * PH; }

  // Contour domain: intercept on x-axis, slope on y-axis
  var M0 = -0.5, M1 = 3.5, B0 = -4.0, B1 = 14.0;
  function cx(b) { return PD.l + (b - B0) / (B1 - B0) * PW; }
  function cy(m) { return PD.t + (M1 - m) / (M1 - M0) * PH; }

  // Pre-compute 120x120 heatmap losses once
  var GN = 120, cw = PW / GN, ch = PH / GN;
  var losses = new Array(GN * GN);
  var lMin = Infinity;
  for (var gi = 0; gi < GN; gi++) {
    for (var gj = 0; gj < GN; gj++) {
      var lv = olsLoss(M0 + (gj + 0.5) / GN * (M1 - M0), B0 + (gi + 0.5) / GN * (B1 - B0));
      losses[gi * GN + gj] = lv;
      if (lv < lMin) lMin = lv;
    }
  }
  var lRange = 20; // colormap spans 20 loss units above minimum

  function drawHeatmap() {
    var g = document.getElementById('ols-contour-heatmap');
    for (var gi = 0; gi < GN; gi++) {
      for (var gj = 0; gj < GN; gj++) {
        g.appendChild(el('rect', {
          x: PD.l + gi * cw,
          y: PD.t + (GN - 1 - gj) * ch,
          width: cw + 0.5, height: ch + 0.5,
          fill: viridis(Math.min(1, (losses[gi * GN + gj] - lMin) / lRange))
        }));
      }
    }
  }

  function drawAxes(gId, xVals, xFmt, xToSvg, yVals, yFmt, yToSvg, xLabel, yLabel) {
    var g = document.getElementById(gId); g.innerHTML = '';
    g.appendChild(el('line', { x1: PD.l, y1: H - PD.b, x2: W - PD.r, y2: H - PD.b, stroke: '#8b949e', 'stroke-width': 1 }));
    g.appendChild(el('line', { x1: PD.l, y1: PD.t,     x2: PD.l,     y2: H - PD.b, stroke: '#8b949e', 'stroke-width': 1 }));
    xVals.forEach(function (v) {
      var px = xToSvg(v);
      g.appendChild(el('line', { x1: px, y1: H - PD.b, x2: px, y2: H - PD.b + 4, stroke: '#8b949e', 'stroke-width': 1 }));
      g.appendChild(txt(xFmt(v), { x: px, y: H - PD.b + 15, 'text-anchor': 'middle', fill: '#8b949e', 'font-size': 10, 'font-family': 'monospace' }));
    });
    yVals.forEach(function (v) {
      var py = yToSvg(v);
      g.appendChild(el('line', { x1: PD.l - 4, y1: py, x2: PD.l, y2: py, stroke: '#8b949e', 'stroke-width': 1 }));
      g.appendChild(txt(yFmt(v), { x: PD.l - 6, y: py + 4, 'text-anchor': 'end', fill: '#8b949e', 'font-size': 10, 'font-family': 'monospace' }));
    });
    g.appendChild(txt(xLabel, { x: PD.l + PW / 2, y: H - 5, 'text-anchor': 'middle', fill: '#8b949e', 'font-size': 11, 'font-family': 'monospace' }));
    var mid = PD.t + PH / 2;
    g.appendChild(txt(yLabel, { x: 12, y: mid, 'text-anchor': 'middle', fill: '#8b949e', 'font-size': 11, 'font-family': 'monospace', transform: 'rotate(-90,12,' + mid + ')' }));
  }

  function drawScatterAxes() {
    drawAxes('ols-scatter-axes',
      [0, 2, 4, 6, 8, 10], function (v) { return v; }, sx,
      [0, 5, 10, 15, 20],  function (v) { return v; }, sy,
      'x', 'y');
  }

  function drawContourAxes() {
    drawAxes('ols-contour-axes',
      [0, 4, 8, 12], function (v) { return v; }, cx,
      [0, 1, 2, 3], function (v) { return v; }, cy,
      'β₀', 'β₁');
  }

  function drawPoints() {
    var g = document.getElementById('ols-scatter-points'); g.innerHTML = '';
    for (var i = 0; i < xs.length; i++) {
      g.appendChild(el('circle', { cx: sx(xs[i]), cy: sy(ys[i]), r: 4, fill: '#c9d1d9', stroke: '#0d1117', 'stroke-width': 1 }));
    }
  }

  function updateLine(m, b) {
    var g = document.getElementById('ols-scatter-line'); g.innerHTML = '';
    g.appendChild(el('line', { x1: sx(SX0), y1: sy(m * SX0 + b), x2: sx(SX1), y2: sy(m * SX1 + b), stroke: '#58a6ff', 'stroke-width': 2 }));
  }

  function updateMarker(m, b) {
    var g = document.getElementById('ols-contour-marker'); g.innerHTML = '';
    var px = cx(b), py = cy(m), r = 6;
    g.appendChild(el('line', { x1: px - r, y1: py,     x2: px + r, y2: py,     stroke: 'white', 'stroke-width': 1.5 }));
    g.appendChild(el('line', { x1: px,     y1: py - r, x2: px,     y2: py + r, stroke: 'white', 'stroke-width': 1.5 }));
    g.appendChild(el('circle', { cx: px, cy: py, r: r, fill: 'none', stroke: 'white', 'stroke-width': 1.5 }));
  }

  function render() {
    var m = parseFloat(document.getElementById('ols-slope-slider').value);
    var b = parseFloat(document.getElementById('ols-intercept-slider').value);
    document.getElementById('ols-slope-readout').textContent = m.toFixed(2);
    document.getElementById('ols-intercept-readout').textContent = b.toFixed(2);
    updateLine(m, b);
    updateMarker(m, b);
    document.getElementById('ols-info').textContent =
      'β₁ = ' + m.toFixed(2) + ' | β₀ = ' + b.toFixed(2) + ' | OLS loss = ' + olsLoss(m, b).toFixed(2);
  }

  drawHeatmap();
  drawScatterAxes();
  drawContourAxes();
  drawPoints();
  render();

  document.getElementById('ols-slope-slider').addEventListener('input', render);
  document.getElementById('ols-intercept-slider').addEventListener('input', render);

  // Compute OLS closed-form optimal
  var N = xs.length, sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (var i = 0; i < N; i++) { sumX += xs[i]; sumY += ys[i]; sumXY += xs[i]*ys[i]; sumX2 += xs[i]*xs[i]; }
  var mOLS = (N*sumXY - sumX*sumY) / (N*sumX2 - sumX*sumX);
  var bOLS = (sumY - mOLS*sumX) / N;

  document.getElementById('ols-minimize-btn').addEventListener('click', function () {
    document.getElementById('ols-slope-slider').value = mOLS;
    document.getElementById('ols-intercept-slider').value = bOLS;
    render();
  });
})();
</script>
