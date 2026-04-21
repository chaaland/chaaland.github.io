---
title: "Loss Spikes in Gradient Descent"
categories:
  - Optimization
date: 2026-04-19 19:00:00 +0000
mathjax: true
tags:
  - Optimization
  - Gradient Descent
  - Machine Learning
toc: true
classes: wide
excerpt: "Loss spikes aren't noise. They're gradient descent briefly exceeding the edge of stability and snapping back. Here's why."
---

Loss spikes are a familiar sight when training neural networks: the loss drops steadily, then suddenly jumps before recovering. This post explains why they happen. Starting from the simple case of a quadratic loss, we build up to the edge of stability and derive, via a Taylor expansion of the gradient, why spikes are self-correcting.

## Gradient Descent

Given a differentiable loss function $$f: \mathbf{R}^d \to \mathbf{R}$$, gradient descent iteratively updates parameters according to

$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

where $$\eta > 0$$ is the learning rate (or step size).
When $$f$$ is complicated, as in the case of a neural network loss landscape, it's difficult to choose an appropriate value of $$\eta$$.
Choosing a value that's too small leads to very slow convergence while choosing a value that's too large leads to divergence.

To understand this, it helps to first study a simple case to understand the source of the instability.

## 1D Quadratic Case

Consider the simplest case of minimizing the 1D quadratic $$f(x) = \frac{S}{2}x^2$$ where $$S > 0$$ is the _sharpness_ (curvature) of the function. Larger $$S$$ means a steeper, narrower parabola as shown in Figure 1.

<div class="widget-container" id="quadratic-widget">
  <div class="widget-controls">
    <label>
      S (sharpness)
      <input type="range" min="0.2" max="4" value="1" step="0.2" data-param="sharpness">
      <span class="widget-readout" data-readout="sharpness">1.0</span>
    </label>
  </div>
  <svg class="widget-plot" id="quadratic-svg" viewBox="0 0 500 280" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="500" height="280" fill="#0d1117"></rect>
    <g id="quadratic-grid"></g>
    <g id="quadratic-axes"></g>
    <path id="quadratic-curve" fill="none" stroke="#58a6ff" stroke-width="2.5" stroke-linejoin="round"></path>
  </svg>
  <div class="widget-info" id="quadratic-info">
    S = 1.0 &nbsp;
  </div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 1: Larger S makes the parabola steeper and smaller S makes it flatter.</figcaption>
</div>

The gradient/derivate is $$\nabla f(x) = Sx$$, so gradient descent becomes

$$x_{k+1} = x_k - \eta \cdot S x_k = (1 - S\eta) x_k$$

This is a simple geometric sequence! Starting from $$x_0$$, we have $$x_k = (1-S\eta)^k x_0.$$

For convergence to zero, we need

$$\lvert 1 - S\eta\rvert < 1$$

Since $$S, \eta > 0$$, this means

$$0 < S\eta < 2 \quad \Rightarrow \quad \eta < \frac{2}{S}.$$

This says that for a quadratic function, the maximum stable learning rate is inversely proportional to the sharpness.
Intuitively, sharper quadratics require smaller learning rates as demonstrated in Figure 2.

<div class="widget-container" id="gd1d-widget">
  <div class="widget-controls">
    <label>
      S (sharpness)
      <input type="range" min="0.2" max="4" value="1" step="0.2" data-param="gd1d-sharpness">
      <span class="widget-readout" data-readout="gd1d-sharpness">1.0</span>
    </label>
    <label>
      η (learning rate)
      <input type="range" min="0.05" max="2.5" value="0.5" step="0.05" data-param="gd1d-lr">
      <span class="widget-readout" data-readout="gd1d-lr">0.50</span>
    </label>
  </div>
  <svg class="widget-plot" id="gd1d-svg" viewBox="0 0 720 300" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="720" height="300" fill="#0d1117"></rect>
    <g id="gd1d-grid"></g>
    <g id="gd1d-axes"></g>
    <path id="gd1d-curve" fill="none" stroke="#58a6ff" stroke-width="2"></path>
    <path id="gd1d-path" fill="none" stroke="#f78166" stroke-width="1.5" stroke-dasharray="4,3"></path>
    <g id="gd1d-points"></g>
    <g id="gd1d-loss-grid"></g>
    <g id="gd1d-loss-axes"></g>
    <path id="gd1d-loss-path" fill="none" stroke="#f78166" stroke-width="2"></path>
    <g id="gd1d-loss-points"></g>
  </svg>
  <div class="widget-legend">
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#58a6ff"></span>f(x)</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#3fb950"></span>x₀</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#f78166"></span>Gradient descent</span>
  </div>
  <div class="widget-info" id="gd1d-info">
    S = 1.0 &nbsp;|&nbsp; η = 0.50 &nbsp;|&nbsp; η<sub>crit</sub> = 2.00 &nbsp;|&nbsp; <span style="color:#3fb950">Converging</span>
  </div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 2: Gradient descent for 15 steps. Left: iterates on the loss surface. Right: loss vs step on a log scale.</figcaption>
</div>

## The Multi-Dimensional Quadratic Case

In higher dimensions, the picture becomes richer. Consider a $$d$$-dimensional quadratic

$$f(x) = \frac{1}{2}x^T A x = \sum_{i=1}^d \sum_{j=1}^d A_{ij}x_ix_j$$

where $$A\in \mathbf{S}_+^{d}$$[^spd], the gradient of which is $$\nabla f = Ax$$.[^grad-general]
We can form the eigenvalue decomposition of the quadratic form as 

$$A = VDV^T, \quad D = \mathbf{diag}(\lambda_1, \ldots, \lambda_d)$$

where $$V^TV = I$$.

The quadratic form can then be written as

$$f(x) = \frac{1}{2} \sum_{k=1}^d \lambda_k \cdot (x^Tv_k)^2$$

where $$v_k$$ is the $$k^{th}$$ column of $$V$$.
Comparing this decomposition to our 1D case, we can see that the eigenvalues are _exactly_ the sharpnesses of the quadratic along the directions $$v_1, \ldots, v_d$$ (called the principal axes).[^sharpness-proof]

The level sets of a 2D quadratic are shown in Figure 3 which demonstrate how the eigenvalues affect the shape of the ellipses and thus the sharpness in the direction of the eigenvectors.

<div class="widget-container" id="contour-widget">
  <div class="widget-controls">
    <label>
      λ₁
      <input type="range" min="0.5" max="8" value="4" step="0.5" data-param="lambda1">
      <span class="widget-readout" data-readout="lambda1">4.0</span>
    </label>
    <label>
      λ₂
      <input type="range" min="0.5" max="8" value="1" step="0.5" data-param="lambda2">
      <span class="widget-readout" data-readout="lambda2">1.0</span>
    </label>
    <label>
      θ
      <input type="range" min="0" max="90" value="30" step="5" data-param="ctheta">
      <span class="widget-readout" data-readout="ctheta">30°</span>
    </label>
  </div>
  <svg class="widget-plot" id="contour-svg" viewBox="0 0 500 480" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="500" height="480" fill="#0d1117"></rect>
    <g id="contour-grid"></g>
    <g id="contour-axes"></g>
    <g id="contour-ellipses"></g>
    <text x="468" y="38" fill="#8b949e" font-size="13" text-anchor="end" font-style="italic">f(x) = ½(λ₁x₁² + λ₂x₂²)</text>
  </svg>
  <div class="widget-info" id="contour-info">
    λ₁ = 4.0 &nbsp;|&nbsp; λ₂ = 1.0
  </div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 3: Level sets (contours) of the 2D quadratic. When λ₁ = λ₂ the contours are circles; unequal eigenvalues produce ellipses elongated along the less-sharp direction.</figcaption>
</div>

In the 1D quadratic case, we derived the simple rule that $$\eta < 2/S$$ for gradient descent to converge.
For the $$d$$-dimensional quadratic case, how should we set the learning rate to ensure convergence?

Figure 4 shows the trajectory of gradient descent on a 2D quadratic.

<div class="widget-container" id="gd-widget" style="max-width: 900px;">
  <div class="widget-controls">
    <label>
      λ₁
      <input type="range" min="0.5" max="4" value="4" step="0.5" data-param="gd-lambda1">
      <span class="widget-readout" data-readout="gd-lambda1">4.0</span>
    </label>
    <label>
      λ₂
      <input type="range" min="0.5" max="4" value="1" step="0.5" data-param="gd-lambda2">
      <span class="widget-readout" data-readout="gd-lambda2">1.0</span>
    </label>
    <label>
      θ
      <input type="range" min="0" max="90" value="30" step="5" data-param="theta">
      <span class="widget-readout" data-readout="theta">30°</span>
    </label>
    <label>
      η
      <input type="range" min="0.01" max="1.0" value="0.15" step="0.01" data-param="lr">
      <span class="widget-readout" data-readout="lr">0.15</span>
    </label>
    <label>
      steps
      <input type="range" min="5" max="100" value="30" step="1" data-param="steps">
      <span class="widget-readout" data-readout="steps">30</span>
    </label>
    <button type="button" class="widget-button" id="gd-reset">Reset</button>
  </div>
  <svg class="widget-plot" id="gd-svg" viewBox="0 0 880 420" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="880" height="420" fill="#0d1117"></rect>
    <!-- Left plot: Contours -->
    <g id="gd-left-plot">
      <g id="gd-grid"></g>
      <g id="gd-contours"></g>
      <g id="gd-axes"></g>
      <path id="gd-path" fill="none" stroke="#f78166" stroke-width="2" stroke-linecap="round" stroke-dasharray="6,4"></path>
      <g id="gd-points"></g>
      <circle id="gd-start" r="7" fill="#3fb950" stroke="#0d1117" stroke-width="2"></circle>
      <circle id="gd-end" r="6" fill="#f78166" stroke="#0d1117" stroke-width="2"></circle>
    </g>
    <!-- Right plot: Loss vs Step -->
    <g id="gd-right-plot">
      <g id="loss-grid"></g>
      <g id="loss-axes"></g>
      <path id="loss-gd-path" fill="none" stroke="#f78166" stroke-width="2" stroke-linecap="round"></path>
      <g id="loss-points"></g>
      <text id="loss-ylabel" x="470" y="210" fill="#8b949e" font-size="13" text-anchor="middle" transform="rotate(-90, 470, 210)">Loss f(x)</text>
      <text id="loss-xlabel" x="670" y="405" fill="#8b949e" font-size="13" text-anchor="middle">Step</text>
    </g>
  </svg>
  <div class="widget-legend">
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#f78166"></span>Gradient descent</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#3fb950"></span>Start</span>
  </div>
  <div class="widget-info" id="gd-info">
    λ<sub>max</sub> = <span id="gd-lambda">4.00</span> &nbsp;|&nbsp;
    η<sub>crit</sub> = 2/λ<sub>max</sub> = <span id="gd-eta-crit">0.50</span> &nbsp;|&nbsp;
    <span id="gd-status" style="color:#3fb950">Converging</span>
  </div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 4: Gradient descent trajectory on a quadratic. The left panel shows iterates on the loss contours; the right panel shows loss vs. step. Increasing η toward the critical value 2/λ<sub>max</sub> causes oscillations along the sharpest eigendirection, and crossing it leads to divergence.</figcaption>
</div>

From the figure, we can see that convergence occurs when

$$
\eta < \frac{2}{\max\{\lambda_1, \lambda_2\}}.
$$

For the general $$d$$-dimensional case

$$\eta < \frac{2}{\max\{\lambda_1, \ldots, \lambda_d\}}.$$

This says that, for a quadratic, the maximal learning rate with which gradient descent will converge is governed by the sharpness along each principal axis.
More specifically, it is determined by the _sharpest_ of these directions.
Since the principal axis with the largest sharpness is the only one that governs convergence, we define $$S= \max\{\lambda_1, \ldots, \lambda_d\}$$ as the sharpness of a quadratic in the general case.
This generalizes the 1D definition, where the single eigenvalue $$S$$ of $$f(x) = \frac{S}{2}x^2$$ was the sharpness.

## Non-convex Case

Let's turn to the more complicated case of minimizing a general objective $$f(x)$$.
Writing the second order Taylor expansion about point $$a\in \mathbf{R}^d$$

$$f(x)\approx f(a) + \nabla f(a)^T(x-a) + \frac{1}{2}(x-a)^T\nabla^2 f(a) (x-a)$$

we define the sharpness at $$a$$, denoted $$S(a)\in \mathbf{R}_+$$, to be the maximum eigenvalue of the Hessian $$\nabla^2 f(a)$$.[^sharpness-local]

In order to have a more concrete example to visualize, let's consider the optimization problem

$$
\begin{align*}
\underset{C,\alpha}{\text{minimize}}&\quad \frac{1}{2N}\sum_{k=1}^N (y_k - Cx_k^\alpha)^2\\
\end{align*}
$$

which comes from fitting a power law $$f(x) = Cx^\alpha$$ to $$N$$ data points.

Figure 5 shows the level sets of the objective along with the gradient descent trajectory

<div class="widget-container" id="nc-widget">
  <div class="widget-controls">
    <label>
      η (learning rate)
      <input type="range" min="1" max="100" value="1" step="1" data-param="nc-eta">
      <span class="widget-readout" data-readout="nc-eta">1</span>
    </label>
  </div>
  <svg class="widget-plot" id="nc-svg" viewBox="0 0 500 400" preserveAspectRatio="xMidYMid meet">
    <defs>
      <clipPath id="nc-clip">
        <rect x="60" y="30" width="420" height="325"></rect>
      </clipPath>
    </defs>
    <rect x="0" y="0" width="500" height="400" fill="#0d1117"></rect>
    <g id="nc-heatmap" clip-path="url(#nc-clip)"></g>
    <g id="nc-traj-g" clip-path="url(#nc-clip)">
      <g id="nc-traj-segs"></g>
      <circle id="nc-start" r="5" fill="#3fb950"></circle>
      <circle id="nc-end" r="5" fill="#f78166" stroke="#0d1117" stroke-width="1.5"></circle>
    </g>
    <g id="nc-axes"></g>
  </svg>
  <div class="widget-legend">
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#3fb950"></span>Start (K=0, α=−0.8)</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#f78166"></span>End</span>
  </div>
  <div class="widget-info" id="nc-info">η = 1</div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 5: Level sets of the power law  non-linear least squares objective on word frequency data. Brighter regions indicate higher loss.</figcaption>
</div>

Notice how for $$\eta > 57$$, the trajectory shows increasingly oscillatory behavior and eventually completely diverges.
This indicates that our learning rate is somehow "too big" for our optimization landscape.
We previously found the maximum learning rate we could tolerate for a quadratic was $$2/S$$.

We can turn the question around and ask, "for a fixed learning rate $$\eta$$, what's the maximum sharpness $$S$$ that can be tolerated?".
The answer is, a sharpness less than $$2/\eta$$ which is referred to variously as the _critical sharpness_ or _the edge of stability_.

Figure 6 plots both the loss and sharpness at each point along the gradient descent trajectory. The sharpness threshold for the chosen $$\eta$$ is also displayed as a horizontal line on the sharpness plot.

<div class="widget-container" id="ls-widget">
  <div class="widget-controls">
    <label>
      η (learning rate)
      <input type="range" min="1" max="100" value="1" step="1" data-param="ls-eta">
      <span class="widget-readout" data-readout="ls-eta">1</span>
    </label>
  </div>
  <svg class="widget-plot" id="ls-svg" viewBox="0 0 700 460" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="700" height="460" fill="#0d1117"></rect>
    <g id="ls-loss-grid"></g>
    <g id="ls-loss-axes"></g>
    <path id="ls-loss-path" fill="none" stroke="#58a6ff" stroke-width="1.5" stroke-linejoin="round"></path>
    <g id="ls-sharp-grid"></g>
    <g id="ls-sharp-axes"></g>
    <path id="ls-sharp-path" fill="none" stroke="#58a6ff" stroke-width="1.5" stroke-linejoin="round"></path>
    <line id="ls-crit-line" stroke="#f78166" stroke-width="1.5" stroke-dasharray="5,3" stroke-opacity="0"></line>
    <text id="ls-crit-label" fill="#f78166" font-size="10" visibility="hidden"></text>
  </svg>
  <div class="widget-legend">
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#58a6ff"></span>Loss</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#f78166"></span>S = 2/η</span>
  </div>
  <div class="widget-info" id="ls-info">η = 70 &nbsp;|&nbsp; 2/η ≈ 0.0286</div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 6: Loss (log scale) and sharpness during gradient descent for 2500 steps on the non-linear least squares objective. The dashed orange line marks S&nbsp;=&nbsp;2/η (the critical sharpness above which gradient descent would diverge on a quadratic).</figcaption>
</div>

Notice that for $$\eta$$ below ~57, the loss decreases monotonically and the sharpness along the gradient descent trajectory never exceeds the critical sharpness $$2/\eta$$.

After $$\eta> 57$$, we see two things occur.
The first is that _loss spikes_ begin appearing and we no longer observe a monotone decreasing loss during gradient descent.
The second is that the sharpness begins exceeding the critical sharpness and then rapidly plunging back down below it.
This is very different than the quadratic case where once we exceeded the critical threshold, gradient descent diverged.

Also notice that as we increase $$\eta$$, we see more oscillations around the critical sharpness with an exponentially decaying envelope.
Simultaneously, we see larger and more frequent loss spikes.
After around $$\eta=90$$, we see oscillations in the sharpness display an exponentially _growing_ envelope as well as the loss diverging.

To understand what causes the sharpness to suddently decrease after exceeding the critical sharpness, Figure 7 decomposes the gradient at each step into its components along the dominant eigenvector $$v_1$$ (sharpest direction) and the non-dominant eigenvector $$v_2$$ of the local Hessian $$\nabla^2 f$$.

<div class="widget-container" id="gc-widget">
  <svg class="widget-plot" id="gc-svg" viewBox="0 0 700 480" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="700" height="480" fill="#0d1117"></rect>
    <g id="gc-sharp-grid"></g>
    <g id="gc-sharp-axes"></g>
    <path id="gc-sharp-path" fill="none" stroke="#58a6ff" stroke-width="1.5" stroke-linejoin="round"></path>
    <line id="gc-crit-line" stroke="#f78166" stroke-width="1.5" stroke-dasharray="5,3" stroke-opacity="0"></line>
    <text id="gc-crit-label" fill="#f78166" font-size="10" visibility="hidden"></text>
    <g id="gc-comp-grid"></g>
    <g id="gc-comp-axes"></g>
    <path id="gc-v1-path" fill="none" stroke="#58a6ff" stroke-width="1.5" stroke-linejoin="round"></path>
    <path id="gc-v2-path" fill="none" stroke="#3fb950" stroke-width="1.5" stroke-linejoin="round"></path>
  </svg>
  <div class="widget-legend">
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#58a6ff"></span>∇f·v₁ (dominant)</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#3fb950"></span>∇f·v₂ (non-dominant)</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#f78166"></span>S = 2/η</span>
  </div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 7: Sharpness (top) and gradient decomposed along the dominant eigenvector v₁ and non-dominant eigenvector v₂ of the local Hessian (bottom) for η = 70. The dominant component oscillates (sign-flipping) once sharpness exceeds 2/η.</figcaption>
</div>

Notice how just when the sharpness drops down precipitously below the edge of stability, we see large oscillations back and forth in the component of the gradient vector in the direction of maximum sharpness.
It turns out that this is the crux to understanding the phenomenon of loss spikes as we'll see in the next section.

## Mathematics of Loss Spikes

To study the dynamics of the loss curve, let's look at the Taylor expansion, not of the loss, but of the loss's gradient.
For ease of notation, we'll denote the gradient as $$g(x)=\nabla f(x)$$.

We'll perform the second order Taylor expansion around the point $$a\in\mathbf{R}^d$$.
Since this would lead to a third order tensor, we'll just look a single scalar component $$g_k$$ of the gradient approximation centered at $$a$$

$$g_k(x) \approx g_k(a) + Dg_k(a)(x-a) + \frac{1}{2}(x-a)^T \nabla^2 g_k(a) (x-a), \quad k=1,\ldots, d$$

Evaluating this quadratic approximation at a perturbed point $$a+\delta\in\mathbf{R}^d$$ we have

$$g_k(a+\delta) \approx g_k(a) + Dg_k(a)\delta + \frac{1}{2}\delta^T \nabla^2 g_k(a) \delta, \quad k=1,\ldots, d$$

From analyzing the quadratic case, we observed that when the sharpness exceeds the critical threshold $$2/\eta$$, the perturbation was largely in the direction of maximal sharpness. 
For this reason we set $$\delta = \sigma u$$ where $$u\in\mathbf{R}^d$$ is a unit vector in the direction of the dominant eigenvector of $$\nabla^2f$$ and $$\sigma \in \mathbf{R}$$. 
This gives,

$$
g_k(a+\sigma u) \approx g_k(a) + \sigma Dg_k(a)u + \frac{\sigma^2}{2}u^T \nabla^2 g_k(a) u.
$$

The term $$Dg_k(a)\in \mathbf{R}^{1\times d}$$ is the total derivative of the $$k^{th}$$ component of the gradient at $$a$$.
This is exactly the $$k^{th}$$ row of the hessian $$\nabla^2 f(a)$$.
This means $$Dg_k(a) u$$ is simply, $$[\nabla^2 f(a) u]_k$$, the $$k^{th}$$ component of the product of the hessian and its dominant eigenvector.

Since the eigenvalue associated with the dominant eigenvector is _by definition_ the sharpness, we have $$[\nabla^2 f(a) u]_k = S(a)u_k$$. 
So the Taylor approximation of the gradient becomes

$$
g_k(a+\sigma u) \approx g_k(a) + \sigma S(a)u_k + \frac{\sigma^2}{2}u^T \nabla^2 g_k(a) u.
$$

We can also simplify the final term involving the hessian of the gradient (a third derivative!) using the definition $$g_k(a) = \partial f(a)/\partial x_k$$ and the identity $$x^TAx = \sum_{i=1}^N\sum_{j=1}^N A_{ij}x_ix_j$$

$$
\begin{align*}
\frac{\sigma^2}{2}u^T \nabla^2 g_k(a) u &= \frac{\sigma^2}{2}\sum_{i=1}^d\sum_{j=1}^d \frac{\partial^2 g_k(a)}{ \partial x_i \partial x_j} u_i u_j  \\
&= \frac{\sigma^2}{2}\sum_{i=1}^d\sum_{j=1}^d \frac{\partial^3 f(a)}{\partial x_k \partial x_i \partial x_j} u_i u_j  \\
&= \frac{\sigma^2}{2}\left[\frac{\partial}{\partial x_k}\sum_{i=1}^d\sum_{j=1}^d \frac{\partial^2 f(x)}{\partial x_i \partial x_j} u_i u_j\right]_{x=a}  \\
&= \frac{\sigma^2}{2}\left[\frac{\partial}{\partial x_k}\left(u^T \nabla^2 f(x) u\right)\right]_{x=a}  \\
\end{align*}
$$

Simplifying the last part is tricky but a sketch of the argument comes from looking at the limit definition of the partial derivative with respect to $$x_k$$, where $$e_k$$ is the $$k^{th}$$ standard basis vector.

$$
\left[\frac{\partial}{\partial x_k}\left(u^T \nabla^2 f(x) u\right)\right]_{x=a} = \lim_{h\rightarrow 0} \frac{u^T \nabla^2 f(a+he_k) u - u^T\nabla^2 f(a)u}{h}
$$

Since $$u$$ is the dominant eigenvector of $$\nabla^2 f(a)$$, we have $$\nabla^2 f(a)u = S(a)u$$.
Combined with the fact that $$u^Tu=1$$, we can simplify the above to

$$
\left[\frac{\partial}{\partial x_k}\left(u^T \nabla^2 f(x) u\right)\right]_{x=a} = \lim_{h\rightarrow 0} \frac{u^T \nabla^2 f(a+he_k) u - S(a)}{h}
$$

Since $$u$$ is an eigenvector of $$\nabla^2 f(a)$$ and _not_ $$\nabla^2 f(a+he_k)$$, this does not immediately simplify as before.
However, for very small $$h$$ we can treat $$u$$ as constant and as an eigenvector of $$\nabla^2 f(a+he_k)$$, in which case the corresponding eigenvalue would be $$S(a+he_k)$$.
This simplifies the partial derivative to

$$
\begin{align*}
\left[\frac{\partial}{\partial x_k}\left(u^T \nabla^2 f(x) u\right)\right]_{x=a} &= \lim_{h\rightarrow 0} \frac{S(a+he_k) - S(a)}{h}\\
&= \frac{\partial S(a)}{\partial x_k}
\end{align*}
$$

With this simplification, the second order gradient approximation for the $$k^{th}$$ component at $$a+\sigma u$$ is

$$g_k(a+\sigma u) \approx g_k(a) + \sigma S(a)u_k + \frac{\sigma^2}{2}\frac{\partial S(a)}{\partial x_k}.$$

The approximation for the entire gradient vector is then

$$g(a+\sigma u) \approx g(a) + \sigma S(a)u + \frac{\sigma^2}{2}\nabla S(a).$$

This says that when the perturbation is small, a step in the negative gradient direction is dominated by the term $$-\sigma S(a)u$$ which pushes in the _opposite_ direction of the original perturbation $$\sigma u$$, causing the oscillatory behavior observed in Figure 7.
When the perturbation is sufficiently large however, a step in the negative gradient direction is strongly in the direction of _decreasing_ sharpness (i.e. $$-\frac{\sigma^2}{2}\nabla S(a)$$).

This effectively explains our observations!
During gradient descent, our loss spikes occur when the sharpness exceeds the critical sharpness $$2/\eta$$ for the given learning rate.
The loss drops back down because the sharpness rapidly drops back below $$2/\eta$$.
But the reason the sharpness drops after having exceeded the edge of stability is because there is a built in negative feedback!
When the perturbation along the dominant eigenvector of the hessian gets too large, the negative gradient of the loss also begins to point in the direction of the negative gradient of the sharpness, driving the sharpness back below the edge of stability.

## Conclusion



[^grad-general]: More precisely, $$\nabla_x \tfrac{1}{2}x^T A x = \tfrac{1}{2}(A + A^T)x$$. When $$A$$ is symmetric this reduces to $$Ax$$.

[^spd]: $$\mathbf{S}_+^{d}$$ denotes the set of $$d\times d$$ symmetric positive semidefinite matrices.

[^sharpness-local]: For a quadratic $$f(x) = \tfrac{1}{2}x^TAx$$, the Hessian is the constant matrix $$A$$, so the sharpness is the same at every point (i.e. a global property). For a general nonlinear function the Hessian varies, making sharpness point-dependent. Note also that this definition is a direct generalization of the $$d$$-dimensional quadratic case: the second order Taylor expansion of a quadratic is just the quadratic itself, so the maximum eigenvalue of $$\nabla^2 f(a) = A$$ recovers exactly $$S = \max\{\lambda_1,\ldots,\lambda_d\}$$.

[^sharpness-proof]: To see why, set $$x = t\,v_k$$ for scalar $$t$$. Because the columns of $$V$$ are orthonormal, $$x^T v_j = t\,v_k^T v_j = t\,\delta_{kj}$$, so every term in the sum vanishes except the $$k^{th}$$ one: $$f(tv_k) = \tfrac{1}{2}\lambda_k t^2$$. This is exactly the 1D quadratic with sharpness $$\lambda_k$$, confirming that $$\lambda_k$$ governs the curvature of $$f$$ along $$v_k$$.

## References

1. [How does gradient descent work](https://centralflows.github.io/part1/)
2. Boyd, S. & Vandenberghe, L. "Convex Optimization." Cambridge University Press, 2004.

<script>
// 2D contour-only widget (Figure 3)
(function() {
  const W = 500, H = 480;
  const margin = { left: 50, right: 30, top: 20, bottom: 40 };
  const xMin = -3.5, xMax = 3.5, yMin = -3.5, yMax = 3.5;
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  function toSvgX(x) { return margin.left + (x - xMin) / (xMax - xMin) * plotW; }
  function toSvgY(y) { return H - margin.bottom - (y - yMin) / (yMax - yMin) * plotH; }

  const gridG = document.getElementById('contour-grid');
  const axesG = document.getElementById('contour-axes');
  const ellipsesG = document.getElementById('contour-ellipses');
  const infoDiv = document.getElementById('contour-info');

  function drawGrid() {
    gridG.innerHTML = '';
    for (let v = Math.ceil(xMin); v <= Math.floor(xMax); v++) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', toSvgX(v)); line.setAttribute('y1', margin.top);
      line.setAttribute('x2', toSvgX(v)); line.setAttribute('y2', H - margin.bottom);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
    for (let v = Math.ceil(yMin); v <= Math.floor(yMax); v++) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', margin.left); line.setAttribute('y1', toSvgY(v));
      line.setAttribute('x2', W - margin.right); line.setAttribute('y2', toSvgY(v));
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
  }

  function drawAxes() {
    axesG.innerHTML = '';
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin.left); xAxis.setAttribute('y1', toSvgY(0));
    xAxis.setAttribute('x2', W - margin.right); xAxis.setAttribute('y2', toSvgY(0));
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(xAxis);
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', toSvgX(0)); yAxis.setAttribute('y1', margin.top);
    yAxis.setAttribute('x2', toSvgX(0)); yAxis.setAttribute('y2', H - margin.bottom);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(yAxis);
    for (let v = Math.ceil(xMin); v <= Math.floor(xMax); v++) {
      if (v === 0) continue;
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', toSvgX(v)); label.setAttribute('y', H - margin.bottom + 15);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'middle');
      label.textContent = v;
      axesG.appendChild(label);
    }
    for (let v = Math.ceil(yMin); v <= Math.floor(yMax); v++) {
      if (v === 0) continue;
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - 8); label.setAttribute('y', toSvgY(v) + 4);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'end');
      label.textContent = v;
      axesG.appendChild(label);
    }
    const xl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xl.setAttribute('x', W / 2); xl.setAttribute('y', H - 5);
    xl.setAttribute('fill', '#8b949e'); xl.setAttribute('font-size', '12');
    xl.setAttribute('text-anchor', 'middle');
    xl.textContent = 'x\u2081';
    axesG.appendChild(xl);
    const yl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yl.setAttribute('x', 12); yl.setAttribute('y', H / 2);
    yl.setAttribute('fill', '#8b949e'); yl.setAttribute('font-size', '12');
    yl.setAttribute('text-anchor', 'middle');
    yl.setAttribute('transform', `rotate(-90, 12, ${H / 2})`);
    yl.textContent = 'x\u2082';
    axesG.appendChild(yl);
  }

  function drawEllipses(l1, l2, thetaDeg) {
    ellipsesG.innerHTML = '';
    const scaleX = plotW / (xMax - xMin);
    const scaleY = plotH / (yMax - yMin);
    const levels = [0.5, 1, 2, 3, 5, 8, 12];

    levels.forEach(c => {
      // l1 governs v1=[cos θ, sin θ] (horizontal when θ=0), l2 governs v2 (vertical)
      const rx = Math.sqrt(2 * c / l1) * scaleX;
      const ry = Math.sqrt(2 * c / l2) * scaleY;
      if (rx > W || ry > H) return;

      const ellipse = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
      ellipse.setAttribute('cx', toSvgX(0));
      ellipse.setAttribute('cy', toSvgY(0));
      ellipse.setAttribute('rx', rx);
      ellipse.setAttribute('ry', ry);
      // SVG y-axis is flipped, so negate rotation to match data-space angle
      ellipse.setAttribute('transform', `rotate(${-thetaDeg}, ${toSvgX(0)}, ${toSvgY(0)})`);
      ellipse.setAttribute('fill', 'none');
      ellipse.setAttribute('stroke', '#30363d');
      ellipse.setAttribute('stroke-width', 1.5);
      ellipsesG.appendChild(ellipse);
    });

    // Draw principal axes as dashed lines
    const theta = thetaDeg * Math.PI / 180;
    const axisLen = 3.2;
    [[Math.cos(theta), Math.sin(theta)], [-Math.sin(theta), Math.cos(theta)]].forEach(([ex, ey]) => {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', toSvgX(-axisLen * ex)); line.setAttribute('y1', toSvgY(-axisLen * ey));
      line.setAttribute('x2', toSvgX(axisLen * ex));  line.setAttribute('y2', toSvgY(axisLen * ey));
      line.setAttribute('stroke', '#484f58'); line.setAttribute('stroke-width', 1);
      line.setAttribute('stroke-dasharray', '4,3');
      ellipsesG.appendChild(line);
    });
  }

  function update() {
    const l1 = parseFloat(document.querySelector('[data-param="lambda1"]').value);
    const l2 = parseFloat(document.querySelector('[data-param="lambda2"]').value);
    const theta = parseFloat(document.querySelector('[data-param="ctheta"]').value);
    document.querySelector('[data-readout="lambda1"]').textContent = l1.toFixed(1);
    document.querySelector('[data-readout="lambda2"]').textContent = l2.toFixed(1);
    document.querySelector('[data-readout="ctheta"]').textContent = theta + '\u00b0';
    infoDiv.innerHTML = `\u03bb\u2081 = ${l1.toFixed(1)} &nbsp;|&nbsp; \u03bb\u2082 = ${l2.toFixed(1)}`;
    drawEllipses(l1, l2, theta);
  }

  drawGrid();
  drawAxes();
  update();

  document.querySelector('[data-param="lambda1"]').addEventListener('input', update);
  document.querySelector('[data-param="lambda2"]').addEventListener('input', update);
  document.querySelector('[data-param="ctheta"]').addEventListener('input', update);
})();

// 1D quadratic + GD widget (Figure 2)
(function() {
  const totalW = 720, totalH = 300;
  const leftW = 330, rightX = 370, rightW = 350;
  const margin = { left: 45, right: 20, top: 20, bottom: 40 };
  const STEPS = 15;
  const x0 = 2.5;
  const xMin = -3, xMax = 3, yMax = 12;

  const plotLeft = margin.left;
  const plotRight = leftW - margin.right;
  const plotTop = margin.top;
  const plotBottom = totalH - margin.bottom;

  function toSvgX(x) {
    return plotLeft + (x - xMin) / (xMax - xMin) * (plotRight - plotLeft);
  }
  function toSvgY(y) {
    return plotTop + (yMax - Math.min(y, yMax)) / yMax * (plotBottom - plotTop);
  }

  let logLossMax = 1;
  const logLossMin = -8;

  function toLossX(step) {
    return rightX + margin.left + (step / STEPS) * (rightW - margin.left - margin.right);
  }
  function toLossY(loss) {
    const logL = Math.log10(Math.max(loss, 1e-8));
    const clamped = Math.max(logLossMin, Math.min(logLossMax, logL));
    return plotTop + (logLossMax - clamped) / (logLossMax - logLossMin) * (plotBottom - plotTop);
  }

  function superscript(n) {
    const sup = ['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹'];
    const digits = String(Math.abs(n)).split('').map(d => sup[+d]).join('');
    return n < 0 ? '⁻' + digits : digits;
  }

  const gridG = document.getElementById('gd1d-grid');
  const axesG = document.getElementById('gd1d-axes');
  const curvePath = document.getElementById('gd1d-curve');
  const gdPath = document.getElementById('gd1d-path');
  const pointsG = document.getElementById('gd1d-points');
  const lossGridG = document.getElementById('gd1d-loss-grid');
  const lossAxesG = document.getElementById('gd1d-loss-axes');
  const lossPath = document.getElementById('gd1d-loss-path');
  const lossPointsG = document.getElementById('gd1d-loss-points');
  const infoDiv = document.getElementById('gd1d-info');

  function drawLeftStatic() {
    gridG.innerHTML = '';
    for (let v = -3; v <= 3; v++) {
      const sx = toSvgX(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', sx); line.setAttribute('y1', plotTop);
      line.setAttribute('x2', sx); line.setAttribute('y2', plotBottom);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
    for (let v = 0; v <= yMax; v += 2) {
      const sy = toSvgY(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', plotLeft); line.setAttribute('y1', sy);
      line.setAttribute('x2', plotRight); line.setAttribute('y2', sy);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
    axesG.innerHTML = '';
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', plotLeft); xAxis.setAttribute('y1', toSvgY(0));
    xAxis.setAttribute('x2', plotRight); xAxis.setAttribute('y2', toSvgY(0));
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(xAxis);
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', toSvgX(0)); yAxis.setAttribute('y1', plotTop);
    yAxis.setAttribute('x2', toSvgX(0)); yAxis.setAttribute('y2', plotBottom);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(yAxis);
    for (let v = -3; v <= 3; v++) {
      if (v === 0) continue;
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', toSvgX(v)); label.setAttribute('y', plotBottom + 15);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'middle');
      label.textContent = v;
      axesG.appendChild(label);
    }
    for (let v = 2; v <= yMax; v += 2) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', plotLeft - 8); label.setAttribute('y', toSvgY(v) + 4);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'end');
      label.textContent = v;
      axesG.appendChild(label);
    }
    const xl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xl.setAttribute('x', (plotLeft + plotRight) / 2); xl.setAttribute('y', totalH - 5);
    xl.setAttribute('fill', '#8b949e'); xl.setAttribute('font-size', '12');
    xl.setAttribute('text-anchor', 'middle');
    xl.textContent = 'x';
    axesG.appendChild(xl);
    const yl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yl.setAttribute('x', 12); yl.setAttribute('y', totalH / 2);
    yl.setAttribute('fill', '#8b949e'); yl.setAttribute('font-size', '12');
    yl.setAttribute('text-anchor', 'middle');
    yl.setAttribute('transform', `rotate(-90, 12, ${totalH / 2})`);
    yl.textContent = 'f(x)';
    axesG.appendChild(yl);
  }

  function drawRightAxes() {
    lossGridG.innerHTML = '';
    lossAxesG.innerHTML = '';
    for (let s = 0; s <= STEPS; s += 5) {
      const sx = toLossX(s);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', sx); line.setAttribute('y1', plotTop);
      line.setAttribute('x2', sx); line.setAttribute('y2', plotBottom);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      lossGridG.appendChild(line);
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', sx); label.setAttribute('y', plotBottom + 15);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'middle');
      label.textContent = s;
      lossAxesG.appendChild(label);
    }
    for (let exp = logLossMin; exp <= logLossMax; exp += 2) {
      const sy = toLossY(Math.pow(10, exp));
      if (sy < plotTop || sy > plotBottom) continue;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', rightX + margin.left); line.setAttribute('y1', sy);
      line.setAttribute('x2', rightX + rightW - margin.right); line.setAttribute('y2', sy);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      lossGridG.appendChild(line);
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', rightX + margin.left - 8); label.setAttribute('y', sy + 4);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'end');
      label.textContent = '10' + superscript(exp);
      lossAxesG.appendChild(label);
    }
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', rightX + margin.left); xAxis.setAttribute('y1', plotBottom);
    xAxis.setAttribute('x2', rightX + rightW - margin.right); xAxis.setAttribute('y2', plotBottom);
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    lossAxesG.appendChild(xAxis);
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', rightX + margin.left); yAxis.setAttribute('y1', plotTop);
    yAxis.setAttribute('x2', rightX + margin.left); yAxis.setAttribute('y2', plotBottom);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    lossAxesG.appendChild(yAxis);
    const xl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xl.setAttribute('x', rightX + margin.left + (rightW - margin.left - margin.right) / 2);
    xl.setAttribute('y', totalH - 5);
    xl.setAttribute('fill', '#8b949e'); xl.setAttribute('font-size', '12');
    xl.setAttribute('text-anchor', 'middle');
    xl.textContent = 'Step';
    lossAxesG.appendChild(xl);
    const yl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yl.setAttribute('x', rightX + 12); yl.setAttribute('y', totalH / 2);
    yl.setAttribute('fill', '#8b949e'); yl.setAttribute('font-size', '12');
    yl.setAttribute('text-anchor', 'middle');
    yl.setAttribute('transform', `rotate(-90, ${rightX + 12}, ${totalH / 2})`);
    yl.textContent = 'f(xₖ)';
    lossAxesG.appendChild(yl);
  }

  function update() {
    const S = parseFloat(document.querySelector('[data-param="gd1d-sharpness"]').value);
    const lr = parseFloat(document.querySelector('[data-param="gd1d-lr"]').value);
    document.querySelector('[data-readout="gd1d-sharpness"]').textContent = S.toFixed(1);
    document.querySelector('[data-readout="gd1d-lr"]').textContent = lr.toFixed(2);

    // Compute GD trajectory
    const xs = [x0];
    for (let k = 0; k < STEPS; k++) {
      xs.push(xs[xs.length - 1] * (1 - lr * S));
    }
    const losses = xs.map(x => 0.5 * S * x * x);

    // Draw parabola (clipped at yMax)
    const xClip = Math.sqrt(2 * yMax / S);
    const xStart = Math.max(xMin, -xClip), xEnd = Math.min(xMax, xClip);
    let d = '';
    for (let i = 0; i <= 300; i++) {
      const x = xStart + (xEnd - xStart) * i / 300;
      d += i === 0 ? `M ${toSvgX(x)} ${toSvgY(0.5 * S * x * x)}` : ` L ${toSvgX(x)} ${toSvgY(0.5 * S * x * x)}`;
    }
    curvePath.setAttribute('d', d);

    // Draw GD path connecting iterates on the parabola
    let gd = '';
    xs.forEach((x, i) => {
      const xc = Math.max(xMin, Math.min(xMax, x));
      const sx = toSvgX(xc), sy = toSvgY(0.5 * S * xc * xc);
      gd += i === 0 ? `M ${sx} ${sy}` : ` L ${sx} ${sy}`;
    });
    gdPath.setAttribute('d', gd);

    // Draw GD iterate dots on left plot
    pointsG.innerHTML = '';
    xs.forEach((x, i) => {
      const xc = Math.max(xMin, Math.min(xMax, x));
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', toSvgX(xc));
      circle.setAttribute('cy', toSvgY(0.5 * S * xc * xc));
      circle.setAttribute('r', i === 0 ? 5 : 3.5);
      circle.setAttribute('fill', i === 0 ? '#3fb950' : '#f78166');
      circle.setAttribute('stroke', '#0d1117');
      circle.setAttribute('stroke-width', i === 0 ? 1.5 : 1);
      pointsG.appendChild(circle);
    });

    // Set loss axis scale from initial loss
    logLossMax = Math.ceil(Math.log10(losses[0] + 1e-10));
    drawRightAxes();

    // Draw loss path
    let ld = '';
    losses.forEach((loss, i) => {
      const sy = toLossY(loss);
      ld += i === 0 ? `M ${toLossX(i)} ${sy}` : ` L ${toLossX(i)} ${sy}`;
    });
    lossPath.setAttribute('d', ld);

    // Draw loss dots
    lossPointsG.innerHTML = '';
    losses.forEach((loss, i) => {
      const sy = toLossY(loss);
      if (sy < plotTop - 5 || sy > plotBottom + 5) return;
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', toLossX(i));
      circle.setAttribute('cy', sy);
      circle.setAttribute('r', 3);
      circle.setAttribute('fill', '#f78166');
      circle.setAttribute('opacity', '0.85');
      lossPointsG.appendChild(circle);
    });

    // Update info bar
    const etaCrit = 2 / S;
    const converging = lr < etaCrit;
    const col = converging ? '#3fb950' : '#f85149';
    const status = converging ? 'Converging' : 'Diverging!';
    infoDiv.innerHTML = `S = ${S.toFixed(1)} &nbsp;|&nbsp; η = ${lr.toFixed(2)} &nbsp;|&nbsp; η<sub>crit</sub> = ${etaCrit.toFixed(2)} &nbsp;|&nbsp; <span style="color:${col}">${status}</span>`;
  }

  drawLeftStatic();
  update();

  document.querySelector('[data-param="gd1d-sharpness"]').addEventListener('input', update);
  document.querySelector('[data-param="gd1d-lr"]').addEventListener('input', update);
})();

// 1D quadratic widget
(function() {
  const W = 500, H = 280;
  const margin = { left: 50, right: 30, top: 20, bottom: 40 };
  const xMin = -3, xMax = 3;
  const yMax = 14;

  function toSvgX(x) {
    return margin.left + (x - xMin) / (xMax - xMin) * (W - margin.left - margin.right);
  }
  function toSvgY(y) {
    return margin.top + (yMax - Math.min(y, yMax)) / yMax * (H - margin.top - margin.bottom);
  }

  const gridG = document.getElementById('quadratic-grid');
  const axesG = document.getElementById('quadratic-axes');
  const curvePath = document.getElementById('quadratic-curve');
  const infoDiv = document.getElementById('quadratic-info');

  function drawGrid() {
    gridG.innerHTML = '';
    for (let v = -3; v <= 3; v++) {
      const sx = toSvgX(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', sx); line.setAttribute('y1', margin.top);
      line.setAttribute('x2', sx); line.setAttribute('y2', H - margin.bottom);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
    for (let v = 0; v <= yMax; v += 2) {
      const sy = toSvgY(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', margin.left); line.setAttribute('y1', sy);
      line.setAttribute('x2', W - margin.right); line.setAttribute('y2', sy);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
  }

  function drawAxes() {
    axesG.innerHTML = '';
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin.left); xAxis.setAttribute('y1', toSvgY(0));
    xAxis.setAttribute('x2', W - margin.right); xAxis.setAttribute('y2', toSvgY(0));
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(xAxis);
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', toSvgX(0)); yAxis.setAttribute('y1', margin.top);
    yAxis.setAttribute('x2', toSvgX(0)); yAxis.setAttribute('y2', H - margin.bottom);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(yAxis);
    for (let v = -3; v <= 3; v++) {
      if (v === 0) continue;
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', toSvgX(v)); label.setAttribute('y', H - margin.bottom + 15);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'middle');
      label.textContent = v;
      axesG.appendChild(label);
    }
    for (let v = 2; v <= yMax; v += 2) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - 8); label.setAttribute('y', toSvgY(v) + 4);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'end');
      label.textContent = v;
      axesG.appendChild(label);
    }
    const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xLabel.setAttribute('x', W / 2); xLabel.setAttribute('y', H - 5);
    xLabel.setAttribute('fill', '#8b949e'); xLabel.setAttribute('font-size', '12');
    xLabel.setAttribute('text-anchor', 'middle');
    xLabel.textContent = 'x';
    axesG.appendChild(xLabel);
    const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yLabel.setAttribute('x', 12); yLabel.setAttribute('y', H / 2);
    yLabel.setAttribute('fill', '#8b949e'); yLabel.setAttribute('font-size', '12');
    yLabel.setAttribute('text-anchor', 'middle');
    yLabel.setAttribute('transform', `rotate(-90, 12, ${H / 2})`);
    yLabel.textContent = 'f(x)';
    axesG.appendChild(yLabel);
  }

  function update() {
    const S = parseFloat(document.querySelector('[data-param="sharpness"]').value);
    document.querySelector('[data-readout="sharpness"]').textContent = S.toFixed(1);

    const xClip = Math.sqrt(2 * yMax / S);
    const xStart = Math.max(xMin, -xClip);
    const xEnd = Math.min(xMax, xClip);
    const N = 300;
    let d = '';
    for (let i = 0; i <= N; i++) {
      const x = xStart + (xEnd - xStart) * i / N;
      const y = 0.5 * S * x * x;
      const sx = toSvgX(x), sy = toSvgY(y);
      d += i === 0 ? `M ${sx} ${sy}` : ` L ${sx} ${sy}`;
    }
    curvePath.setAttribute('d', d);

    infoDiv.innerHTML = `S = ${S.toFixed(1)}`;
  }

  drawGrid();
  drawAxes();
  update();

  document.querySelector('[data-param="sharpness"]').addEventListener('input', update);
})();

// GD widget (2D ellipse + loss vs step)
(function() {
  // Layout constants
  const totalW = 880, totalH = 420;
  const leftW = 420, rightW = 400;
  const rightX = 480; // Starting x of right plot
  const margin = 40;

  // Left plot (contours) coordinate bounds
  const xMin = -3.5, xMax = 3.5, yMin = -3.5, yMax = 3.5;

  // Default starting point
  let x0_default = 2.5, y0_default = 2.0;
  let x0 = x0_default, y0 = y0_default;

  // Get DOM elements - left plot
  const svg = document.getElementById('gd-svg');
  const gridG = document.getElementById('gd-grid');
  const contoursG = document.getElementById('gd-contours');
  const axesG = document.getElementById('gd-axes');
  const gdPath = document.getElementById('gd-path');
  const pointsG = document.getElementById('gd-points');
  const startCircle = document.getElementById('gd-start');
  const endCircle = document.getElementById('gd-end');

  // Get DOM elements - right plot (loss)
  const lossGridG = document.getElementById('loss-grid');
  const lossAxesG = document.getElementById('loss-axes');
  const lossGdPath = document.getElementById('loss-gd-path');
  const lossPointsG = document.getElementById('loss-points');

  // Info elements
  const lambdaSpan = document.getElementById('gd-lambda');
  const etaCritSpan = document.getElementById('gd-eta-crit');
  const statusSpan = document.getElementById('gd-status');

  // Left plot coordinate transforms
  function toSvgX(x) { return margin + (x - xMin) / (xMax - xMin) * (leftW - 2 * margin); }
  function toSvgY(y) { return totalH - margin - (y - yMin) / (yMax - yMin) * (totalH - 2 * margin); }
  function toDataX(sx) { return xMin + (sx - margin) / (leftW - 2 * margin) * (xMax - xMin); }
  function toDataY(sy) { return yMin + (totalH - margin - sy) / (totalH - 2 * margin) * (yMax - yMin); }

  // Right plot coordinate transforms (dynamic based on loss range)
  let lossMax = 10;
  function toLossSvgX(step, maxSteps) {
    return rightX + margin + (step / maxSteps) * (rightW - 2 * margin);
  }
  function toLossSvgY(loss) {
    const logLoss = loss > 0 ? Math.log10(Math.max(loss, 1e-6)) : -6;
    const logMax = Math.log10(lossMax);
    const logMin = -6;
    return totalH - margin - ((logLoss - logMin) / (logMax - logMin)) * (totalH - 2 * margin);
  }

  // Helper to format superscript numbers (Unicode superscripts aren't contiguous)
  function toSuperscript(n) {
    const superDigits = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    const absN = Math.abs(n);
    const digits = String(absN).split('').map(d => superDigits[parseInt(d)]).join('');
    return n < 0 ? '⁻' + digits : digits;
  }

  // Build the Hessian matrix A = R·diag(l1,l2)·Rᵀ from eigenvalues and rotation
  function buildA(l1, l2, thetaDeg) {
    const theta = thetaDeg * Math.PI / 180;
    const c = Math.cos(theta), s = Math.sin(theta);
    const a11 = c * c * l1 + s * s * l2;
    const a12 = c * s * (l1 - l2);
    const a22 = s * s * l1 + c * c * l2;
    return [[a11, a12], [a12, a22]];
  }

  // Compute eigenvalues of 2x2 symmetric matrix
  function eigenvalues(A) {
    const tr = A[0][0] + A[1][1];
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    return [tr / 2 + disc, tr / 2 - disc];
  }

  // f(x) = 0.5 * x^T A x
  function f(x, y, A) {
    return 0.5 * (A[0][0] * x * x + 2 * A[0][1] * x * y + A[1][1] * y * y);
  }

  // Gradient: grad f = A x
  function grad(x, y, A) {
    return [A[0][0] * x + A[0][1] * y, A[0][1] * x + A[1][1] * y];
  }

  // Draw left plot grid
  function drawGrid() {
    gridG.innerHTML = '';
    for (let v = Math.ceil(xMin); v <= Math.floor(xMax); v++) {
      const sx = toSvgX(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', sx); line.setAttribute('y1', margin);
      line.setAttribute('x2', sx); line.setAttribute('y2', totalH - margin);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
    for (let v = Math.ceil(yMin); v <= Math.floor(yMax); v++) {
      const sy = toSvgY(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', margin); line.setAttribute('y1', sy);
      line.setAttribute('x2', leftW - margin); line.setAttribute('y2', sy);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
  }

  // Draw left plot axes
  function drawAxes() {
    axesG.innerHTML = '';
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin); xAxis.setAttribute('y1', toSvgY(0));
    xAxis.setAttribute('x2', leftW - margin); xAxis.setAttribute('y2', toSvgY(0));
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(xAxis);
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', toSvgX(0)); yAxis.setAttribute('y1', margin);
    yAxis.setAttribute('x2', toSvgX(0)); yAxis.setAttribute('y2', totalH - margin);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(yAxis);
  }

  // Draw right plot (loss) grid and axes
  function drawLossAxes(maxSteps) {
    lossGridG.innerHTML = '';
    lossAxesG.innerHTML = '';

    // Vertical grid lines (steps)
    const stepInterval = maxSteps <= 20 ? 5 : maxSteps <= 50 ? 10 : 20;
    for (let s = 0; s <= maxSteps; s += stepInterval) {
      const sx = toLossSvgX(s, maxSteps);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', sx); line.setAttribute('y1', margin);
      line.setAttribute('x2', sx); line.setAttribute('y2', totalH - margin);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      lossGridG.appendChild(line);
      // Label
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', sx); label.setAttribute('y', totalH - margin + 18);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'middle');
      label.textContent = s;
      lossAxesG.appendChild(label);
    }

    // Horizontal grid lines (log loss)
    const logMax = Math.log10(lossMax);
    for (let exp = -6; exp <= logMax; exp += 2) {
      const sy = toLossSvgY(Math.pow(10, exp));
      if (sy < margin || sy > totalH - margin) continue;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', rightX + margin); line.setAttribute('y1', sy);
      line.setAttribute('x2', rightX + rightW - margin); line.setAttribute('y2', sy);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      lossGridG.appendChild(line);
      // Label
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', rightX + margin - 8); label.setAttribute('y', sy + 4);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'end');
      label.textContent = '10' + toSuperscript(exp);
      lossAxesG.appendChild(label);
    }

    // Axes
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', rightX + margin); xAxis.setAttribute('y1', totalH - margin);
    xAxis.setAttribute('x2', rightX + rightW - margin); xAxis.setAttribute('y2', totalH - margin);
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    lossAxesG.appendChild(xAxis);
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', rightX + margin); yAxis.setAttribute('y1', margin);
    yAxis.setAttribute('x2', rightX + margin); yAxis.setAttribute('y2', totalH - margin);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    lossAxesG.appendChild(yAxis);
  }

  // Draw elliptical contours
  function drawContours(l1, l2, thetaDeg) {
    contoursG.innerHTML = '';
    const levels = [0.5, 1, 2, 3, 5, 8, 12];

    levels.forEach(c => {
      // l1 governs v1=[cos θ, sin θ] (horizontal before rotation), l2 governs v2
      const ax = Math.sqrt(2 * c / l1);
      const ay = Math.sqrt(2 * c / l2);
      if (ax > 10 || ay > 10) return;

      const ellipse = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
      ellipse.setAttribute('cx', toSvgX(0));
      ellipse.setAttribute('cy', toSvgY(0));
      const scaleX = (leftW - 2 * margin) / (xMax - xMin);
      const scaleY = (totalH - 2 * margin) / (yMax - yMin);
      ellipse.setAttribute('rx', ax * scaleX);
      ellipse.setAttribute('ry', ay * scaleY);
      ellipse.setAttribute('transform', `rotate(${-thetaDeg}, ${toSvgX(0)}, ${toSvgY(0)})`);
      ellipse.setAttribute('fill', 'none');
      ellipse.setAttribute('stroke', '#30363d');
      ellipse.setAttribute('stroke-width', 1);
      contoursG.appendChild(ellipse);
    });
  }

  // Compute gradient flow trajectory with loss values
  // Compute gradient descent trajectory with loss values
  function computeGDPath(x0, y0, A, lr, steps) {
    const pts = [[x0, y0]];
    const losses = [f(x0, y0, A)];
    let x = x0, y = y0;
    for (let i = 0; i < steps; i++) {
      const [gx, gy] = grad(x, y, A);
      x -= lr * gx;
      y -= lr * gy;
      pts.push([x, y]);
      losses.push(f(x, y, A));
      if (Math.abs(x) > 100 || Math.abs(y) > 100) break;
    }
    return { pts, losses };
  }

  // Convert contour points to SVG path
  function pointsToPath(pts) {
    if (pts.length === 0) return '';
    let d = `M ${toSvgX(pts[0][0])} ${toSvgY(pts[0][1])}`;
    for (let i = 1; i < pts.length; i++) {
      d += ` L ${toSvgX(pts[i][0])} ${toSvgY(pts[i][1])}`;
    }
    return d;
  }

  // Convert loss values to SVG path
  function lossToPath(losses, maxSteps) {
    if (losses.length === 0) return '';
    let d = `M ${toLossSvgX(0, maxSteps)} ${toLossSvgY(losses[0])}`;
    for (let i = 1; i < losses.length; i++) {
      const loss = Math.min(losses[i], lossMax * 10);
      d += ` L ${toLossSvgX(i, maxSteps)} ${toLossSvgY(loss)}`;
    }
    return d;
  }

  // Draw GD points on left plot
  function drawGDPoints(pts) {
    pointsG.innerHTML = '';
    pts.forEach((p, i) => {
      if (i === 0 || i === pts.length - 1) return;
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', toSvgX(p[0]));
      circle.setAttribute('cy', toSvgY(p[1]));
      circle.setAttribute('r', 3);
      circle.setAttribute('fill', '#f78166');
      circle.setAttribute('opacity', 0.7);
      pointsG.appendChild(circle);
    });
  }

  // Draw loss points on right plot
  function drawLossPoints(losses, maxSteps) {
    lossPointsG.innerHTML = '';
    losses.forEach((loss, i) => {
      if (loss > lossMax * 10) return;
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', toLossSvgX(i, maxSteps));
      circle.setAttribute('cy', toLossSvgY(loss));
      circle.setAttribute('r', 3);
      circle.setAttribute('fill', '#f78166');
      circle.setAttribute('opacity', 0.7);
      lossPointsG.appendChild(circle);
    });
  }

  // Main update function
  function update() {
    const l1 = parseFloat(document.querySelector('[data-param="gd-lambda1"]').value);
    const l2 = parseFloat(document.querySelector('[data-param="gd-lambda2"]').value);
    const theta = parseFloat(document.querySelector('[data-param="theta"]').value);
    const lr = parseFloat(document.querySelector('[data-param="lr"]').value);
    const steps = parseInt(document.querySelector('[data-param="steps"]').value);

    // Update readouts
    document.querySelector('[data-readout="gd-lambda1"]').textContent = l1.toFixed(1);
    document.querySelector('[data-readout="gd-lambda2"]').textContent = l2.toFixed(1);
    document.querySelector('[data-readout="theta"]').textContent = theta + '°';
    document.querySelector('[data-readout="lr"]').textContent = lr.toFixed(2);
    document.querySelector('[data-readout="steps"]').textContent = steps;

    const A = buildA(l1, l2, theta);
    const eigs = eigenvalues(A);
    const lambdaMax = Math.max(...eigs);
    const etaCrit = 2 / lambdaMax;

    lambdaSpan.textContent = lambdaMax.toFixed(2);
    etaCritSpan.textContent = etaCrit.toFixed(2);

    // Update status
    if (lr < etaCrit) {
      statusSpan.textContent = 'Converging';
      statusSpan.style.color = '#3fb950';
    } else {
      statusSpan.textContent = 'Diverging!';
      statusSpan.style.color = '#f85149';
    }

    // Draw contours
    drawContours(l1, l2, theta);

    // Compute GD trajectory with losses
    const gdResult = computeGDPath(x0, y0, A, lr, steps);

    // Set loss scale based on initial loss
    const initialLoss = f(x0, y0, A);
    lossMax = Math.pow(10, Math.ceil(Math.log10(initialLoss * 2)));

    // Draw loss axes
    drawLossAxes(steps);

    // Draw left plot path
    gdPath.setAttribute('d', pointsToPath(gdResult.pts));
    drawGDPoints(gdResult.pts);

    // Draw right plot (loss) path
    lossGdPath.setAttribute('d', lossToPath(gdResult.losses, steps));
    drawLossPoints(gdResult.losses, steps);

    // Position start/end circles
    startCircle.setAttribute('cx', toSvgX(x0));
    startCircle.setAttribute('cy', toSvgY(y0));

    const lastGD = gdResult.pts[gdResult.pts.length - 1];
    if (Math.abs(lastGD[0]) < 50 && Math.abs(lastGD[1]) < 50) {
      endCircle.setAttribute('cx', toSvgX(lastGD[0]));
      endCircle.setAttribute('cy', toSvgY(lastGD[1]));
      endCircle.style.display = 'block';
    } else {
      endCircle.style.display = 'none';
    }
  }

  // Initialize
  drawGrid();
  drawAxes();
  update();

  // Event listeners for sliders
  document.querySelectorAll('#gd-widget input[type="range"]').forEach(input => {
    input.addEventListener('input', update);
  });

  // Reset button
  document.getElementById('gd-reset').addEventListener('click', () => {
    x0 = x0_default;
    y0 = y0_default;
    document.querySelector('[data-param="gd-lambda1"]').value = 4;
    document.querySelector('[data-param="gd-lambda2"]').value = 1;
    document.querySelector('[data-param="theta"]').value = 30;
    document.querySelector('[data-param="lr"]').value = 0.15;
    document.querySelector('[data-param="steps"]').value = 30;
    update();
  });

  // Click on left plot to set start point
  svg.addEventListener('click', (e) => {
    const rect = svg.getBoundingClientRect();
    const scaleX = totalW / rect.width;
    const scaleY = totalH / rect.height;
    const sx = (e.clientX - rect.left) * scaleX;
    const sy = (e.clientY - rect.top) * scaleY;

    // Only respond to clicks on left plot area
    if (sx > leftW) return;

    const newX = toDataX(sx);
    const newY = toDataY(sy);
    if (newX >= xMin && newX <= xMax && newY >= yMin && newY <= yMax) {
      x0 = newX;
      y0 = newY;
      update();
    }
  });

  svg.style.cursor = 'crosshair';
})();

// Loss and Sharpness widget (Figure 6 — Zipf NLLS on Hamlet word frequencies)
(function() {
  const W = 700, H = 460;
  const N_STEPS = 2500;
  const ns = 'http://www.w3.org/2000/svg';

  // Panel layout
  const mLeft = 65, mRight = 20;
  const lossTop = 30, lossBot = 205;
  const sharpTop = 240, sharpBot = 425;
  const xLeft = mLeft, xRight = W - mRight;
  const panelW = xRight - xLeft;

  const lossGridG   = document.getElementById('ls-loss-grid');
  const lossAxesG   = document.getElementById('ls-loss-axes');
  const lossPathEl  = document.getElementById('ls-loss-path');
  const sharpGridG  = document.getElementById('ls-sharp-grid');
  const sharpAxesG  = document.getElementById('ls-sharp-axes');
  const sharpPathEl = document.getElementById('ls-sharp-path');
  const critLineEl  = document.getElementById('ls-crit-line');
  const critLabelEl = document.getElementById('ls-crit-label');
  const infoDiv     = document.getElementById('ls-info');
  const etaSlider   = document.querySelector('[data-param="ls-eta"]');
  const etaReadout  = document.querySelector('[data-readout="ls-eta"]');

  let freqs = null, ranks = null, logRanks = null;

  function mk(tag, attrs) {
    const el = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }
  function mkTxt(text, attrs) { const el = mk('text', attrs); el.textContent = text; return el; }

  function sup(n) {
    const m = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻'};
    return String(n).split('').map(c => m[c]||c).join('');
  }

  // Single-pass: loss + gradient + Hessian + sharpness
  function computeAll(K, alpha) {
    const N = freqs.length;
    let lossAcc = 0, dK = 0, dA = 0, h11 = 0, h12 = 0, h22 = 0;
    for (let i = 0; i < N; i++) {
      const ra  = Math.pow(ranks[i], alpha);
      const lr  = logRanks[i];
      const yh  = K * ra;
      const res = freqs[i] - yh;       // freqs - yhat  (for loss & gradient)
      lossAcc += res * res;
      dK  -= ra * res;
      dA  -= lr * yh * res;
      h11 += ra * ra;
      h12 += K * ra * ra * lr - res * ra * lr;
      h22 += yh * lr * yh * lr - res * K * ra * lr * lr;
    }
    dK /= N; dA /= N; h11 /= N; h12 /= N; h22 /= N;
    const disc = Math.sqrt((h11 - h22) * (h11 - h22) + 4 * h12 * h12);
    return { loss: lossAcc / (2 * N), dK, dA, sharp: (h11 + h22 + disc) / 2 };
  }

  function runGD(eta) {
    let K = 0.0, alpha = -0.8;
    const losses = [], sharps = [];
    for (let s = 0; s <= N_STEPS; s++) {
      const { loss, dK, dA, sharp } = computeAll(K, alpha);
      losses.push(loss);
      sharps.push(sharp);
      if (s < N_STEPS) { K -= eta * dK; alpha -= eta * dA; }
    }
    return { losses, sharps };
  }

  function toX(step) { return xLeft + (step / N_STEPS) * panelW; }
  function toLossY(v, logMin, logMax) {
    const lv = Math.log10(Math.max(v, 1e-15));
    return lossBot - (Math.min(logMax, Math.max(logMin, lv)) - logMin) / (logMax - logMin) * (lossBot - lossTop);
  }
  const SHARP_MIN = 0.02, SHARP_MAX = 0.04;
  function toSharpY(v) { return sharpBot - (Math.min(SHARP_MAX, Math.max(SHARP_MIN, v)) - SHARP_MIN) / (SHARP_MAX - SHARP_MIN) * (sharpBot - sharpTop); }

  function drawLossPanel(losses, logMin, logMax) {
    lossGridG.innerHTML = ''; lossAxesG.innerHTML = '';
    [0,500,1000,1500,2000,2500].forEach(t => {
      lossGridG.appendChild(mk('line',{x1:toX(t),y1:lossTop,x2:toX(t),y2:lossBot,stroke:'#21262d','stroke-width':1}));
    });
    for (let e = Math.ceil(logMin); e <= Math.floor(logMax); e++) {
      const sy = toLossY(Math.pow(10, e), logMin, logMax);
      lossGridG.appendChild(mk('line',{x1:xLeft,y1:sy,x2:xRight,y2:sy,stroke:'#21262d','stroke-width':1}));
      lossAxesG.appendChild(mkTxt('10'+sup(e),{x:xLeft-4,y:sy+4,fill:'#6e7681','font-size':10,'text-anchor':'end'}));
    }
    lossAxesG.appendChild(mk('line',{x1:xLeft,y1:lossTop,x2:xLeft,y2:lossBot,stroke:'#484f58','stroke-width':1.5}));
    lossAxesG.appendChild(mk('line',{x1:xLeft,y1:lossBot,x2:xRight,y2:lossBot,stroke:'#484f58','stroke-width':1.5}));
    const midY = (lossTop + lossBot) / 2;
    lossAxesG.appendChild(mkTxt('Loss',{x:14,y:midY,fill:'#8b949e','font-size':12,'text-anchor':'middle',transform:`rotate(-90,14,${midY})`}));
    lossAxesG.appendChild(mkTxt('Loss vs Step',{x:xLeft+panelW/2,y:lossTop-8,fill:'#8b949e','font-size':12,'text-anchor':'middle'}));
    let d = '';
    for (let i = 0; i <= N_STEPS; i++) {
      const sx = toX(i), ly = toLossY(losses[i], logMin, logMax);
      d += (i ? ` L ${sx} ${ly}` : `M ${sx} ${ly}`);
    }
    lossPathEl.setAttribute('d', d);
  }

  function drawSharpPanel(sharps, critVal) {
    sharpGridG.innerHTML = ''; sharpAxesG.innerHTML = '';
    [0,500,1000,1500,2000,2500].forEach(t => {
      const sx = toX(t);
      sharpGridG.appendChild(mk('line',{x1:sx,y1:sharpTop,x2:sx,y2:sharpBot,stroke:'#21262d','stroke-width':1}));
      sharpAxesG.appendChild(mkTxt(t,{x:sx,y:sharpBot+14,fill:'#6e7681','font-size':10,'text-anchor':'middle'}));
    });
    for (let i = 0; i <= 4; i++) {
      const v = SHARP_MIN + (i / 4) * (SHARP_MAX - SHARP_MIN), sy = toSharpY(v);
      sharpGridG.appendChild(mk('line',{x1:xLeft,y1:sy,x2:xRight,y2:sy,stroke:'#21262d','stroke-width':1}));
      sharpAxesG.appendChild(mkTxt(v.toFixed(3),{x:xLeft-4,y:sy+4,fill:'#6e7681','font-size':10,'text-anchor':'end'}));
    }
    sharpAxesG.appendChild(mk('line',{x1:xLeft,y1:sharpTop,x2:xLeft,y2:sharpBot,stroke:'#484f58','stroke-width':1.5}));
    sharpAxesG.appendChild(mk('line',{x1:xLeft,y1:sharpBot,x2:xRight,y2:sharpBot,stroke:'#484f58','stroke-width':1.5}));
    const midY = (sharpTop + sharpBot) / 2;
    sharpAxesG.appendChild(mkTxt('Sharpness',{x:14,y:midY,fill:'#8b949e','font-size':12,'text-anchor':'middle',transform:`rotate(-90,14,${midY})`}));
    sharpAxesG.appendChild(mkTxt('Sharpness vs Step',{x:xLeft+panelW/2,y:sharpTop-8,fill:'#8b949e','font-size':12,'text-anchor':'middle'}));
    sharpAxesG.appendChild(mkTxt('Step',{x:xLeft+panelW/2,y:sharpBot+28,fill:'#8b949e','font-size':12,'text-anchor':'middle'}));
    let d = '';
    for (let i = 0; i <= N_STEPS; i++) {
      const sx = toX(i), sy = toSharpY(sharps[i]);
      d += (i ? ` L ${sx} ${sy}` : `M ${sx} ${sy}`);
    }
    sharpPathEl.setAttribute('d', d);
    // Critical stability line S = 2/η
    const csy = toSharpY(critVal);
    if (csy >= sharpTop && csy <= sharpBot) {
      critLineEl.setAttribute('x1', xLeft);  critLineEl.setAttribute('y1', csy);
      critLineEl.setAttribute('x2', xRight); critLineEl.setAttribute('y2', csy);
      critLineEl.setAttribute('stroke-opacity', 1);
      critLabelEl.setAttribute('x', xRight - 4); critLabelEl.setAttribute('y', csy - 4);
      critLabelEl.setAttribute('text-anchor', 'end');
      critLabelEl.setAttribute('visibility', 'visible');
      critLabelEl.textContent = `S=2/\u03b7\u2248${critVal.toFixed(4)}`;
    } else {
      critLineEl.setAttribute('stroke-opacity', 0);
      critLabelEl.setAttribute('visibility', 'hidden');
    }
  }

  function update() {
    if (!freqs) return;
    const eta = parseFloat(etaSlider.value);
    etaReadout.textContent = eta;
    const { losses, sharps } = runGD(eta);

    let logMin = Infinity, logMax = -Infinity;
    for (const v of losses) {
      if (v > 0) { const lv = Math.log10(v); if (lv < logMin) logMin = lv; if (lv > logMax) logMax = lv; }
    }
    logMin = Math.floor(logMin) - 1;
    logMax = Math.ceil(logMax);

    drawLossPanel(losses, logMin, logMax);
    drawSharpPanel(sharps, 2 / eta);
    infoDiv.innerHTML = `\u03b7 = ${eta} &nbsp;|&nbsp; 2/\u03b7 \u2248 ${(2/eta).toFixed(4)}`;
  }

  fetch('/assets/shared/data/zipf-hamlet.json')
    .then(r => r.json())
    .then(data => {
      freqs    = data.freqs;
      ranks    = freqs.map((_, i) => i + 1);
      logRanks = ranks.map(r => Math.log(r));
      update();
    });

  etaSlider.addEventListener('input', update);
})();

// Non-convex contour widget (Figure 5 — Zipf NLLS loss landscape)
(function() {
  const ns = 'http://www.w3.org/2000/svg';
  const W = 500, H = 400;
  const ML = 60, MR = 20, MT = 30, MB = 45;
  const PW = W - ML - MR, PH = H - MT - MB; // 420 x 325
  const K_MIN = -0.03, K_MAX = 0.07;
  const A_MIN = -1.0, A_MAX = -0.4;
  const LOG_MIN = -6, LOG_MAX = -4;
  const N_STEPS = 1500;
  const GN = 100;

  const VIR = [[68,1,84],[59,82,139],[33,144,140],[93,201,99],[253,231,37]];
  function viridis(t) {
    t = Math.max(0, Math.min(1, t));
    const s = t * 4, lo = Math.floor(s), hi = Math.min(lo+1, 4), f = s - lo;
    return `rgb(${Math.round(VIR[lo][0]+f*(VIR[hi][0]-VIR[lo][0]))},${Math.round(VIR[lo][1]+f*(VIR[hi][1]-VIR[lo][1]))},${Math.round(VIR[lo][2]+f*(VIR[hi][2]-VIR[lo][2]))})`;
  }

  function el(tag, attrs) {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }
  function txt(text, attrs) { const e = el('text', attrs); e.textContent = text; return e; }

  const cw = PW / GN, ch = PH / GN;
  function toX(K)     { return ML + (K - K_MIN) / (K_MAX - K_MIN) * PW; }
  function toY(alpha) { return MT + (A_MAX - alpha) / (A_MAX - A_MIN) * PH; }

  const N_SEGS = 20;

  const heatmapG  = document.getElementById('nc-heatmap');
  const axesG     = document.getElementById('nc-axes');
  const trajSegsG = document.getElementById('nc-traj-segs');
  const startCirc = document.getElementById('nc-start');
  const endCirc   = document.getElementById('nc-end');
  const etaSliderNc  = document.querySelector('[data-param="nc-eta"]');
  const etaReadoutNc = document.querySelector('[data-readout="nc-eta"]');
  const infoDivNc    = document.getElementById('nc-info');

  function lerpColor(c0, c1, t) {
    return `rgb(${Math.round(c0[0]+t*(c1[0]-c0[0]))},${Math.round(c0[1]+t*(c1[1]-c0[1]))},${Math.round(c0[2]+t*(c1[2]-c0[2]))})`;
  }

  let freqs = null, ranks = null, logRanks = null;

  function buildHeatmap() {
    const N = freqs.length;

    for (let gj = 0; gj < GN; gj++) {
      const alpha = A_MAX - ((gj + 0.5) / GN) * (A_MAX - A_MIN);
      const rAlpha = new Float64Array(N);
      for (let n = 0; n < N; n++) rAlpha[n] = Math.exp(logRanks[n] * alpha);

      for (let gi = 0; gi < GN; gi++) {
        const K = K_MIN + ((gi + 0.5) / GN) * (K_MAX - K_MIN);
        let loss = 0;
        for (let n = 0; n < N; n++) { const r = freqs[n] - K * rAlpha[n]; loss += r * r; }
        loss /= 2 * N;
        const t = Math.max(0, Math.min(1, (Math.log10(Math.max(loss, 1e-12)) - LOG_MIN) / (LOG_MAX - LOG_MIN)));
        heatmapG.appendChild(el('rect', {
          x: ML + gi * cw, y: MT + gj * ch,
          width: cw + 0.5, height: ch + 0.5,
          fill: viridis(t)
        }));
      }
    }

    // Axes
    axesG.appendChild(el('line', {x1:ML, y1:MT, x2:ML, y2:MT+PH, stroke:'#484f58', 'stroke-width':1.5}));
    axesG.appendChild(el('line', {x1:ML, y1:MT+PH, x2:ML+PW, y2:MT+PH, stroke:'#484f58', 'stroke-width':1.5}));
    [-0.02, 0, 0.02, 0.04, 0.06].forEach(K => {
      const x = toX(K);
      axesG.appendChild(el('line', {x1:x, y1:MT+PH, x2:x, y2:MT+PH+4, stroke:'#484f58', 'stroke-width':1}));
      axesG.appendChild(txt(K.toFixed(2), {x, y:MT+PH+14, fill:'#6e7681', 'font-size':10, 'text-anchor':'middle'}));
    });
    [-1.0, -0.8, -0.6, -0.4].forEach(a => {
      const y = toY(a);
      axesG.appendChild(el('line', {x1:ML, y1:y, x2:ML-4, y2:y, stroke:'#484f58', 'stroke-width':1}));
      axesG.appendChild(txt(a.toFixed(1), {x:ML-6, y:y+4, fill:'#6e7681', 'font-size':10, 'text-anchor':'end'}));
    });
    const midY = MT + PH/2;
    axesG.appendChild(txt('K', {x:ML+PW/2, y:MT+PH+30, fill:'#8b949e', 'font-size':12, 'text-anchor':'middle'}));
    axesG.appendChild(txt('\u03b1', {x:14, y:midY+4, fill:'#8b949e', 'font-size':12, 'text-anchor':'middle', transform:`rotate(-90,14,${midY})`}));
    axesG.appendChild(txt('Zipf Non-Negative Least Squares Loss', {x:ML+PW/2, y:MT-8, fill:'#8b949e', 'font-size':12, 'text-anchor':'middle'}));
  }

  function runGD(eta) {
    const N = freqs.length;
    let K = 0.0, alpha = -0.8;
    const traj = [[K, alpha]];
    for (let s = 0; s < N_STEPS; s++) {
      let dK = 0, dA = 0;
      for (let n = 0; n < N; n++) {
        const ra = Math.exp(logRanks[n] * alpha);
        const yh = K * ra;
        const res = freqs[n] - yh;
        dK -= ra * res;
        dA -= logRanks[n] * yh * res;
      }
      dK /= N; dA /= N;
      K -= eta * dK;
      alpha -= eta * dA;
      traj.push([K, alpha]);
    }
    return traj;
  }

  function update() {
    if (!freqs) return;
    const eta = parseFloat(etaSliderNc.value);
    etaReadoutNc.textContent = eta;

    const traj = runGD(eta);

    const [Kf, af] = traj[N_STEPS];
    let finalLoss = 0;
    for (let n = 0; n < freqs.length; n++) {
      const r = freqs[n] - Kf * Math.exp(logRanks[n] * af);
      finalLoss += r * r;
    }
    finalLoss /= 2 * freqs.length;
    infoDivNc.textContent = `\u03b7 = ${eta} \u00a0|\u00a0 final loss = ${finalLoss.toExponential(3)}`;

    trajSegsG.innerHTML = '';

    const stepsPerSeg = Math.ceil(N_STEPS / N_SEGS);
    const startColor = [63, 185, 80];   // #3fb950
    const endColor   = [247, 129, 102]; // #f78166
    for (let s = 0; s < N_SEGS; s++) {
      const t  = s / (N_SEGS - 1);
      const i0 = s * stepsPerSeg;
      const i1 = Math.min((s + 1) * stepsPerSeg, N_STEPS);
      let d = '';
      for (let i = i0; i <= i1; i++) {
        const x = toX(traj[i][0]), y = toY(traj[i][1]);
        d += (i === i0 ? `M${x} ${y}` : ` L${x} ${y}`);
      }
      trajSegsG.appendChild(el('path', {
        d, fill: 'none', stroke: lerpColor(startColor, endColor, t),
        'stroke-width': 1.5, 'stroke-linejoin': 'round'
      }));
    }

    startCirc.setAttribute('cx', toX(traj[0][0]));
    startCirc.setAttribute('cy', toY(traj[0][1]));
    endCirc.setAttribute('cx', toX(traj[N_STEPS][0]));
    endCirc.setAttribute('cy', toY(traj[N_STEPS][1]));
  }

  fetch('/assets/shared/data/zipf-hamlet.json')
    .then(r => r.json())
    .then(data => {
      freqs    = data.freqs;
      ranks    = freqs.map((_, i) => i + 1);
      logRanks = ranks.map(r => Math.log(r));
      buildHeatmap();
      update();
    });

  etaSliderNc.addEventListener('input', update);
})();

// Gradient Components widget (Figure 7 — Zipf NLLS eigenvector decomposition)
(function() {
  const W = 700, H = 480;
  const ETA = 70, N_STEPS = 750;
  const ns = 'http://www.w3.org/2000/svg';

  // Panel layout — shared x-axis
  const mLeft = 65, mRight = 20;
  const sharpTop = 30,  sharpBot = 210;
  const compTop  = 245, compBot  = 450;
  const xLeft = mLeft, xRight = W - mRight;
  const panelW = xRight - xLeft;

  const SHARP_MIN = 0.02, SHARP_MAX = 0.04;

  const sharpGridG  = document.getElementById('gc-sharp-grid');
  const sharpAxesG  = document.getElementById('gc-sharp-axes');
  const sharpPathEl = document.getElementById('gc-sharp-path');
  const critLineEl  = document.getElementById('gc-crit-line');
  const critLabelEl = document.getElementById('gc-crit-label');
  const compGridG   = document.getElementById('gc-comp-grid');
  const compAxesG   = document.getElementById('gc-comp-axes');
  const v1PathEl    = document.getElementById('gc-v1-path');
  const v2PathEl    = document.getElementById('gc-v2-path');

  let freqs = null, ranks = null, logRanks = null;

  function mk(tag, attrs) {
    const el = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }
  function mkTxt(text, attrs) { const el = mk('text', attrs); el.textContent = text; return el; }

  function computeAll(K, alpha) {
    const N = freqs.length;
    let dK = 0, dA = 0, h11 = 0, h12 = 0, h22 = 0;
    for (let i = 0; i < N; i++) {
      const ra  = Math.pow(ranks[i], alpha);
      const lr  = logRanks[i];
      const yh  = K * ra;
      const res = freqs[i] - yh;
      dK  -= ra * res;
      dA  -= lr * yh * res;
      h11 += ra * ra;
      h12 += K * ra * ra * lr - res * ra * lr;
      h22 += yh * lr * yh * lr - res * K * ra * lr * lr;
    }
    dK /= N; dA /= N; h11 /= N; h12 /= N; h22 /= N;
    const disc  = Math.sqrt((h11 - h22) * (h11 - h22) + 4 * h12 * h12);
    const sharp = (h11 + h22 + disc) / 2;

    let v1x, v1y;
    if (Math.abs(h12) > 1e-14) {
      const nx = sharp - h22, ny = h12;
      const norm = Math.sqrt(nx * nx + ny * ny);
      v1x = nx / norm; v1y = ny / norm;
    } else {
      v1x = h11 >= h22 ? 1 : 0;
      v1y = h11 >= h22 ? 0 : 1;
    }
    const v2x = -v1y, v2y = v1x;

    return { dK, dA, sharp, c1: dK * v1x + dA * v1y, c2: dK * v2x + dA * v2y };
  }

  function runGD() {
    let K = 0.0, alpha = -0.8;
    const sharps = [], c1s = [], c2s = [];
    for (let s = 0; s <= N_STEPS; s++) {
      const { dK, dA, sharp, c1, c2 } = computeAll(K, alpha);
      sharps.push(sharp); c1s.push(c1); c2s.push(c2);
      if (s < N_STEPS) { K -= ETA * dK; alpha -= ETA * dA; }
    }
    return { sharps, c1s, c2s };
  }

  function toX(step) { return xLeft + (step / N_STEPS) * panelW; }
  function toSharpY(v) {
    return sharpBot - (Math.min(SHARP_MAX, Math.max(SHARP_MIN, v)) - SHARP_MIN) / (SHARP_MAX - SHARP_MIN) * (sharpBot - sharpTop);
  }

  function drawSharpPanel(sharps) {
    sharpGridG.innerHTML = ''; sharpAxesG.innerHTML = '';
    const ticks = [0, 150, 300, 450, 600, 750];
    ticks.forEach(t => {
      sharpGridG.appendChild(mk('line',{x1:toX(t),y1:sharpTop,x2:toX(t),y2:sharpBot,stroke:'#21262d','stroke-width':1}));
    });
    for (let i = 0; i <= 4; i++) {
      const v = SHARP_MIN + (i / 4) * (SHARP_MAX - SHARP_MIN), sy = toSharpY(v);
      sharpGridG.appendChild(mk('line',{x1:xLeft,y1:sy,x2:xRight,y2:sy,stroke:'#21262d','stroke-width':1}));
      sharpAxesG.appendChild(mkTxt(v.toFixed(3),{x:xLeft-4,y:sy+4,fill:'#6e7681','font-size':10,'text-anchor':'end'}));
    }
    sharpAxesG.appendChild(mk('line',{x1:xLeft,y1:sharpTop,x2:xLeft,y2:sharpBot,stroke:'#484f58','stroke-width':1.5}));
    sharpAxesG.appendChild(mk('line',{x1:xLeft,y1:sharpBot,x2:xRight,y2:sharpBot,stroke:'#484f58','stroke-width':1.5}));
    const midY = (sharpTop + sharpBot) / 2;
    sharpAxesG.appendChild(mkTxt('Sharpness',{x:14,y:midY,fill:'#8b949e','font-size':12,'text-anchor':'middle',transform:`rotate(-90,14,${midY})`}));
    sharpAxesG.appendChild(mkTxt('Sharpness vs Step',{x:xLeft+panelW/2,y:sharpTop-8,fill:'#8b949e','font-size':12,'text-anchor':'middle'}));
    let d = '';
    for (let i = 0; i <= N_STEPS; i++) {
      const sx = toX(i), sy = toSharpY(sharps[i]);
      d += (i ? ` L ${sx} ${sy}` : `M ${sx} ${sy}`);
    }
    sharpPathEl.setAttribute('d', d);
    const critVal = 2 / ETA;
    const csy = toSharpY(critVal);
    if (csy >= sharpTop && csy <= sharpBot) {
      critLineEl.setAttribute('x1', xLeft);  critLineEl.setAttribute('y1', csy);
      critLineEl.setAttribute('x2', xRight); critLineEl.setAttribute('y2', csy);
      critLineEl.setAttribute('stroke-opacity', 1);
      critLabelEl.setAttribute('x', xLeft + 4); critLabelEl.setAttribute('y', csy - 4);
      critLabelEl.setAttribute('text-anchor', 'start');
      critLabelEl.setAttribute('visibility', 'visible');
      critLabelEl.textContent = `S=2/\u03b7\u2248${critVal.toFixed(4)}`;
    }
  }

  function drawCompPanel(c1s, c2s) {
    compGridG.innerHTML = ''; compAxesG.innerHTML = '';

    let maxAbs = 1e-10;
    for (let i = 0; i <= N_STEPS; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(c1s[i]), Math.abs(c2s[i]));
    }
    const cLim = maxAbs * 1.05;

    function toCompY(v) {
      return compBot - (Math.min(cLim, Math.max(-cLim, v)) + cLim) / (2 * cLim) * (compBot - compTop);
    }

    const ticks = [0, 150, 300, 450, 600, 750];
    ticks.forEach(t => {
      compGridG.appendChild(mk('line',{x1:toX(t),y1:compTop,x2:toX(t),y2:compBot,stroke:'#21262d','stroke-width':1}));
      compAxesG.appendChild(mkTxt(t,{x:toX(t),y:compBot+14,fill:'#6e7681','font-size':10,'text-anchor':'middle'}));
    });
    [-1, -0.5, 0, 0.5, 1].forEach(s => {
      const v = s * cLim, cy = toCompY(v);
      compGridG.appendChild(mk('line',{x1:xLeft,y1:cy,x2:xRight,y2:cy,stroke:'#21262d','stroke-width':1}));
      compAxesG.appendChild(mkTxt(v.toExponential(1),{x:xLeft-4,y:cy+4,fill:'#6e7681','font-size':10,'text-anchor':'end'}));
    });
    compAxesG.appendChild(mk('line',{x1:xLeft,y1:compTop,x2:xLeft,y2:compBot,stroke:'#484f58','stroke-width':1.5}));
    compAxesG.appendChild(mk('line',{x1:xLeft,y1:compBot,x2:xRight,y2:compBot,stroke:'#484f58','stroke-width':1.5}));
    const zy = toCompY(0);
    compAxesG.appendChild(mk('line',{x1:xLeft,y1:zy,x2:xRight,y2:zy,stroke:'#484f58','stroke-width':1,'stroke-dasharray':'3,3'}));
    const midY = (compTop + compBot) / 2;
    compAxesG.appendChild(mkTxt('\u2207f\u00b7v\u2096',{x:14,y:midY,fill:'#8b949e','font-size':12,'text-anchor':'middle',transform:`rotate(-90,14,${midY})`}));
    compAxesG.appendChild(mkTxt('Gradient Components vs Step',{x:xLeft+panelW/2,y:compTop-8,fill:'#8b949e','font-size':12,'text-anchor':'middle'}));
    compAxesG.appendChild(mkTxt('Step',{x:xLeft+panelW/2,y:compBot+28,fill:'#8b949e','font-size':12,'text-anchor':'middle'}));

    let d1 = '', d2 = '';
    for (let i = 0; i <= N_STEPS; i++) {
      const sx = toX(i);
      d1 += (i ? ` L ${sx} ${toCompY(c1s[i])}` : `M ${sx} ${toCompY(c1s[i])}`);
      d2 += (i ? ` L ${sx} ${toCompY(c2s[i])}` : `M ${sx} ${toCompY(c2s[i])}`);
    }
    v1PathEl.setAttribute('d', d1);
    v2PathEl.setAttribute('d', d2);
  }

  fetch('/assets/shared/data/zipf-hamlet.json')
    .then(r => r.json())
    .then(data => {
      freqs    = data.freqs;
      ranks    = freqs.map((_, i) => i + 1);
      logRanks = ranks.map(r => Math.log(r));
      const { sharps, c1s, c2s } = runGD();
      drawSharpPanel(sharps);
      drawCompPanel(c1s, c2s);
    });
})();

</script>