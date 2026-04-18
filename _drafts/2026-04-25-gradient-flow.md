---
title: "Gradient Flow and Gradient Descent"
categories:
  - Optimization
date: 2026-04-09 19:00:00 +0000
mathjax: true
tags:
  - Optimization
  - Gradient Descent
  - Machine Learning
toc: true
classes: wide
excerpt: "Understand the effect of sharpness on gradient descent dynamics."
---

This post explores the relationship between gradient flow (the continuous-time limit of gradient descent) and gradient descent, with a focus on how the learning rate and loss landscape curvature determine convergence behavior.

## Motivation

Given a differentiable loss function $$f: \mathbf{R}^d \to \mathbf{R}$$, gradient descent iteratively updates parameters according to

$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

where $$\eta > 0$$ is the learning rate (or step size).
When $$f$$ is complicated, as in the case of a neural network loss landscape, it's difficult to choose an appropriate value of $$\eta$$.
Choosing a value that's too small leads to very slow convergence while choosing a value that's too large leads to divergence.

To understand this, it helps to first study a simple case to understand the source of the instability.

## 1D Quadratic Case

Consider the simplest case of minimizing the 1D quadratic $$f(x) = \frac{S}{2}x^2$$ where $$S > 0$$ is the **sharpness** (curvature) of the function. Larger $$S$$ means a steeper, narrower parabola as shown in Figure 1.

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

The gradient is $$\nabla f(x) = Sx$$, so gradient descent becomes

$$x_{k+1} = x_k - \eta \cdot S x_k = (1 - S\eta) x_k$$

This is a simple geometric sequence! Starting from $$x_0$$, we have $$x_k = (1-S\eta)^k x_0.$$

For convergence to zero, we need 

$$\lvert 1 - S\eta\rvert < 1$$

Since $$S, \eta > 0$$, this means:

$$0 < S\eta < 2 \quad \Rightarrow \quad \eta < \frac{2}{S}$$

This means that for a quadratic function, the maximum stable learning rate is inversely proportional to the sharpness.
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

## The n-D Quadratic Case

In higher dimensions, the picture becomes richer. Consider an $$n$$-D quadratic

$$f(x) = \frac{1}{2}x^T A x$$

where $$A\in \mathbf{S}_n^{+}$$, the gradient of which is $$\nabla f = Ax$$.[^grad-general]
We can form the eigenvalue decomposition of the quadratic form as 

$$A = V\mathbf{diag}(\lambda_1, \ldots, \lambda_n)V^T$$

where $$V^TV = I$$.

The quadratic form can then be written as

$$f(x) = \frac{1}{2} \sum_{k=1}^n \lambda_k \cdot (x^Tv_k)^2$$

where $$v_k$$ is the $$k^{th}$$ column of $$V$$.
Comparing this decomposition to our 1D case, we can see that the eigenvalues are _exactly_ the sharpnesses of the quadratic along the directions $$v_1, \ldots, v_n$$ (called the principal axes).[^sharpness-proof]

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
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 3: Level sets (contours) of the 2D quadratic f(x) = ½(λ₁x₁² + λ₂x₂²). When λ₁ = λ₂ the contours are circles; unequal eigenvalues produce ellipses elongated along the less-sharp direction.</figcaption>
</div>

In the 1D quadratic case, we derived the simple rule that $$\eta < 2/S$$ for gradient descent to converge.
For the $$n$$-D quadratic case, how should we set the learning rate to ensure convergence?

Figure 4 shows the trajectory of gradient descent on our quadratic.

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
</div>

From the figure, we can see that convergence occurs when
$$
\eta < \frac{2}{\max\{\lambda_1, \lambda_2\}}.
$$
For the general $$n$$-D case, $$\eta < \frac{2}{\max\{\lambda_1, \ldots, \lambda_n\}}.$$

This says the maximal learning rate with which gradient descent can converge for a quadratic is governed by the sharpness along each principal axis.
More specifically, it is determined by the _sharpest_ of these directions.
Since the principal axis with the largest sharpness is the only one that governs convergence, we define to $$S= \max\{\lambda_1, \ldots, \lambda_n\}$$ as the sharpness of a quadratic in the general case.

## Gradient Flow

Gradient descent is an inherently discrete algorithm.
By taking a finite step size $$\eta$$ in the direction of the negative gradient, we are not taking the path of steepest descent since the gradient changes as soon as we move any finite distance from our current iterate.

In gradient descent, the update rule is

$$x_{k+1} = x_k - \eta \nabla f(x_k).$$

If we parametrise the path by the continuous variable $$t$$ rather than the discrete index $$k$$, the update rule becomes

$$x(t+h) = x(t) - h \nabla_x f(x(t)).$$

Rearranging these terms and taking the limit as $$h\rightarrow 0$$ gives the differential equation for the trajectory $$x(t)$$

$$\frac{\mathrm{d}f(x(t))}{\mathrm{d}t} = \nabla_x f(x(t)).$$

The solution to this differential equation $$x(t)$$ is called the _gradient flow_.
For the case of a quadratic form, the equation is

$$
\frac{\mathrm{d}f(x(t))}{\mathrm{d}t} = -Ax(t)
$$

which has solution $$x(t) = e^{tA}x(0)$$.[^mat-exp]
Because $$A = VDV^T$$, the gradient flow solution can be written as 

$$x(t) = V\mathbf{diag}(e^{-\lambda_1 t},\ldots, e^{-\lambda_n t})V^Tx(0) = \sum_{k=1}^n e^{-\lambda_k t}(v_k^Tx(0)) v_k.$$

Figure 5 illustrates the gradient flow trajectory for a 2D quadratic.
Each component decays exponentially at its own eigenvalue rate, so high-sharpness directions vanish first.



<div class="widget-container" id="gf-widget" style="max-width: 900px;">
  <div class="widget-controls">
    <label>
      λ₁
      <input type="range" min="0.5" max="8" value="4" step="0.5" data-param="gf-lambda1">
      <span class="widget-readout" data-readout="gf-lambda1">4.0</span>
    </label>
    <label>
      λ₂
      <input type="range" min="0.5" max="8" value="1" step="0.5" data-param="gf-lambda2">
      <span class="widget-readout" data-readout="gf-lambda2">1.0</span>
    </label>
    <label>
      θ
      <input type="range" min="0" max="90" value="30" step="5" data-param="gf-theta">
      <span class="widget-readout" data-readout="gf-theta">30°</span>
    </label>
    <button type="button" class="widget-button" id="gf-reset">Reset</button>
  </div>
  <svg class="widget-plot" id="gf-svg" viewBox="0 0 880 420" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="880" height="420" fill="#0d1117"></rect>
    <g id="gf-grid"></g>
    <g id="gf-contours"></g>
    <g id="gf-axes"></g>
    <path id="gf-path" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round"></path>
    <circle id="gf-start" r="7" fill="#3fb950" stroke="#0d1117" stroke-width="2"></circle>
    <g id="gf-loss-grid"></g>
    <g id="gf-loss-axes"></g>
    <path id="gf-loss-path" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round"></path>
  </svg>
  <div class="widget-legend">
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#58a6ff"></span>Gradient flow</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#3fb950"></span>Start (click to move)</span>
  </div>
  <div class="widget-info" id="gf-info"></div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 5: Gradient flow trajectory on a 2D quadratic (left) and the corresponding loss vs. continuous time t (right). Each eigenvector component decays independently at rate λᵢ — the high-curvature direction (larger λ) vanishes first.</figcaption>
</div>

Unlike gradient descent, gradient flow cannot diverge for a quadratic.

## Non-convex Case

Introduce new level sets for non-linear problem. Show trajectory of gradient descent. Propose a second order taylor approx to measure sharpness at each point along the trajectory.
The gradient flow solution is

$$x(t) = e^{-At}x_0$$

which traces a smooth curve toward the origin. In contrast, gradient descent with a fixed learning rate takes discrete steps that may overshoot along the high-curvature directions.

## Loss vs Step: The Effect of Learning Rate

The learning rate $$\eta$$ dramatically affects convergence behavior. The critical threshold is

$$\eta_{\text{crit}} = \frac{2}{\lambda_{\max}}$$

where $$\lambda_{\max}$$ is the largest eigenvalue of $$2A$$ (i.e., twice the sharpness along the steepest direction).

- For $$\eta < \eta_{\text{crit}}$$: Loss monotonically decreases
- For $$\eta > \eta_{\text{crit}}$$: Iterates diverge to infinity

## Interactive Exploration

The following interactive visualization lets you explore how the ellipse parameters ($$a$$, $$b$$, $$\theta$$) and learning rate ($$\eta$$) affect gradient descent behavior. The blue curve shows gradient flow (the continuous limit), while red points show gradient descent iterates.

Key observations to make:
- When $$a = b$$, the contours are circular and gradient descent moves directly toward the origin
- When $$a \neq b$$ (elliptical contours), gradient descent can exhibit "zigzag" behavior
- Larger $$\eta$$ causes more aggressive steps that may overshoot the minimum
- The rotation angle $$\theta$$ changes the principal axes but not the convergence rate

## Connection to Sharpness in Deep Learning

In deep learning, the "sharpness" of the loss landscape—measured by the largest eigenvalue of the Hessian—plays a crucial role in:

1. **Determining maximum learning rate**: The stability bound $$\eta < 2/S$$ applies locally
2. **Generalization**: Flatter minima are often associated with better generalization
3. **Adaptive methods**: Algorithms like Adam implicitly adapt to local curvature

Recent research has shown that during training, neural networks often operate near the "edge of stability" where $$\eta \approx 2/S$$, with the sharpness dynamically adjusting to maintain this balance.

## Conclusion

Gradient flow provides a clean theoretical framework for understanding gradient descent:

- The continuous limit removes discretization artifacts
- Stability analysis reveals the critical role of sharpness
- The maximum learning rate is $$\eta_{\max} = 2/\lambda_{\max}$$

For quadratic functions, gradient flow always converges while gradient descent requires careful step size selection. In the non-quadratic case, these insights apply locally near critical points, explaining why adaptive learning rate methods are so effective in practice.

## Try It Yourself

Explore gradient descent dynamics with this interactive widget. Adjust the quadratic's shape (semi-axes $$a$$ and $$b$$), rotation angle $$\theta$$, starting point, and learning rate to see how they affect convergence.
**Tips:**

- Click anywhere on the left plot to set a new starting point
- The right plot shows loss vs step on a log scale—watch how the loss decays (or explodes when diverging)
- When $$\eta > \eta_{\text{crit}}$$, gradient descent diverges (loss explodes upward)
- Try setting $$a = b$$ to see circular contours where GD moves directly toward the origin
- Notice how gradient flow always takes the smooth optimal path while GD can overshoot

[^grad-general]: More precisely, $$\nabla_x \tfrac{1}{2}x^T A x = \tfrac{1}{2}(A + A^T)x$$. When $$A$$ is symmetric this reduces to $$Ax$$.

[^mat-exp]: The matrix exponential is defined by the same Taylor series as the scalar exponential, applied entry-wise to powers of the matrix: $$e^{tA} = \sum_{k=0}^{\infty} \frac{(tA)^k}{k!} = I + tA + \frac{t^2 A^2}{2!} + \frac{t^3 A^3}{3!} + \cdots$$. This series converges for any square matrix $$A$$ and any scalar $$t$$. For symmetric $$A = V\mathbf{diag}(\lambda_1,\ldots,\lambda_n)V^T$$ the result simplifies to $$e^{tA} = V\,\mathbf{diag}(e^{t\lambda_1},\ldots,e^{t\lambda_n})\,V^T$$, which makes the connection to eigenvalues explicit.

[^sharpness-proof]: To see why, set $$x = t\,v_k$$ for scalar $$t$$. Because the columns of $$V$$ are orthonormal, $$x^T v_j = t\,v_k^T v_j = t\,\delta_{kj}$$, so every term in the sum vanishes except the $$k^{th}$$ one: $$f(tv_k) = \tfrac{1}{2}\lambda_k t^2$$. This is exactly the 1D quadratic with sharpness $$\lambda_k$$, confirming that $$\lambda_k$$ governs the curvature of $$f$$ along $$v_k$$.

## References

1. [Gradient Flow](https://en.wikipedia.org/wiki/Gradient_descent#Gradient_flow)
2. Cohen, J., et al. "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability." ICLR 2021.
3. Boyd, S. & Vandenberghe, L. "Convex Optimization." Cambridge University Press, 2004.

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
    const lMax = Math.max(l1, l2), lMin = Math.min(l1, l2);
    const scaleX = plotW / (xMax - xMin);
    const scaleY = plotH / (yMax - yMin);
    const levels = [0.5, 1, 2, 3, 5, 8, 12];

    levels.forEach(c => {
      // Semi-axes: larger semi-axis along the less-sharp (smaller eigenvalue) direction
      const rx = Math.sqrt(2 * c / lMin) * scaleX;
      const ry = Math.sqrt(2 * c / lMax) * scaleY;
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
  function drawContours(A, thetaDeg) {
    contoursG.innerHTML = '';
    const eigs = eigenvalues(A);
    const l1 = Math.max(...eigs), l2 = Math.min(...eigs);
    const levels = [0.5, 1, 2, 3, 5, 8, 12];

    levels.forEach(c => {
      const ax = Math.sqrt(2 * c / l2);
      const ay = Math.sqrt(2 * c / l1);
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
    drawContours(A, theta);

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

// Gradient flow widget (Figure 5)
(function() {
  'use strict';

  // Layout — identical to gd-widget so plots align visually
  const totalW = 880, totalH = 420;
  const leftW = 420, rightW = 400, rightX = 480, margin = 40;
  const xMin = -3.5, xMax = 3.5, yMin = -3.5, yMax = 3.5;
  const T_MAX = 5;   // continuous time horizon
  const N_PTS = 300; // samples along the flow curve

  let x0 = 2.5, y0 = 2.0;
  const x0_default = 2.5, y0_default = 2.0;

  // ── Coordinate transforms (left plot) ─────────────────────────────────────
  const toSvgX  = x  => margin + (x - xMin) / (xMax - xMin) * (leftW - 2 * margin);
  const toSvgY  = y  => totalH - margin - (y - yMin) / (yMax - yMin) * (totalH - 2 * margin);
  const toDataX = sx => xMin + (sx - margin) / (leftW - 2 * margin) * (xMax - xMin);
  const toDataY = sy => yMin + (totalH - margin - sy) / (totalH - 2 * margin) * (yMax - yMin);

  // ── Loss plot transforms (right plot, x = continuous time t) ──────────────
  let lossMax = 10;
  const toLossSvgX = t    => rightX + margin + (t / T_MAX) * (rightW - 2 * margin);
  const toLossSvgY = loss => {
    const logL = Math.log10(Math.max(loss, 1e-8));
    const logMax = Math.log10(lossMax), logMin = -8;
    return totalH - margin - (logL - logMin) / (logMax - logMin) * (totalH - 2 * margin);
  };

  // ── Math helpers (same logic as gd-widget) ────────────────────────────────
  const buildA = (l1, l2, thetaDeg) => {
    const t = thetaDeg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
    return [[c*c*l1 + s*s*l2, c*s*(l1-l2)], [c*s*(l1-l2), s*s*l1 + c*c*l2]];
  };
  const quadLoss = (x, y, A) => 0.5 * (A[0][0]*x*x + 2*A[0][1]*x*y + A[1][1]*y*y);

  // Exact gradient flow via eigendecomposition: x(t) = Σ e^{-λk t} (vk·x0) vk
  // With A = R diag(l1,l2) Rᵀ, columns of R are v1=[c,s], v2=[-s,c]
  const computeFlow = (px, py, l1, l2, thetaDeg) => {
    const th = thetaDeg * Math.PI / 180, c = Math.cos(th), s = Math.sin(th);
    const u1 =  c*px + s*py;  // projection onto v1
    const u2 = -s*px + c*py;  // projection onto v2
    const pts = [], losses = [];
    for (let i = 0; i <= N_PTS; i++) {
      const t  = T_MAX * i / N_PTS;
      const e1 = Math.exp(-l1 * t), e2 = Math.exp(-l2 * t);
      pts.push([e1*u1*c + e2*u2*(-s), e1*u1*s + e2*u2*c]);
      losses.push(0.5 * (l1*u1*u1*e1*e1 + l2*u2*u2*e2*e2));
    }
    return { pts, losses };
  };

  // ── SVG helpers ───────────────────────────────────────────────────────────
  const NS = 'http://www.w3.org/2000/svg';
  const mkEl  = (tag, attrs) => { const e = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([k,v]) => e.setAttribute(k, v)); return e; };
  const mkTxt = (text, attrs) => { const e = mkEl('text', attrs); e.textContent = text; return e; };
  const sup   = n => { const d = ['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹']; return (n<0?'⁻':'') + String(Math.abs(n)).split('').map(c=>d[+c]).join(''); };

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const gridG     = document.getElementById('gf-grid');
  const axesG     = document.getElementById('gf-axes');
  const contoursG = document.getElementById('gf-contours');
  const flowPath  = document.getElementById('gf-path');
  const startDot  = document.getElementById('gf-start');
  const lossGridG = document.getElementById('gf-loss-grid');
  const lossAxesG = document.getElementById('gf-loss-axes');
  const lossPath  = document.getElementById('gf-loss-path');
  const infoDiv   = document.getElementById('gf-info');
  const svg       = document.getElementById('gf-svg');

  // ── Static left-plot grid & axes (drawn once) ─────────────────────────────
  function drawGrid() {
    for (let v = Math.ceil(xMin); v <= Math.floor(xMax); v++)
      gridG.appendChild(mkEl('line', { x1: toSvgX(v), y1: margin, x2: toSvgX(v), y2: totalH - margin, stroke: '#21262d', 'stroke-width': 1 }));
    for (let v = Math.ceil(yMin); v <= Math.floor(yMax); v++)
      gridG.appendChild(mkEl('line', { x1: margin, y1: toSvgY(v), x2: leftW - margin, y2: toSvgY(v), stroke: '#21262d', 'stroke-width': 1 }));
  }

  function drawAxes() {
    axesG.appendChild(mkEl('line', { x1: margin, y1: toSvgY(0), x2: leftW - margin, y2: toSvgY(0), stroke: '#484f58', 'stroke-width': 1.5 }));
    axesG.appendChild(mkEl('line', { x1: toSvgX(0), y1: margin, x2: toSvgX(0), y2: totalH - margin, stroke: '#484f58', 'stroke-width': 1.5 }));
    for (let v = Math.ceil(xMin); v <= Math.floor(xMax); v++)
      if (v !== 0) axesG.appendChild(mkTxt(v, { x: toSvgX(v), y: totalH - margin + 18, fill: '#6e7681', 'font-size': 11, 'text-anchor': 'middle' }));
    for (let v = Math.ceil(yMin); v <= Math.floor(yMax); v++)
      if (v !== 0) axesG.appendChild(mkTxt(v, { x: margin - 8, y: toSvgY(v) + 4, fill: '#6e7681', 'font-size': 11, 'text-anchor': 'end' }));
    axesG.appendChild(mkTxt('x₁', { x: (margin + leftW - margin) / 2, y: totalH - 5, fill: '#8b949e', 'font-size': 12, 'text-anchor': 'middle' }));
    const yl = mkTxt('x₂', { x: 12, y: totalH / 2, fill: '#8b949e', 'font-size': 12, 'text-anchor': 'middle' });
    yl.setAttribute('transform', `rotate(-90, 12, ${totalH / 2})`);
    axesG.appendChild(yl);
  }

  // ── Dynamic: contours, flow path, loss plot ───────────────────────────────
  function drawContours(l1, l2, thetaDeg) {
    contoursG.innerHTML = '';
    const lBig = Math.max(l1, l2), lSml = Math.min(l1, l2);
    const sx = (leftW - 2*margin) / (xMax - xMin), sy = (totalH - 2*margin) / (yMax - yMin);
    [0.5, 1, 2, 3, 5, 8, 12].forEach(c => {
      const rx = Math.sqrt(2*c / lSml) * sx, ry = Math.sqrt(2*c / lBig) * sy;
      if (rx > leftW || ry > totalH) return;
      contoursG.appendChild(mkEl('ellipse', {
        cx: toSvgX(0), cy: toSvgY(0), rx, ry,
        transform: `rotate(${-thetaDeg}, ${toSvgX(0)}, ${toSvgY(0)})`,
        fill: 'none', stroke: '#30363d', 'stroke-width': 1
      }));
    });
  }

  function drawLossAxes() {
    lossGridG.innerHTML = ''; lossAxesG.innerHTML = '';
    for (let ti = 0; ti <= T_MAX; ti++) {
      const sx = toLossSvgX(ti);
      lossGridG.appendChild(mkEl('line', { x1: sx, y1: margin, x2: sx, y2: totalH - margin, stroke: '#21262d', 'stroke-width': 1 }));
      lossAxesG.appendChild(mkTxt(ti, { x: sx, y: totalH - margin + 18, fill: '#6e7681', 'font-size': 11, 'text-anchor': 'middle' }));
    }
    const logMax = Math.ceil(Math.log10(lossMax));
    for (let exp = -8; exp <= logMax; exp += 2) {
      const sy = toLossSvgY(Math.pow(10, exp));
      if (sy < margin || sy > totalH - margin) continue;
      lossGridG.appendChild(mkEl('line', { x1: rightX + margin, y1: sy, x2: rightX + rightW - margin, y2: sy, stroke: '#21262d', 'stroke-width': 1 }));
      lossAxesG.appendChild(mkTxt('10' + sup(exp), { x: rightX + margin - 8, y: sy + 4, fill: '#6e7681', 'font-size': 11, 'text-anchor': 'end' }));
    }
    lossAxesG.appendChild(mkEl('line', { x1: rightX + margin, y1: totalH - margin, x2: rightX + rightW - margin, y2: totalH - margin, stroke: '#484f58', 'stroke-width': 1.5 }));
    lossAxesG.appendChild(mkEl('line', { x1: rightX + margin, y1: margin, x2: rightX + margin, y2: totalH - margin, stroke: '#484f58', 'stroke-width': 1.5 }));
    lossAxesG.appendChild(mkTxt('t', { x: rightX + margin + (rightW - 2*margin) / 2, y: totalH - 5, fill: '#8b949e', 'font-size': 12, 'text-anchor': 'middle' }));
    const yl = mkTxt('f(x(t))', { x: rightX + 12, y: totalH / 2, fill: '#8b949e', 'font-size': 12, 'text-anchor': 'middle' });
    yl.setAttribute('transform', `rotate(-90, ${rightX + 12}, ${totalH / 2})`);
    lossAxesG.appendChild(yl);
  }

  // ── Main update ───────────────────────────────────────────────────────────
  function update() {
    const l1    = parseFloat(document.querySelector('[data-param="gf-lambda1"]').value);
    const l2    = parseFloat(document.querySelector('[data-param="gf-lambda2"]').value);
    const theta = parseFloat(document.querySelector('[data-param="gf-theta"]').value);
    document.querySelector('[data-readout="gf-lambda1"]').textContent = l1.toFixed(1);
    document.querySelector('[data-readout="gf-lambda2"]').textContent = l2.toFixed(1);
    document.querySelector('[data-readout="gf-theta"]').textContent   = theta + '°';

    const A = buildA(l1, l2, theta);
    lossMax = Math.pow(10, Math.ceil(Math.log10(quadLoss(x0, y0, A) * 2)));

    const { pts, losses } = computeFlow(x0, y0, l1, l2, theta);

    drawContours(l1, l2, theta);
    drawLossAxes();

    // Flow path on left plot
    flowPath.setAttribute('d', pts.reduce((d, [px, py], i) =>
      d + (i === 0 ? `M ${toSvgX(px)} ${toSvgY(py)}` : ` L ${toSvgX(px)} ${toSvgY(py)}`), ''));

    // Loss curve on right plot
    lossPath.setAttribute('d', losses.reduce((d, loss, i) =>
      d + (i === 0 ? `M ${toLossSvgX(0)} ${toLossSvgY(loss)}` : ` L ${toLossSvgX(T_MAX * i / N_PTS)} ${toLossSvgY(loss)}`), ''));

    startDot.setAttribute('cx', toSvgX(x0));
    startDot.setAttribute('cy', toSvgY(y0));

    infoDiv.innerHTML = `λ<sub>1</sub> = ${l1.toFixed(1)} &nbsp;|&nbsp; λ<sub>2</sub> = ${l2.toFixed(1)} &nbsp;|&nbsp; f(x(0)) = ${quadLoss(x0, y0, A).toFixed(3)} &nbsp;|&nbsp; f(x(5)) = ${losses[losses.length-1].toExponential(2)}`;
  }

  drawGrid();
  drawAxes();
  update();

  document.querySelectorAll('#gf-widget input[type="range"]').forEach(el => el.addEventListener('input', update));

  document.getElementById('gf-reset').addEventListener('click', () => {
    x0 = x0_default; y0 = y0_default;
    document.querySelector('[data-param="gf-lambda1"]').value = 4;
    document.querySelector('[data-param="gf-lambda2"]').value = 1;
    document.querySelector('[data-param="gf-theta"]').value   = 30;
    update();
  });

  svg.style.cursor = 'crosshair';
  svg.addEventListener('click', e => {
    const rect = svg.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * (totalW / rect.width);
    const sy = (e.clientY - rect.top)  * (totalH / rect.height);
    if (sx > leftW) return;
    const nx = toDataX(sx), ny = toDataY(sy);
    if (nx >= xMin && nx <= xMax && ny >= yMin && ny <= yMax) { x0 = nx; y0 = ny; update(); }
  });
})();
</script>