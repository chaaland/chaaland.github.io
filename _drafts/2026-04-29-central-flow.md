---
title: "Central Flow"
categories:
  - Optimization
date: 2026-04-29 19:00:00 +0000
mathjax: true
tags:
  - Optimization
  - Gradient Descent
  - Machine Learning
toc: true
classes: wide
excerpt: "The central flow as the moving average of gradient descent iterates near the edge of stability."
---

## Gradient Flow

Gradient descent is an inherently discrete algorithm.
By taking a finite step size $$\eta$$ in the direction of the negative gradient, we are not taking the path of steepest descent since the gradient changes as soon as we move any finite distance from our current iterate.

In gradient descent, the update rule is

$$x_{k+1} = x_k - \eta \nabla f(x_k).$$

If we parametrize the path by the continuous variable $$t$$ rather than the discrete index $$k$$, the update rule becomes

$$x(t+h) = x(t) - h \nabla f(x(t)).$$

Rearranging these terms and taking the limit as $$h\rightarrow 0$$ gives the differential equation for the trajectory $$x(t)$$

$$\frac{\mathrm{d}x(t)}{\mathrm{d}t} = -\nabla_x f(x(t)).$$

The solution to this differential equation $$x(t)$$ is called the _gradient flow_.[^velocity]
For the case of a quadratic form, the equation is

$$
\frac{\mathrm{d}x(t)}{\mathrm{d}t} = -Ax(t)
$$

which has solution $$x(t) = e^{-tA}x(0)$$.[^mat-exp]
Because $$A = VDV^T$$, the gradient flow solution can be written as

$$x(t) = V\mathbf{diag}(e^{-\lambda_1 t},\ldots, e^{-\lambda_n t})V^Tx(0) = \sum_{k=1}^n e^{-\lambda_k t}(v_k^Tx(0)) v_k.$$

From this representation, we can see that each component decays exponentially at its own eigenvalue rate, which leads to high-sharpness directions vanishing first.
Figure 1 illustrates the gradient flow trajectory for a 2D quadratic.

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
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 1: Gradient flow trajectory on a 2D quadratic (left) and the corresponding loss vs. continuous time t (right). Each eigenvector component decays independently at rate λᵢ — the high-curvature direction (larger λ) vanishes first.</figcaption>
</div>

Unlike gradient descent, gradient flow cannot diverge for a quadratic.

## Smoothed Gradient Descent

Unlike gradient flow, which glides smoothly along each eigendirection toward the minimum, gradient descent oscillates back and forth across the high-curvature direction near the stability limit $$2/\lambda_{\max}$$.
Figure 2 shows this directly on both a quadratic (left) and a more complicated non-linear least squares objective (right).

<div class="widget-container" id="eos-nlls-widget" style="max-width: 900px;">
  <svg class="widget-plot" id="eos-nlls-svg" viewBox="0 0 880 440" preserveAspectRatio="xMidYMid meet">
    <defs>
      <clipPath id="quad-clip">
        <rect x="50" y="30" width="380" height="360"></rect>
      </clipPath>
      <clipPath id="nlls2-clip">
        <rect x="490" y="30" width="380" height="360"></rect>
      </clipPath>
    </defs>
    <rect x="0" y="0" width="880" height="440" fill="#0d1117"></rect>
    <g id="quad-grid"></g>
    <g id="quad-contours" clip-path="url(#quad-clip)"></g>
    <g id="quad-gd" clip-path="url(#quad-clip)"></g>
    <g id="quad-axes"></g>
    <g id="quad-markers" clip-path="url(#quad-clip)"></g>
    <g id="nlls2-grid"></g>
    <g id="nlls2-contours" clip-path="url(#nlls2-clip)"></g>
    <g id="nlls2-gd" clip-path="url(#nlls2-clip)"></g>
    <g id="nlls2-axes"></g>
    <g id="nlls2-markers" clip-path="url(#nlls2-clip)"></g>
  </svg>
  <div class="widget-legend">
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:linear-gradient(to right,#3fb950,#f78166)"></span>Gradient descent (green = early, orange = late)</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#3fb950"></span>Start</span>
  </div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 2: Gradient descent near the edge of stability on a quadratic (left, λ₁ = 8, λ₂ = 0.5, η = 1.9/λ₁ ≈ 0.95 × 2/λ₁, 30 steps) and a non-convex objective (right, η = 75, 400 steps). Color runs green → orange with increasing step number. In both cases the iterates oscillate back and forth across the high-curvature direction rather than descending smoothly toward the minimum.</figcaption>
</div>

The analysis in the [previous post on loss spikes](/loss-spikes/) explains _why_ loss spikes occur during gradient descent.
They occur when the sharpness of the loss landscape temporarily exceeds $$2/\eta$$ but then, due to the relationship of the hessian of the gradient to the gradient of the sharpness, the iterates are pushed in the direction of decreasing-sharpness.

However, from Figure 2 it is clear that gradient descent not only has an oscillatory component in the direction of maximum sharpness, but also a drift component towards the minimum.

To get the slower, more steady, drift component, we can average out the oscillations by taking a windowed average of the gradient descent iterates

$$\bar{x}_k = \frac{1}{W}\sum_{j=k-W+1}^{k} x_j.$$

Figure 3 overlays the window averaged trajectory (W = 2 for the quadratic, W = 20 for the NLLS) on the same parameter-space plots as Figure 2.

<div class="widget-container" id="cf-traj-widget" style="max-width: 900px;">
  <svg class="widget-plot" id="cf-traj-svg" viewBox="0 0 880 440" preserveAspectRatio="xMidYMid meet">
    <defs>
      <clipPath id="cfq-clip">
        <rect x="50" y="30" width="380" height="360"></rect>
      </clipPath>
      <clipPath id="cfn-clip">
        <rect x="490" y="30" width="380" height="360"></rect>
      </clipPath>
    </defs>
    <rect x="0" y="0" width="880" height="440" fill="#0d1117"></rect>
    <g id="cfq-grid"></g>
    <g id="cfq-contours" clip-path="url(#cfq-clip)"></g>
    <g id="cfq-gd" clip-path="url(#cfq-clip)" opacity="0.3"></g>
    <g id="cfq-cf" clip-path="url(#cfq-clip)"></g>
    <g id="cfq-axes"></g>
    <g id="cfq-markers" clip-path="url(#cfq-clip)"></g>
    <g id="cfn-grid"></g>
    <g id="cfn-contours" clip-path="url(#cfn-clip)"></g>
    <g id="cfn-gd" clip-path="url(#cfn-clip)" opacity="0.3"></g>
    <g id="cfn-cf" clip-path="url(#cfn-clip)"></g>
    <g id="cfn-axes"></g>
    <g id="cfn-markers" clip-path="url(#cfn-clip)"></g>
  </svg>
  <div class="widget-legend">
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:linear-gradient(to right,#3fb950,#f78166)"></span>Gradient descent (green = early, orange = late)</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:linear-gradient(to right,#58a6ff,#a371f7)"></span>Central flow (blue = early, purple = late)</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#3fb950"></span>Start</span>
  </div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 3: The same gradient descent trajectories from Figure 2, with the central flow (blue, windowed average) overlaid. Left: quadratic (W = 2, λ₁ = 8, λ₂ = 0.5, η ≈ 0.24, 30 steps). Right: NLLS objective (W = 20, η = 75, 400 steps). The trajectory of the average iterates cuts through the oscillations and drifts steadily toward the minimum.</figcaption>
</div>

This figure shows that by averaging over $$W$$ consecutive iterates, the oscillations along the sharpest eigendirection cancel out, revealing a smoother underlying trajectory that resembles a sort of discrete analog of gradient flow.

Figure 4 shows the same dynamic in loss space for the NLLS objective. A larger window size produces a smoother loss curve. A larger learning rate produces more dramatic spikes that the averaging must absorb.

<div class="widget-container" id="cf-widget">
  <div class="widget-controls">
    <label>
      η (learning rate)
      <input type="range" min="1" max="100" value="70" step="1" data-param="cf-eta">
      <span class="widget-readout" data-readout="cf-eta">70</span>
    </label>
    <label>
      W (window)
      <input type="range" min="2" max="50" value="10" step="2" data-param="cf-window">
      <span class="widget-readout" data-readout="cf-window">10</span>
    </label>
  </div>
  <svg class="widget-plot" id="cf-svg" viewBox="0 0 700 340" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="700" height="340" fill="#0d1117"></rect>
    <g id="cf-grid"></g>
    <g id="cf-axes"></g>
    <path id="cf-gd-path" fill="none" stroke="#f78166" stroke-width="1.5" stroke-linejoin="round" opacity="0.4"></path>
    <path id="cf-flow-path" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linejoin="round"></path>
  </svg>
  <div class="widget-legend">
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#f78166"></span>Gradient descent</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#58a6ff"></span>Central flow</span>
  </div>
  <div class="widget-info" id="cf-info">η = 70 &nbsp;|&nbsp; W = 10</div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 4: Loss (log scale) for gradient descent (orange, 40% opacity) and central flow (blue) on the power law NLLS objective for 1500 steps. The central flow is the trailing moving average of GD iterates with window W — it absorbs oscillations near the edge of stability, revealing the smooth underlying trajectory.</figcaption>
</div>

## Central Flow

In the first section we were able to derive the gradient flow as the continuous limit of gradient descent.
Analogously, we'll derive a continuous version of our windowed average gradient descent trajectory called the _central flow_.

Assuming we only have a single eigenvalue at the edge of stability, we can model our central flow $$x(t)$$ as being composed of a slow moving component $$x_c(t)$$ and a rapidly oscillating component $$u(t)$$ in the direction of highest sharpness

$$x(t) = x_c(t) + \delta_t u(t).$$

From the [previous post on loss spikes](/loss-spikes/), the gradient in a region of high sharpness can be approximated by

$$
\nabla f (x_c(t)+\delta_t u(t)) \approx \nabla f(x_c(t)) + \delta_t S(x_c(t))u(t) + \frac{\delta_t^2}{2}\nabla S(x_c(t)).
$$

Taking the time average over a window $$W$$, we have

$$
{1 \over W} \sum_{k=0}^{W-1} \nabla f (x(t-k)) \approx {1 \over W} \sum_{k=0}^{W-1} \left[\nabla f(x_c(t-k)) + \delta_{t-k} S(x_c(t-k))u(t-k) + \frac{\delta_{t-k}^2}{2}\nabla S(x_c(t-k))\right].
$$

Since $$x_c(t)$$ is slow moving, if $$W$$ is not too large, then $$x_c(t) \approx x_c(t-W+1)$$ and $$u(t) \approx u(t-W-1)$$.
The first term reduces to

$${1 \over W} \sum_{k=0}^{W-1} \nabla f(x_c(t-k)) \approx \nabla f(x_c(t)).$$

The second term becomes

$$
\begin{align*}
{1 \over W} \sum_{k=0}^{W-1} \delta_{t-k} S(x_c(t-k))u(t-k) &\approx  S(x_c(t)) u(t)\cdot {1 \over W}\sum_{k=0}^{W-1} \delta_t \\
&\approx 0 \\
\end{align*}
$$

since $$\delta_t$$ is oscillating rapidly and has 0 mean.
The final term becomes

$$
\begin{align*}
{1 \over W} \sum_{k=0}^{W-1} \frac{\delta_{t-k}^2}{2}\nabla S(x_c(t-k)) &\approx \nabla S(x_c(t))\cdot {1 \over W} \sum_{k=0}^{W-1} \frac{\delta_{t-k}^2}{2} \\
&\approx \nabla S(x_c(t)) {\sigma_t^2 \over 2}. \\
\end{align*}
$$

where $$\sigma_t$$ denotes the second moment of the oscillation magnitude.
The windowed average gradient becomes

$$
{1 \over W} \sum_{k=0}^{W-1} \nabla f (x(t-k)) \approx \nabla f(x_c(t)) + {\sigma_t^2 \over 2} \nabla S(x_c(t)).
$$

While the gradient flow is governed by

$$
\frac{\mathrm{d}x(t)}{\mathrm{d}t} = -\nabla f(x(t))
$$

the central flow is governed by 

$$
\frac{\mathrm{d}x_c(t)}{\mathrm{d}t} = -\left(\nabla f(x_c(t)) + {\sigma_t^2 \over 2} \nabla S(x_c(t))\right).
$$

This is very nearly the same as the central flow except 

- it is only valid when past the critical threshold $$2/\eta$$
- it includes a term that steers the velocity vector in the direction of decreasing sharpness

To find the second moment $$\sigma_t^2$$, we assert that the derivative of the sharpness, $$\mathrm{d}S/\mathrm{d}t$$ is 0


[^velocity]: Equivalently, the velocity vector $$\dot{x}(t)$$ at every point along the trajectory equals the negative gradient of $$f$$ at that point. The particle always moves in the direction of steepest descent.

[^mat-exp]: The matrix exponential is defined by the same Taylor series as the scalar exponential, applied entry-wise to powers of the matrix: $$e^{tA} = \sum_{k=0}^{\infty} \frac{(tA)^k}{k!} = I + tA + \frac{t^2 A^2}{2!} + \frac{t^3 A^3}{3!} + \cdots$$. This series converges for any square matrix $$A$$ and any scalar $$t$$. For symmetric $$A = V\mathbf{diag}(\lambda_1,\ldots,\lambda_n)V^T$$ the result simplifies to $$e^{tA} = V\,\mathbf{diag}(e^{t\lambda_1},\ldots,e^{t\lambda_n})\,V^T$$, which makes the connection to eigenvalues explicit.

[^warmup]: For the first $$W-1$$ steps ($$k < W-1$$), there are fewer than $$W$$ iterates available, so the average is taken over all $$k+1$$ iterates seen so far.

[^cf]: Formally studied in Cohen et al. "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability" (ICLR 2021) and further analyzed in Damian et al. "Self-stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability" (NeurIPS 2023).

<script>
// Gradient flow widget (Figure 1)
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
    const sx = (leftW - 2*margin) / (xMax - xMin), sy = (totalH - 2*margin) / (yMax - yMin);
    [0.5, 1, 2, 3, 5, 8, 12].forEach(c => {
      // l1 governs v1=[cos θ, sin θ] (horizontal before rotation), l2 governs v2
      const rx = Math.sqrt(2*c / l1) * sx, ry = Math.sqrt(2*c / l2) * sy;
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

// Combined EOS figure (Figure 2)
(function() {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';

  // Two 440-wide panels in an 880x440 SVG
  const mL = 50, mR = 10, mT = 30, mB = 50;
  const PW = 380, PH = 360;
  const Q_OX = 0, N_OX = 440;

  // Quadratic parameters
  const xMin = -2.5, xMax = 2.5, yMin = -2.5, yMax = 2.5;
  const L1 = 8, L2 = 0.5;
  const THETA_DEG = 30;
  const TH = THETA_DEG * Math.PI / 180;
  const C = Math.cos(TH), S = Math.sin(TH);
  const ETA_Q = 1.9 / L1;
  const N_STEPS_Q = 30;
  const X0_Q = [2.0, -1.5];
  const A = [
    [L1*C*C + L2*S*S,   (L1 - L2)*C*S],
    [(L1 - L2)*C*S,      L1*S*S + L2*C*C]
  ];

  // NLLS parameters
  const K_MIN = -0.03, K_MAX = 0.07;
  const A_MIN = -1.0,  A_MAX = -0.4;
  const N_GRID = 60;
  const X0_N = [0.0, -0.8];
  const ETA_N = 75;
  const N_STEPS_N = 400;

  function el(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }
  function txt(content, attrs) { const e = el('text', attrs); e.textContent = content; return e; }

  // Quadratic coordinate transforms
  const qSX = x => Q_OX + mL + (x - xMin) / (xMax - xMin) * PW;
  const qSY = y => mT + (yMax - y) / (yMax - yMin) * PH;

  // NLLS coordinate transforms
  const nSX = K => N_OX + mL + (K - K_MIN) / (K_MAX - K_MIN) * PW;
  const nSY = a => mT + PH * (1 - (a - A_MIN) / (A_MAX - A_MIN));
  const nColSX = c => N_OX + mL + (c / (N_GRID - 1)) * PW;
  const nRowSY = r => mT + PH * (1 - r / (N_GRID - 1));

  const qGridG = document.getElementById('quad-grid');
  const qContG = document.getElementById('quad-contours');
  const qGdG   = document.getElementById('quad-gd');
  const qAxesG = document.getElementById('quad-axes');
  const qMarkG = document.getElementById('quad-markers');
  const nGridG = document.getElementById('nlls2-grid');
  const nContG = document.getElementById('nlls2-contours');
  const nGdG   = document.getElementById('nlls2-gd');
  const nAxesG = document.getElementById('nlls2-axes');
  const nMarkG = document.getElementById('nlls2-markers');

  // Precompute quadratic GD iterates
  function gdStepQ(x) {
    return [
      x[0] - ETA_Q * (A[0][0]*x[0] + A[0][1]*x[1]),
      x[1] - ETA_Q * (A[1][0]*x[0] + A[1][1]*x[1])
    ];
  }
  const gdPtsQ = [X0_Q.slice()];
  for (let k = 0; k < N_STEPS_Q; k++) gdPtsQ.push(gdStepQ(gdPtsQ[k]));

  // Green (#3fb950) at step 0 → orange (#f78166) at step N, matching the start marker
  const GD_C0 = [63, 185, 80];
  const GD_C1 = [247, 129, 102];
  function lerpColor(t) {
    const r = Math.round(GD_C0[0] + t * (GD_C1[0] - GD_C0[0]));
    const g = Math.round(GD_C0[1] + t * (GD_C1[1] - GD_C0[1]));
    const b = Math.round(GD_C0[2] + t * (GD_C1[2] - GD_C0[2]));
    return `rgb(${r},${g},${b})`;
  }

  function drawQuad() {
    for (let v = -2; v <= 2; v++) {
      qGridG.appendChild(el('line', {x1:qSX(v), y1:mT, x2:qSX(v), y2:mT+PH, stroke:'#21262d', 'stroke-width':1}));
      qGridG.appendChild(el('line', {x1:Q_OX+mL, y1:qSY(v), x2:Q_OX+mL+PW, y2:qSY(v), stroke:'#21262d', 'stroke-width':1}));
    }
    qAxesG.appendChild(el('rect', {x:Q_OX+mL, y:mT, width:PW, height:PH, fill:'none', stroke:'#484f58', 'stroke-width':1}));
    qAxesG.appendChild(el('line', {x1:Q_OX+mL, y1:qSY(0), x2:Q_OX+mL+PW, y2:qSY(0), stroke:'#484f58', 'stroke-width':1.5}));
    qAxesG.appendChild(el('line', {x1:qSX(0), y1:mT, x2:qSX(0), y2:mT+PH, stroke:'#484f58', 'stroke-width':1.5}));
    for (let v = -2; v <= 2; v++) {
      if (v !== 0) {
        qAxesG.appendChild(el('line', {x1:qSX(v), y1:mT+PH, x2:qSX(v), y2:mT+PH+4, stroke:'#8b949e', 'stroke-width':1}));
        qAxesG.appendChild(txt(String(v), {x:qSX(v), y:mT+PH+16, fill:'#6e7681', 'font-size':11, 'text-anchor':'middle'}));
        qAxesG.appendChild(el('line', {x1:Q_OX+mL-4, y1:qSY(v), x2:Q_OX+mL, y2:qSY(v), stroke:'#8b949e', 'stroke-width':1}));
        qAxesG.appendChild(txt(String(v), {x:Q_OX+mL-8, y:qSY(v)+4, fill:'#6e7681', 'font-size':11, 'text-anchor':'end'}));
      }
    }
    qAxesG.appendChild(txt('x₁', {x:Q_OX+mL+PW/2, y:mT+PH+32, fill:'#8b949e', 'font-size':12, 'text-anchor':'middle'}));
    const qYl = txt('x₂', {x:Q_OX+14, y:mT+PH/2, fill:'#8b949e', 'font-size':12, 'text-anchor':'middle'});
    qYl.setAttribute('transform', `rotate(-90,${Q_OX+14},${mT+PH/2})`);
    qAxesG.appendChild(qYl);

    const scX = PW / (xMax - xMin), scY = PH / (yMax - yMin);
    const cx0 = qSX(0), cy0 = qSY(0);
    [0.25, 0.6, 1.4, 3.2, 7.0, 14.0].forEach(c => {
      qContG.appendChild(el('ellipse', {
        cx: cx0, cy: cy0,
        rx: Math.sqrt(2*c/L1) * scX, ry: Math.sqrt(2*c/L2) * scY,
        transform: `rotate(${-THETA_DEG},${cx0},${cy0})`,
        fill: 'none', stroke: '#30363d', 'stroke-width': 1
      }));
    });

    for (let i = 0; i < gdPtsQ.length - 1; i++) {
      const stroke = lerpColor(i / (gdPtsQ.length - 2));
      qGdG.appendChild(el('line', {x1:qSX(gdPtsQ[i][0]), y1:qSY(gdPtsQ[i][1]), x2:qSX(gdPtsQ[i+1][0]), y2:qSY(gdPtsQ[i+1][1]), stroke, 'stroke-width':1.5, 'stroke-linecap':'round'}));
    }
    gdPtsQ.forEach(([x, y], i) =>
      qGdG.appendChild(el('circle', {cx:qSX(x), cy:qSY(y), r:2.5, fill:lerpColor(i / (gdPtsQ.length - 1))})));
    qMarkG.appendChild(el('circle', {cx:qSX(X0_Q[0]), cy:qSY(X0_Q[1]), r:7, fill:'#3fb950', stroke:'#0d1117', 'stroke-width':2}));
  }

  function zipfLoss(K, a, freqs) {
    const N = freqs.length;
    let s = 0;
    for (let i = 0; i < N; i++) {
      const res = freqs[i] - K * Math.pow(i + 1, a);
      s += res * res;
    }
    return s / (2 * N);
  }

  function zipfGrad(K, a, freqs) {
    const N = freqs.length;
    let dK = 0, dA = 0;
    for (let i = 0; i < N; i++) {
      const r = i + 1;
      const rA = Math.pow(r, a);
      const res = freqs[i] - K * rA;
      dK -= rA * res;
      dA -= Math.log(r) * K * rA * res;
    }
    return [dK / N, dA / N];
  }

  function marchingSquares(Z, level) {
    const segs = [];
    function lerp(va, vb, pa, pb) {
      const t = (level - va) / (vb - va);
      return [pa[0] + t*(pb[0]-pa[0]), pa[1] + t*(pb[1]-pa[1])];
    }
    for (let r = 0; r < N_GRID - 1; r++) {
      for (let c = 0; c < N_GRID - 1; c++) {
        const v00=Z[r][c], v01=Z[r][c+1], v11=Z[r+1][c+1], v10=Z[r+1][c];
        const a00=v00>=level, a01=v01>=level, a11=v11>=level, a10=v10>=level;
        if (a00===a01 && a01===a11 && a11===a10) continue;
        const pts = [];
        if (a00!==a01) pts.push(lerp(v00, v01, [r,c],   [r,c+1]));
        if (a01!==a11) pts.push(lerp(v01, v11, [r,c+1], [r+1,c+1]));
        if (a11!==a10) pts.push(lerp(v11, v10, [r+1,c+1], [r+1,c]));
        if (a10!==a00) pts.push(lerp(v10, v00, [r+1,c], [r,c]));
        if (pts.length === 2) segs.push(pts);
        else if (pts.length === 4) { segs.push([pts[0],pts[1]]); segs.push([pts[2],pts[3]]); }
      }
    }
    return segs;
  }

  function drawNLLS(freqs) {
    const lossGrid = Array.from({length: N_GRID}, (_, r) => {
      const a = A_MIN + r / (N_GRID - 1) * (A_MAX - A_MIN);
      return Array.from({length: N_GRID}, (_, c) => {
        const K = K_MIN + c / (N_GRID - 1) * (K_MAX - K_MIN);
        return Math.log10(Math.max(zipfLoss(K, a, freqs), 1e-12));
      });
    });

    [-0.02, 0.0, 0.02, 0.04, 0.06].forEach(K =>
      nGridG.appendChild(el('line', {x1:nSX(K), y1:mT, x2:nSX(K), y2:mT+PH, stroke:'#21262d', 'stroke-width':1})));
    [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4].forEach(a =>
      nGridG.appendChild(el('line', {x1:N_OX+mL, y1:nSY(a), x2:N_OX+mL+PW, y2:nSY(a), stroke:'#21262d', 'stroke-width':1})));

    nAxesG.appendChild(el('rect', {x:N_OX+mL, y:mT, width:PW, height:PH, fill:'none', stroke:'#484f58', 'stroke-width':1}));
    nAxesG.appendChild(el('line', {x1:nSX(0), y1:mT, x2:nSX(0), y2:mT+PH, stroke:'#484f58', 'stroke-width':1.5}));

    [-0.02, 0.0, 0.02, 0.04, 0.06].forEach(K => {
      nAxesG.appendChild(el('line', {x1:nSX(K), y1:mT+PH, x2:nSX(K), y2:mT+PH+4, stroke:'#8b949e', 'stroke-width':1}));
      nAxesG.appendChild(txt(K.toFixed(2), {x:nSX(K), y:mT+PH+16, fill:'#6e7681', 'font-size':10, 'text-anchor':'middle'}));
    });
    [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4].forEach(a => {
      nAxesG.appendChild(el('line', {x1:N_OX+mL-4, y1:nSY(a), x2:N_OX+mL, y2:nSY(a), stroke:'#8b949e', 'stroke-width':1}));
      nAxesG.appendChild(txt(a.toFixed(1), {x:N_OX+mL-8, y:nSY(a)+4, fill:'#6e7681', 'font-size':10, 'text-anchor':'end'}));
    });

    nAxesG.appendChild(txt('K', {x:N_OX+mL+PW/2, y:mT+PH+32, fill:'#8b949e', 'font-size':13, 'text-anchor':'middle'}));
    const nYl = txt('α', {x:N_OX+14, y:mT+PH/2, fill:'#8b949e', 'font-size':13, 'text-anchor':'middle'});
    nYl.setAttribute('transform', `rotate(-90,${N_OX+14},${mT+PH/2})`);
    nAxesG.appendChild(nYl);

    let logMin = Infinity, logMax = -Infinity;
    for (const row of lossGrid) for (const v of row) { if (v < logMin) logMin = v; if (v > logMax) logMax = v; }
    const N_LEVELS = 14;
    for (let i = 0; i < N_LEVELS; i++) {
      const level = logMin + (i / (N_LEVELS - 1)) * (logMax - logMin);
      for (const [p1, p2] of marchingSquares(lossGrid, level)) {
        nContG.appendChild(el('line', {
          x1: nColSX(p1[1]).toFixed(1), y1: nRowSY(p1[0]).toFixed(1),
          x2: nColSX(p2[1]).toFixed(1), y2: nRowSY(p2[0]).toFixed(1),
          stroke: '#30363d', 'stroke-width': 0.8, 'stroke-linecap': 'round'
        }));
      }
    }

    let [K, a] = X0_N;
    const gdPtsN = [[K, a]];
    for (let s = 0; s < N_STEPS_N; s++) {
      const [dK, dA] = zipfGrad(K, a, freqs);
      K -= ETA_N * dK;
      a -= ETA_N * dA;
      if (!isFinite(K) || !isFinite(a)) break;
      gdPtsN.push([K, a]);
    }

    for (let i = 0; i < gdPtsN.length - 1; i++) {
      const stroke = lerpColor(i / Math.max(1, gdPtsN.length - 2));
      nGdG.appendChild(el('line', {x1:nSX(gdPtsN[i][0]).toFixed(1), y1:nSY(gdPtsN[i][1]).toFixed(1), x2:nSX(gdPtsN[i+1][0]).toFixed(1), y2:nSY(gdPtsN[i+1][1]).toFixed(1), stroke, 'stroke-width':1.5, 'stroke-linecap':'round'}));
    }
    nMarkG.appendChild(el('circle', {cx:nSX(X0_N[0]).toFixed(1), cy:nSY(X0_N[1]).toFixed(1), r:7, fill:'#3fb950', stroke:'#0d1117', 'stroke-width':2}));
  }

  drawQuad();
  fetch('/assets/shared/data/zipf-hamlet.json')
    .then(r => r.json())
    .then(data => drawNLLS(data.freqs));
})();


// Central flow trajectory figure (Figure 3)
(function() {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';

  const mL = 50, mR = 10, mT = 30, mB = 50;
  const PW = 380, PH = 360;
  const Q_OX = 0, N_OX = 440;

  // Quadratic parameters (same as Figure 2)
  const xMin = -2.5, xMax = 2.5, yMin = -2.5, yMax = 2.5;
  const L1 = 8, L2 = 0.5;
  const THETA_DEG = 30;
  const TH = THETA_DEG * Math.PI / 180;
  const C = Math.cos(TH), S = Math.sin(TH);
  const ETA_Q = 1.9 / L1;
  const N_STEPS_Q = 30;
  const X0_Q = [2.0, -1.5];
  const A = [
    [L1*C*C + L2*S*S,   (L1-L2)*C*S],
    [(L1-L2)*C*S,        L1*S*S+L2*C*C]
  ];
  const W_Q = 2;

  // NLLS parameters (same as Figure 2)
  const K_MIN = -0.03, K_MAX = 0.07;
  const A_MIN = -1.0,  A_MAX = -0.4;
  const N_GRID = 60;
  const X0_N = [0.0, -0.8];
  const ETA_N = 75;
  const N_STEPS_N = 400;
  const W_N = 20;

  function el(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }
  function txt(content, attrs) { const e = el('text', attrs); e.textContent = content; return e; }

  const qSX = x => Q_OX + mL + (x - xMin) / (xMax - xMin) * PW;
  const qSY = y => mT + (yMax - y) / (yMax - yMin) * PH;
  const nSX = K => N_OX + mL + (K - K_MIN) / (K_MAX - K_MIN) * PW;
  const nSY = a => mT + PH * (1 - (a - A_MIN) / (A_MAX - A_MIN));
  const nColSX = c => N_OX + mL + (c / (N_GRID - 1)) * PW;
  const nRowSY = r => mT + PH * (1 - r / (N_GRID - 1));

  const GD_C0 = [63, 185, 80], GD_C1 = [247, 129, 102];
  function lerpColor(t) {
    const r = Math.round(GD_C0[0] + t*(GD_C1[0]-GD_C0[0]));
    const g = Math.round(GD_C0[1] + t*(GD_C1[1]-GD_C0[1]));
    const b = Math.round(GD_C0[2] + t*(GD_C1[2]-GD_C0[2]));
    return `rgb(${r},${g},${b})`;
  }

  const CF_C0 = [88, 166, 255], CF_C1 = [163, 113, 247];
  function lerpCfColor(t) {
    const r = Math.round(CF_C0[0] + t*(CF_C1[0]-CF_C0[0]));
    const g = Math.round(CF_C0[1] + t*(CF_C1[1]-CF_C0[1]));
    const b = Math.round(CF_C0[2] + t*(CF_C1[2]-CF_C0[2]));
    return `rgb(${r},${g},${b})`;
  }

  function windowedAvg(pts, W) {
    return pts.map((_, k) => {
      const start = Math.max(0, k - W + 1);
      const slice = pts.slice(start, k + 1);
      const n = slice.length;
      return slice.reduce((a, b) => [a[0]+b[0], a[1]+b[1]], [0, 0]).map(v => v / n);
    });
  }

  // Precompute quadratic GD iterates
  const gdPtsQ = [X0_Q.slice()];
  for (let k = 0; k < N_STEPS_Q; k++) {
    const [x, y] = gdPtsQ[k];
    gdPtsQ.push([
      x - ETA_Q*(A[0][0]*x + A[0][1]*y),
      y - ETA_Q*(A[1][0]*x + A[1][1]*y)
    ]);
  }
  const cfPtsQ = windowedAvg(gdPtsQ, W_Q);

  function drawQuad() {
    const qGridG = document.getElementById('cfq-grid');
    const qContG = document.getElementById('cfq-contours');
    const qGdG   = document.getElementById('cfq-gd');
    const qCfG   = document.getElementById('cfq-cf');
    const qAxesG = document.getElementById('cfq-axes');
    const qMarkG = document.getElementById('cfq-markers');

    for (let v = -2; v <= 2; v++) {
      qGridG.appendChild(el('line', {x1:qSX(v), y1:mT, x2:qSX(v), y2:mT+PH, stroke:'#21262d', 'stroke-width':1}));
      qGridG.appendChild(el('line', {x1:Q_OX+mL, y1:qSY(v), x2:Q_OX+mL+PW, y2:qSY(v), stroke:'#21262d', 'stroke-width':1}));
    }

    qAxesG.appendChild(el('rect', {x:Q_OX+mL, y:mT, width:PW, height:PH, fill:'none', stroke:'#484f58', 'stroke-width':1}));
    qAxesG.appendChild(el('line', {x1:Q_OX+mL, y1:qSY(0), x2:Q_OX+mL+PW, y2:qSY(0), stroke:'#484f58', 'stroke-width':1.5}));
    qAxesG.appendChild(el('line', {x1:qSX(0), y1:mT, x2:qSX(0), y2:mT+PH, stroke:'#484f58', 'stroke-width':1.5}));
    for (let v = -2; v <= 2; v++) {
      if (v !== 0) {
        qAxesG.appendChild(el('line', {x1:qSX(v), y1:mT+PH, x2:qSX(v), y2:mT+PH+4, stroke:'#8b949e', 'stroke-width':1}));
        qAxesG.appendChild(txt(String(v), {x:qSX(v), y:mT+PH+16, fill:'#6e7681', 'font-size':11, 'text-anchor':'middle'}));
        qAxesG.appendChild(el('line', {x1:Q_OX+mL-4, y1:qSY(v), x2:Q_OX+mL, y2:qSY(v), stroke:'#8b949e', 'stroke-width':1}));
        qAxesG.appendChild(txt(String(v), {x:Q_OX+mL-8, y:qSY(v)+4, fill:'#6e7681', 'font-size':11, 'text-anchor':'end'}));
      }
    }
    qAxesG.appendChild(txt('x₁', {x:Q_OX+mL+PW/2, y:mT+PH+32, fill:'#8b949e', 'font-size':12, 'text-anchor':'middle'}));
    const qYl = txt('x₂', {x:Q_OX+14, y:mT+PH/2, fill:'#8b949e', 'font-size':12, 'text-anchor':'middle'});
    qYl.setAttribute('transform', `rotate(-90,${Q_OX+14},${mT+PH/2})`);
    qAxesG.appendChild(qYl);

    const cx0 = qSX(0), cy0 = qSY(0);
    const scX = PW / (xMax - xMin), scY = PH / (yMax - yMin);
    [0.25, 0.6, 1.4, 3.2, 7.0, 14.0].forEach(c => {
      qContG.appendChild(el('ellipse', {
        cx: cx0, cy: cy0,
        rx: Math.sqrt(2*c/L1) * scX, ry: Math.sqrt(2*c/L2) * scY,
        transform: `rotate(${-THETA_DEG},${cx0},${cy0})`,
        fill: 'none', stroke: '#30363d', 'stroke-width': 1
      }));
    });

    for (let i = 0; i < gdPtsQ.length - 1; i++) {
      const stroke = lerpColor(i / (gdPtsQ.length - 2));
      qGdG.appendChild(el('line', {x1:qSX(gdPtsQ[i][0]), y1:qSY(gdPtsQ[i][1]), x2:qSX(gdPtsQ[i+1][0]), y2:qSY(gdPtsQ[i+1][1]), stroke, 'stroke-width':1.5, 'stroke-linecap':'round'}));
    }
    gdPtsQ.forEach(([x, y], i) =>
      qGdG.appendChild(el('circle', {cx:qSX(x), cy:qSY(y), r:2.5, fill:lerpColor(i/(gdPtsQ.length-1))})));

    for (let i = 0; i < cfPtsQ.length - 1; i++) {
      const stroke = lerpCfColor(i / (cfPtsQ.length - 2));
      qCfG.appendChild(el('line', {x1:qSX(cfPtsQ[i][0]), y1:qSY(cfPtsQ[i][1]), x2:qSX(cfPtsQ[i+1][0]), y2:qSY(cfPtsQ[i+1][1]), stroke, 'stroke-width':2.5, 'stroke-linecap':'round'}));
    }

    qMarkG.appendChild(el('circle', {cx:qSX(X0_Q[0]), cy:qSY(X0_Q[1]), r:7, fill:'#3fb950', stroke:'#0d1117', 'stroke-width':2}));
  }

  function zipfLoss(K, a, freqs) {
    const N = freqs.length;
    let s = 0;
    for (let i = 0; i < N; i++) { const res = freqs[i] - K*Math.pow(i+1, a); s += res*res; }
    return s / (2*N);
  }

  function zipfGrad(K, a, freqs) {
    const N = freqs.length;
    let dK = 0, dA = 0;
    for (let i = 0; i < N; i++) {
      const r = i+1, rA = Math.pow(r, a), res = freqs[i]-K*rA;
      dK -= rA*res; dA -= Math.log(r)*K*rA*res;
    }
    return [dK/N, dA/N];
  }

  function marchingSquares(Z, level) {
    const segs = [];
    function lerp(va, vb, pa, pb) {
      const t = (level-va)/(vb-va);
      return [pa[0]+t*(pb[0]-pa[0]), pa[1]+t*(pb[1]-pa[1])];
    }
    for (let r = 0; r < N_GRID-1; r++) {
      for (let c = 0; c < N_GRID-1; c++) {
        const v00=Z[r][c], v01=Z[r][c+1], v11=Z[r+1][c+1], v10=Z[r+1][c];
        const a00=v00>=level, a01=v01>=level, a11=v11>=level, a10=v10>=level;
        if (a00===a01 && a01===a11 && a11===a10) continue;
        const pts = [];
        if (a00!==a01) pts.push(lerp(v00,v01,[r,c],[r,c+1]));
        if (a01!==a11) pts.push(lerp(v01,v11,[r,c+1],[r+1,c+1]));
        if (a11!==a10) pts.push(lerp(v11,v10,[r+1,c+1],[r+1,c]));
        if (a10!==a00) pts.push(lerp(v10,v00,[r+1,c],[r,c]));
        if (pts.length===2) segs.push(pts);
        else if (pts.length===4) { segs.push([pts[0],pts[1]]); segs.push([pts[2],pts[3]]); }
      }
    }
    return segs;
  }

  function drawNLLS(freqs) {
    const nGridG = document.getElementById('cfn-grid');
    const nContG = document.getElementById('cfn-contours');
    const nGdG   = document.getElementById('cfn-gd');
    const nCfG   = document.getElementById('cfn-cf');
    const nAxesG = document.getElementById('cfn-axes');
    const nMarkG = document.getElementById('cfn-markers');

    const lossGrid = Array.from({length: N_GRID}, (_, r) => {
      const a = A_MIN + r/(N_GRID-1)*(A_MAX-A_MIN);
      return Array.from({length: N_GRID}, (_, c) => {
        const K = K_MIN + c/(N_GRID-1)*(K_MAX-K_MIN);
        return Math.log10(Math.max(zipfLoss(K, a, freqs), 1e-12));
      });
    });

    [-0.02, 0.0, 0.02, 0.04, 0.06].forEach(K =>
      nGridG.appendChild(el('line', {x1:nSX(K), y1:mT, x2:nSX(K), y2:mT+PH, stroke:'#21262d', 'stroke-width':1})));
    [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4].forEach(a =>
      nGridG.appendChild(el('line', {x1:N_OX+mL, y1:nSY(a), x2:N_OX+mL+PW, y2:nSY(a), stroke:'#21262d', 'stroke-width':1})));

    nAxesG.appendChild(el('rect', {x:N_OX+mL, y:mT, width:PW, height:PH, fill:'none', stroke:'#484f58', 'stroke-width':1}));
    nAxesG.appendChild(el('line', {x1:nSX(0), y1:mT, x2:nSX(0), y2:mT+PH, stroke:'#484f58', 'stroke-width':1.5}));
    [-0.02, 0.0, 0.02, 0.04, 0.06].forEach(K => {
      nAxesG.appendChild(el('line', {x1:nSX(K), y1:mT+PH, x2:nSX(K), y2:mT+PH+4, stroke:'#8b949e', 'stroke-width':1}));
      nAxesG.appendChild(txt(K.toFixed(2), {x:nSX(K), y:mT+PH+16, fill:'#6e7681', 'font-size':10, 'text-anchor':'middle'}));
    });
    [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4].forEach(a => {
      nAxesG.appendChild(el('line', {x1:N_OX+mL-4, y1:nSY(a), x2:N_OX+mL, y2:nSY(a), stroke:'#8b949e', 'stroke-width':1}));
      nAxesG.appendChild(txt(a.toFixed(1), {x:N_OX+mL-8, y:nSY(a)+4, fill:'#6e7681', 'font-size':10, 'text-anchor':'end'}));
    });
    nAxesG.appendChild(txt('K', {x:N_OX+mL+PW/2, y:mT+PH+32, fill:'#8b949e', 'font-size':13, 'text-anchor':'middle'}));
    const nYl = txt('α', {x:N_OX+14, y:mT+PH/2, fill:'#8b949e', 'font-size':13, 'text-anchor':'middle'});
    nYl.setAttribute('transform', `rotate(-90,${N_OX+14},${mT+PH/2})`);
    nAxesG.appendChild(nYl);

    let logMin = Infinity, logMax = -Infinity;
    for (const row of lossGrid) for (const v of row) { if (v < logMin) logMin = v; if (v > logMax) logMax = v; }
    const N_LEVELS = 14;
    for (let i = 0; i < N_LEVELS; i++) {
      const level = logMin + (i/(N_LEVELS-1))*(logMax-logMin);
      for (const [p1, p2] of marchingSquares(lossGrid, level)) {
        nContG.appendChild(el('line', {
          x1: nColSX(p1[1]).toFixed(1), y1: nRowSY(p1[0]).toFixed(1),
          x2: nColSX(p2[1]).toFixed(1), y2: nRowSY(p2[0]).toFixed(1),
          stroke: '#30363d', 'stroke-width': 0.8, 'stroke-linecap': 'round'
        }));
      }
    }

    let [K, a] = X0_N;
    const gdPtsN = [[K, a]];
    for (let s = 0; s < N_STEPS_N; s++) {
      const [dK, dA] = zipfGrad(K, a, freqs);
      K -= ETA_N*dK; a -= ETA_N*dA;
      if (!isFinite(K) || !isFinite(a)) break;
      gdPtsN.push([K, a]);
    }
    const cfPtsN = windowedAvg(gdPtsN, W_N);

    for (let i = 0; i < gdPtsN.length - 1; i++) {
      const stroke = lerpColor(i / Math.max(1, gdPtsN.length - 2));
      nGdG.appendChild(el('line', {x1:nSX(gdPtsN[i][0]).toFixed(1), y1:nSY(gdPtsN[i][1]).toFixed(1), x2:nSX(gdPtsN[i+1][0]).toFixed(1), y2:nSY(gdPtsN[i+1][1]).toFixed(1), stroke, 'stroke-width':1.5, 'stroke-linecap':'round'}));
    }

    for (let i = 0; i < cfPtsN.length - 1; i++) {
      const stroke = lerpCfColor(i / (cfPtsN.length - 2));
      nCfG.appendChild(el('line', {x1:nSX(cfPtsN[i][0]).toFixed(1), y1:nSY(cfPtsN[i][1]).toFixed(1), x2:nSX(cfPtsN[i+1][0]).toFixed(1), y2:nSY(cfPtsN[i+1][1]).toFixed(1), stroke, 'stroke-width':2.5, 'stroke-linecap':'round'}));
    }

    nMarkG.appendChild(el('circle', {cx:nSX(X0_N[0]).toFixed(1), cy:nSY(X0_N[1]).toFixed(1), r:7, fill:'#3fb950', stroke:'#0d1117', 'stroke-width':2}));
  }

  drawQuad();
  fetch('/assets/shared/data/zipf-hamlet.json')
    .then(r => r.json())
    .then(data => drawNLLS(data.freqs));
})();


// Central flow widget (Figure 4)
(function() {
  const W_SVG = 700, H_SVG = 340;
  const N_STEPS = 1500;
  const ns = 'http://www.w3.org/2000/svg';
  const mLeft = 65, mRight = 20, mTop = 35, mBot = 45;
  const panelW = W_SVG - mLeft - mRight;
  const panelH = H_SVG - mTop - mBot;

  const gridG    = document.getElementById('cf-grid');
  const axesG    = document.getElementById('cf-axes');
  const gdPathEl = document.getElementById('cf-gd-path');
  const cfPathEl = document.getElementById('cf-flow-path');
  const infoDiv  = document.getElementById('cf-info');
  const etaSlider  = document.querySelector('[data-param="cf-eta"]');
  const etaReadout = document.querySelector('[data-readout="cf-eta"]');
  const winSlider  = document.querySelector('[data-param="cf-window"]');
  const winReadout = document.querySelector('[data-readout="cf-window"]');

  let freqs = null, ranks = null, logRanks = null;

  function mk(tag, attrs) {
    const el = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }
  function mkTxt(text, attrs) { const el = mk('text', attrs); el.textContent = text; return el; }
  function sup(n) {
    const m = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻'};
    return String(n).split('').map(c => m[c] || c).join('');
  }

  function computeLossGrad(K, alpha) {
    const N = freqs.length;
    let lossAcc = 0, dK = 0, dA = 0;
    for (let i = 0; i < N; i++) {
      const ra  = Math.pow(ranks[i], alpha);
      const yh  = K * ra;
      const res = freqs[i] - yh;
      lossAcc += res * res;
      dK -= ra * res;
      dA -= logRanks[i] * yh * res;
    }
    return { loss: lossAcc / (2 * N), dK: dK / N, dA: dA / N };
  }

  function runGD(eta) {
    let K = 0.0, alpha = -0.8;
    const iters = [[K, alpha]];
    const losses = [];
    for (let s = 0; s < N_STEPS; s++) {
      const { loss, dK, dA } = computeLossGrad(K, alpha);
      losses.push(loss);
      K     -= eta * dK;
      alpha -= eta * dA;
      iters.push([K, alpha]);
    }
    losses.push(computeLossGrad(K, alpha).loss);
    return { iters, losses };
  }

  function windowedAverage(iters, W) {
    return iters.map((_, k) => {
      const start = Math.max(0, k - W + 1);
      const slice = iters.slice(start, k + 1);
      const n = slice.length;
      return slice.reduce((a, b) => [a[0] + b[0], a[1] + b[1]], [0, 0]).map(v => v / n);
    });
  }

  function toX(step) { return mLeft + (step / N_STEPS) * panelW; }
  function toLossY(v, logMin, logMax) {
    const lv = Math.log10(Math.max(v, 1e-15));
    const clamped = Math.min(logMax, Math.max(logMin, lv));
    return mTop + panelH - (clamped - logMin) / (logMax - logMin) * panelH;
  }

  function drawPanel(gdLosses, cfLosses) {
    let logMin = Infinity, logMax = -Infinity;
    for (const v of gdLosses.concat(cfLosses)) {
      if (v > 0) { const lv = Math.log10(v); if (lv < logMin) logMin = lv; if (lv > logMax) logMax = lv; }
    }
    logMin = Math.floor(logMin) - 1;
    logMax = Math.ceil(logMax);

    gridG.innerHTML = ''; axesG.innerHTML = '';

    [0, 300, 600, 900, 1200, 1500].forEach(t => {
      const sx = toX(t);
      gridG.appendChild(mk('line', {x1:sx, y1:mTop, x2:sx, y2:mTop+panelH, stroke:'#21262d', 'stroke-width':1}));
      axesG.appendChild(mkTxt(t, {x:sx, y:mTop+panelH+14, fill:'#6e7681', 'font-size':10, 'text-anchor':'middle'}));
    });

    for (let e = Math.ceil(logMin); e <= Math.floor(logMax); e++) {
      const sy = toLossY(Math.pow(10, e), logMin, logMax);
      if (sy < mTop || sy > mTop + panelH) continue;
      gridG.appendChild(mk('line', {x1:mLeft, y1:sy, x2:mLeft+panelW, y2:sy, stroke:'#21262d', 'stroke-width':1}));
      axesG.appendChild(mkTxt('10'+sup(e), {x:mLeft-4, y:sy+4, fill:'#6e7681', 'font-size':10, 'text-anchor':'end'}));
    }

    axesG.appendChild(mk('line', {x1:mLeft, y1:mTop, x2:mLeft, y2:mTop+panelH, stroke:'#484f58', 'stroke-width':1.5}));
    axesG.appendChild(mk('line', {x1:mLeft, y1:mTop+panelH, x2:mLeft+panelW, y2:mTop+panelH, stroke:'#484f58', 'stroke-width':1.5}));

    const midY = mTop + panelH / 2;
    axesG.appendChild(mkTxt('Loss', {x:14, y:midY, fill:'#8b949e', 'font-size':12, 'text-anchor':'middle', transform:`rotate(-90,14,${midY})`}));
    axesG.appendChild(mkTxt('Step', {x:mLeft+panelW/2, y:mTop+panelH+30, fill:'#8b949e', 'font-size':12, 'text-anchor':'middle'}));
    axesG.appendChild(mkTxt('Loss vs Step', {x:mLeft+panelW/2, y:mTop-14, fill:'#8b949e', 'font-size':12, 'text-anchor':'middle'}));

    let d = '';
    for (let i = 0; i <= N_STEPS; i++) {
      const sx = toX(i), sy = toLossY(gdLosses[i], logMin, logMax);
      d += (i ? ` L ${sx} ${sy}` : `M ${sx} ${sy}`);
    }
    gdPathEl.setAttribute('d', d);

    d = '';
    for (let i = 0; i <= N_STEPS; i++) {
      const sx = toX(i), sy = toLossY(cfLosses[i], logMin, logMax);
      d += (i ? ` L ${sx} ${sy}` : `M ${sx} ${sy}`);
    }
    cfPathEl.setAttribute('d', d);
  }

  function update() {
    if (!freqs) return;
    const eta   = parseFloat(etaSlider.value);
    const winSz = parseInt(winSlider.value);
    etaReadout.textContent = eta;
    winReadout.textContent = winSz;

    const { iters, losses: gdLosses } = runGD(eta);
    const cfIters  = windowedAverage(iters, winSz);
    const cfLosses = cfIters.map(([K, a]) => computeLossGrad(K, a).loss);

    drawPanel(gdLosses, cfLosses);
    infoDiv.innerHTML = `\u03b7 = ${eta} &nbsp;|&nbsp; W = ${winSz}`;
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
  winSlider.addEventListener('input', update);
})();

</script>
