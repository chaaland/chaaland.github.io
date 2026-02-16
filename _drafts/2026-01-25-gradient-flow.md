---
title: "Gradient Flow and Gradient Descent"
categories:
  - Optimization
date: 2026-01-09 19:00:00 +0000
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

Gradient descent is the workhorse of modern machine learning optimization. Given a differentiable loss function $$f: \mathbb{R}^d \to \mathbb{R}$$, gradient descent iteratively updates parameters according to

$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

where $$\eta > 0$$ is the learning rate (or step size). But how should we choose $$\eta$$? Too small and convergence is slow; too large and the iterates may diverge.

To understand this, it helps to study **gradient flow**—the continuous-time limit of gradient descent obtained as $$\eta \to 0$$:

$$\frac{dx}{dt} = -\nabla f(x)$$

This ODE describes a particle flowing downhill on the loss surface at a rate proportional to the gradient magnitude.

## The 1D Quadratic Case

Consider the simplest case: minimizing a 1D quadratic $$f(x) = \frac{S}{2}x^2$$ where $$S > 0$$ is the **sharpness** (curvature) of the function. Larger $$S$$ means a steeper, narrower parabola.

The gradient is $$\nabla f(x) = Sx$$, so gradient descent becomes

$$x_{k+1} = x_k - \eta \cdot S x_k = (1 - S\eta) x_k$$

This is a simple geometric sequence! Starting from $$x_0$$, we have $$x_k = (1-S\eta)^k x_0$$.

For convergence to zero, we need $$|1 - S\eta| < 1$$. Since $$S, \eta > 0$$, this means:

$$0 < S\eta < 2 \quad \Rightarrow \quad \eta < \frac{2}{S}$$

This is a fundamental result: **the maximum stable learning rate is inversely proportional to the sharpness**.

## Gradient Flow Solution

For the quadratic $$f(x) = \frac{S}{2}x^2$$, gradient flow gives the ODE

$$\frac{dx}{dt} = -Sx$$

with solution $$x(t) = x_0 e^{-St}$$. This exponentially decays to zero for any positive sharpness—gradient flow never diverges on a quadratic!

The loss along the trajectory is

$$f(x(t)) = \frac{S}{2}x_0^2 e^{-2St}$$

which shows exponential decay at rate $$2S$$. Sharper functions have faster gradient flow convergence.

## The 2D Quadratic Case

In higher dimensions, the picture becomes richer. Consider a 2D quadratic

$$f(x) = \frac{1}{2}x^T A x$$

where $$A$$ is a symmetric positive definite matrix. The eigenvalues of $$A$$ determine the curvature along the principal axes.

For an elliptical paraboloid with semi-axes $$a$$ and $$b$$, rotated by angle $$\theta$$, the level sets form ellipses. The matrix $$A$$ has eigenvalues $$\lambda_1 = 1/a^2$$ and $$\lambda_2 = 1/b^2$$.

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

<div class="widget-container" id="gd-widget" style="max-width: 900px;">
  <div class="widget-controls">
    <label>
      a
      <input type="range" min="0.3" max="0.75" value="0.5" step="0.05" data-param="a">
      <span class="widget-readout" data-readout="a">1.00</span>
    </label>
    <label>
      b
      <input type="range" min="0.3" max="0.75" value="0.5" step="0.05" data-param="b">
      <span class="widget-readout" data-readout="b">0.50</span>
    </label>
    <label>
      θ
      <input type="range" min="0" max="90" value="30" step="5" data-param="theta">
      <span class="widget-readout" data-readout="theta">30°</span>
    </label>
    <label>
      η
      <input type="range" min="0.01" max="0.5" value="0.15" step="0.01" data-param="lr">
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
      <path id="gd-flow-path" fill="none" stroke="#58a6ff" stroke-width="2.5" stroke-linecap="round"></path>
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
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#58a6ff"></span>Gradient flow</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#f78166"></span>Gradient descent</span>
    <span class="widget-legend-item"><span class="widget-legend-swatch" style="background:#3fb950"></span>Start</span>
  </div>
  <div class="widget-info" id="gd-info">
    λ<sub>max</sub> = <span id="gd-lambda">4.00</span> &nbsp;|&nbsp;
    η<sub>crit</sub> = 2/λ<sub>max</sub> = <span id="gd-eta-crit">0.50</span> &nbsp;|&nbsp;
    <span id="gd-status" style="color:#3fb950">Converging</span>
  </div>
</div>

<script>
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
  const flowPath = document.getElementById('gd-flow-path');
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

  // Build the Hessian matrix A from ellipse parameters
  function buildA(a, b, thetaDeg) {
    const theta = thetaDeg * Math.PI / 180;
    const c = Math.cos(theta), s = Math.sin(theta);
    const l1 = 1 / (a * a), l2 = 1 / (b * b);
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
  function computeFlowPath(x0, y0, A, T, steps) {
    const dt = 0.01;
    const dtPerStep = T / steps;
    const pts = [[x0, y0]];
    const losses = [f(x0, y0, A)];
    let x = x0, y = y0;
    let stepTime = 0;
    for (let t = 0; t < T; t += dt) {
      const [gx, gy] = grad(x, y, A);
      x -= dt * gx;
      y -= dt * gy;
      pts.push([x, y]);
      stepTime += dt;
      if (stepTime >= dtPerStep) {
        losses.push(f(x, y, A));
        stepTime = 0;
      }
      if (Math.abs(x) < 1e-6 && Math.abs(y) < 1e-6) break;
    }
    // Ensure we have enough loss samples
    while (losses.length <= steps) {
      losses.push(f(x, y, A));
    }
    return { pts, losses };
  }

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
    const a = parseFloat(document.querySelector('[data-param="a"]').value);
    const b = parseFloat(document.querySelector('[data-param="b"]').value);
    const theta = parseFloat(document.querySelector('[data-param="theta"]').value);
    const lr = parseFloat(document.querySelector('[data-param="lr"]').value);
    const steps = parseInt(document.querySelector('[data-param="steps"]').value);

    // Update readouts
    document.querySelector('[data-readout="a"]').textContent = a.toFixed(2);
    document.querySelector('[data-readout="b"]').textContent = b.toFixed(2);
    document.querySelector('[data-readout="theta"]').textContent = theta + '°';
    document.querySelector('[data-readout="lr"]').textContent = lr.toFixed(2);
    document.querySelector('[data-readout="steps"]').textContent = steps;

    const A = buildA(a, b, theta);
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

    // Compute paths with losses
    const flowResult = computeFlowPath(x0, y0, A, 10, steps);
    const gdResult = computeGDPath(x0, y0, A, lr, steps);

    // Set loss scale based on initial loss
    const initialLoss = f(x0, y0, A);
    lossMax = Math.pow(10, Math.ceil(Math.log10(initialLoss * 2)));

    // Draw loss axes
    drawLossAxes(steps);

    // Draw left plot paths
    flowPath.setAttribute('d', pointsToPath(flowResult.pts));
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
    document.querySelector('[data-param="a"]').value = 1.0;
    document.querySelector('[data-param="b"]').value = 0.5;
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
</script>

**Tips:**

- Click anywhere on the left plot to set a new starting point
- The right plot shows loss vs step on a log scale—watch how the loss decays (or explodes when diverging)
- When $$\eta > \eta_{\text{crit}}$$, gradient descent diverges (loss explodes upward)
- Try setting $$a = b$$ to see circular contours where GD moves directly toward the origin
- Notice how gradient flow always takes the smooth optimal path while GD can overshoot

## References

1. [Gradient Flow](https://en.wikipedia.org/wiki/Gradient_descent#Gradient_flow)
2. Cohen, J., et al. "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability." ICLR 2021.
3. Boyd, S. & Vandenberghe, L. "Convex Optimization." Cambridge University Press, 2004.
