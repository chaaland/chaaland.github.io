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
  <svg class="widget-plot" id="golden-ratio-svg" viewBox="0 0 600 150">
    <rect x="0" y="0" width="600" height="150" fill="#0d1117"></rect>
    <!-- Bar background -->
    <rect id="gr-bar-bg" x="80" y="50" width="440" height="30" fill="#1c2128"></rect>
    <!-- Left segment -->
    <rect id="gr-left-segment" x="80" y="50" width="220" height="30" fill="#58a6ff"></rect>
    <!-- Right segment -->
    <rect id="gr-right-segment" x="300" y="50" width="220" height="30" fill="#3fb950"></rect>
    <!-- Draggable point -->
    <circle id="gr-point" cx="300" cy="65" r="8" fill="#58a6ff" style="cursor: pointer; pointer-events: auto;"></circle>
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

{% include widget-scripts.html %}
<script>
(function() {
  'use strict';

  // Golden ratio and threshold
  const PHI = 1.618033988749895;
  const TOLERANCE = 0.05; // 5% tolerance
  const PHI_MIN = PHI * (1 - TOLERANCE);
  const PHI_MAX = PHI * (1 + TOLERANCE);

  // Color scheme
  const COLORS = {
    defaultPoint: '#58a6ff',
    goldenPoint: '#ffd700',
    leftSegment: '#58a6ff',
    rightSegment: '#3fb950',
    goldenText: '#ffd700',
    defaultText: '#8b949e'
  };

  // DOM elements
  const widget = document.getElementById('golden-ratio-widget');
  if (!widget) return; // Widget not present

  const svg = document.getElementById('golden-ratio-svg');
  const point = document.getElementById('gr-point');
  const leftSegment = document.getElementById('gr-left-segment');
  const rightSegment = document.getElementById('gr-right-segment');
  const ratio1El = document.getElementById('gr-ratio1');
  const ratio2El = document.getElementById('gr-ratio2');
  const resetBtn = document.getElementById('golden-ratio-reset');
  const labelsG = document.getElementById('gr-labels');

  // Defensive check: ensure all required elements exist
  if (!svg || !point || !leftSegment || !rightSegment || !ratio1El || !ratio2El || !resetBtn) {
    console.error('Golden ratio widget: missing DOM elements', {
      svg: !!svg,
      point: !!point,
      leftSegment: !!leftSegment,
      rightSegment: !!rightSegment,
      ratio1El: !!ratio1El,
      ratio2El: !!ratio2El,
      resetBtn: !!resetBtn
    });
    return;
  }

  // SVG dimensions
  const BAR_X = 80;
  const BAR_Y = 50;
  const BAR_WIDTH = 440;
  const BAR_HEIGHT = 30;
  const POINT_RADIUS = 8;

  // State
  let isDragging = false;
  let currentPosition = BAR_WIDTH / 2; // Start at middle

  /**
   * Calculate ratios given a point position
   * @param {number} position - Position along bar [0, BAR_WIDTH]
   * @returns {Object} {wholeToLonger, longerToShorter, isGolden}
   */
  function calculateRatios(position) {
    const clampedPos = Math.max(0, Math.min(BAR_WIDTH, position));
    const leftLen = clampedPos;
    const rightLen = BAR_WIDTH - clampedPos;

    const longerLen = Math.max(leftLen, rightLen);
    const shorterLen = Math.min(leftLen, rightLen);

    // Avoid division by zero
    if (shorterLen === 0) {
      return {
        wholeToLonger: 1,
        longerToShorter: Infinity,
        isGolden: false
      };
    }

    const wholeToLonger = BAR_WIDTH / longerLen;
    const longerToShorter = longerLen / shorterLen;

    // Check if either ratio is close to golden ratio
    const isGolden = (wholeToLonger >= PHI_MIN && wholeToLonger <= PHI_MAX) ||
                     (longerToShorter >= PHI_MIN && longerToShorter <= PHI_MAX);

    return {
      wholeToLonger,
      longerToShorter,
      isGolden
    };
  }

  /**
   * Update the widget display
   */
  function update() {
    const ratios = calculateRatios(currentPosition);

    // Debug: log calculation
    console.debug('update()', { currentPosition, ratios });

    // Update point position and color
    const pointX = BAR_X + currentPosition;
    point.setAttribute('cx', pointX);
    point.setAttribute('fill', ratios.isGolden ? COLORS.goldenPoint : COLORS.defaultPoint);

    // Update segments
    leftSegment.setAttribute('width', currentPosition);
    rightSegment.setAttribute('x', BAR_X + currentPosition);
    rightSegment.setAttribute('width', BAR_WIDTH - currentPosition);

    // Format and display ratios
    const ratio1Text = ratios.wholeToLonger === Infinity ? '∞' : ratios.wholeToLonger.toFixed(3);
    const ratio2Text = ratios.longerToShorter === Infinity ? '∞' : ratios.longerToShorter.toFixed(3);

    ratio1El.textContent = ratio1Text;
    ratio2El.textContent = ratio2Text;

    // Highlight text if golden ratio found
    const textColor = ratios.isGolden ? COLORS.goldenText : COLORS.defaultText;
    ratio1El.style.color = textColor;
    ratio2El.style.color = textColor;
  }

  /**
   * Handle mouse down on draggable point
   */
  function handleMouseDown(e) {
    isDragging = true;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    e.preventDefault();
  }

  /**
   * Handle mouse move while dragging
   */
  function handleMouseMove(e) {
    if (!isDragging) return;

    const svgRect = svg.getBoundingClientRect();
    const mouseX = e.clientX - svgRect.left;

    // Convert to SVG coordinates (accounting for viewBox scaling)
    const viewBoxWidth = getViewBoxScale();
    const scale = viewBoxWidth / svgRect.width;
    const svgMouseX = mouseX * scale;

    // Clamp position within bar bounds
    currentPosition = Math.max(0, Math.min(BAR_WIDTH, svgMouseX - BAR_X));

    // Debug logging
    console.debug('handleMouseMove:', { mouseX, svgRect: svgRect.width, viewBoxWidth, scale, svgMouseX, BAR_X, currentPosition });

    update();
  }

  /**
   * Handle mouse up to stop dragging
   */
  function handleMouseUp() {
    isDragging = false;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  }

  /**
   * Handle touch start on draggable point
   */
  function handleTouchStart(e) {
    isDragging = true;
    document.addEventListener('touchmove', handleTouchMove, { passive: false });
    document.addEventListener('touchend', handleTouchEnd);
    e.preventDefault();
  }

  /**
   * Handle touch move while dragging
   */
  function handleTouchMove(e) {
    if (!isDragging) return;

    const touch = e.touches[0];
    const svgRect = svg.getBoundingClientRect();
    const touchX = touch.clientX - svgRect.left;

    // Convert to SVG coordinates (accounting for viewBox scaling)
    const viewBoxWidth = getViewBoxScale();
    const scale = viewBoxWidth / svgRect.width;
    const svgTouchX = touchX * scale;

    // Clamp position within bar bounds
    currentPosition = Math.max(0, Math.min(BAR_WIDTH, svgTouchX - BAR_X));

    // Debug logging
    console.debug('handleTouchMove:', { touchX, svgRect: svgRect.width, viewBoxWidth, scale, svgTouchX, BAR_X, currentPosition });

    update();
  }

  /**
   * Handle touch end to stop dragging
   */
  function handleTouchEnd() {
    isDragging = false;
    document.removeEventListener('touchmove', handleTouchMove);
    document.removeEventListener('touchend', handleTouchEnd);
  }

  /**
   * Get viewBox width with safe parsing
   */
  function getViewBoxScale() {
    const viewBox = svg.getAttribute('viewBox');
    if (!viewBox) return 600; // Default matches SVG viewBox
    const parts = viewBox.split(' ');
    const viewBoxWidth = parseFloat(parts[2]);
    return isNaN(viewBoxWidth) ? 600 : viewBoxWidth;
  }

  /**
   * Reset widget to initial state (middle position)
   */
  function reset() {
    currentPosition = BAR_WIDTH / 2;
    update();
  }

  // Initialize event listeners
  point.addEventListener('mousedown', handleMouseDown);
  point.addEventListener('touchstart', handleTouchStart);
  resetBtn.addEventListener('click', reset);

  // Debug: log initial state
  console.debug('Golden ratio widget initialized', {
    currentPosition,
    BAR_WIDTH,
    pointCx: point.getAttribute('cx'),
    viewBoxWidth: getViewBoxScale()
  });

  // Initial update
  update();

  // Cleanup on page unload
  window.addEventListener('beforeunload', function() {
    if (isDragging) {
      handleMouseUp();
    }
    point.removeEventListener('mousedown', handleMouseDown);
    point.removeEventListener('touchstart', handleTouchStart);
    resetBtn.removeEventListener('click', reset);
  });
})();
</script>
