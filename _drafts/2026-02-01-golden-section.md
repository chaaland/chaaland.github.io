---
title: "Golden Section Search for Robust Regression"
categories:
  - Optimization
date: 2026-01-09 19:00:00 +0000
mathjax: true
tags:
  - Optimization
toc: true
classes: wide
excerpt: "Golden section search reuses objective evaluations to efficiently minimize 1D functions. Learn how this classical algorithm connects to the golden ratio and applies to robust regression."

---

The most common method of fitting a linear model to data is ordinary least squares.
In ordinary least squares we want to solve the following optimization problem

$$
\begin{equation}
\underset{\beta}{\text{minimize}} \quad \sum_{i=1}^N (\beta^T x_i - y_i)^2
\end{equation}
$$

where $$N$$ is the number of samples.
Oftentimes this will be written in matrix form as

$$
\begin{equation}
\underset{\beta}{\text{minimize}} \quad ||X\beta - y||_2^2
\end{equation}
$$

where $$X\in \mathbf{R}^{N\times d}$$ and $$y\in \mathbf{R}^{N}$$.
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
The next section will introduce an iterative algorithm for solving this problem

# Coordinate Descent

Since the objective is the composition of an affine function with the absolute value function (which is convex), the least absolute deviation objective is convex.
This means any local minimum we find will be a global minimum!

In typical applications, the $$\beta$$ we're solving for can be very high dimensional and hard to visualize.
What if we could instead reduce the multi-dimensional optimization down to a series of 1D optimization problems?

This is the idea of coordinate descent.
Instead of optimizing over all $$\beta_1,\ldots,\beta_d$$ jointly, we cycle through each variable holding the other fixed and optimize over just one variable.
For a specific coordinate $$k$$, we hold all other $$\beta_j (j\ne k)$$ fixed, making the residual $$r_i = y_i - Σ_{j\ne k} \beta_j x_{ij} a constant. Now the subproblem becomes a single-variable optimization:

$$
\begin{equation}
\underset{\beta_k}{\text{minimize}} \quad \sum_{i=1}^N |\beta_k x_{ik} - r_i|
\end{equation}
$$

where $$r_i = y_i - \sum_{j\ne k} \beta_j x_{ij}$$.

Now that we have a simple 1D problem, we can graph an example to see what the objective might look like.
In Figure 1, we see an objective with just one absolute value term

<figure class="half">
    <a href="/assets/2026/golden-section/images/1d-abs-deviation-00.png"><img src="/assets/2026/golden-section/images/1d-abs-deviation-00.png"></a>
    <a href="/assets/2026/golden-section/images/1d-abs-deviation-01.png"><img src="/assets/2026/golden-section/images/1d-abs-deviation-01.png"></a>
    <figcaption>Figure 1: Left: individual absolute value terms. Right: their mean (the objective to minimize), showing kinks at the zeros of each term.</figcaption>
</figure>

One observation worth noting is that the graph of the average of the absolute values has non-differentiable points/"kinks" at the vertices of the original absolute value terms.

Figure 2 shows the average of several absolute value terms of the form $$\lvert \beta_k x_{ik} - y_i\rvert $$ and we can see that the kinks in the graph of the mean occur at exactly the non-differentiable points of each absolute value term.

<figure class="half">
    <a href="/assets/2026/golden-section/images/1d-abs-deviation-04.png"><img src="/assets/2026/golden-section/images/1d-abs-deviation-04.png"></a>
    <a href="/assets/2026/golden-section/images/1d-abs-deviation-06.png"><img src="/assets/2026/golden-section/images/1d-abs-deviation-06.png"></a>
    <figcaption>Figure 2: Notice how the kinks in the black curve exactly coincide with the vertices of the absolute terms that make it up.</figcaption>
</figure>

Since these kinks occur at the non-differentiable points of each absolute value term, they occur exactly when $$\beta_k x_{ik} - r_i = 0$$.
More precisely, when $$\beta_k = r_i / x_{ik}$$.

From the figure, it's not hard to see that the minimum will always lie between $$\min\{r_1/x_1, \ldots, r_n/x_n\}$$ and $$\max\{r_1/x_1, \ldots, r_n/x_n\}$$.
If the objective is convex<sup>[2](#footnote2)</sup> and we can bound the minimizer $$\beta_k^\star$$, can we come up with an algorithm to iteratively shrink the bounds on the minimizer?

# Interval Shrinking

We can start by initializing our algorithm with

$$
\begin{align*}
a &=\min\{r_1/x_1, \ldots, r_n/x_n\}\\
b &=\max\{r_1/x_1, \ldots, r_n/x_n\}.
\end{align*}
$$

From the discussion above, it is guaranteed that $$a \le \beta^\star_k \le b$$.

In order to iteratively shrink our interval $$[a, b]$$ in which the solution lies, we need to evaluate the objective at some points inside of the interval.
We can define two new points to evaluate inside the interval by taking an offset from each of the interval endpoints.

Letting $$L=b-a$$, the length of the bounding interval, we can define

$$
\begin{align*}
x_1 &= b - \rho L\\
x_2 &= a + \rho L
\end{align*}
$$

where $$\rho \in (0.5, 1)$$ ensures $$a < x_1 < x_2 < b$$.

There are three possible configurations we can end up in as shown in Figure 3.
<figure class="third">
    <a href="/assets/2026/golden-section/images/case_1.png"><img src="/assets/2026/golden-section/images/case_1.png"></a>
    <a href="/assets/2026/golden-section/images/case_2.png"><img src="/assets/2026/golden-section/images/case_2.png"></a>
    <a href="/assets/2026/golden-section/images/case_3.png"><img src="/assets/2026/golden-section/images/case_3.png"></a>
    <figcaption>Figure 3: The three distinct cases of the two interior points falling to the left of the minimum, straddling the minimum, and to the right of the minimum.</figcaption>
</figure>

At the start of the algorithm, we'll have found $$a$$ and $$b$$ in $$\mathcal{O}(N)$$ time.
For a given $$\rho$$, we can then compute $$x_1$$ and $$x_2$$.
Suppose we evaluate the objective at these new points, and find $$f(x_1) < f(x_2)$$.

Looking at the first subplot of Figure 3, we can see that this scenario would be impossible.
For a convex function, when both $$x_1$$ and $$x_2$$ are to the left of the minimum, the subgradient is negative and therefore $$f(x_1) \ge f(x_2)$$.

However, both the second and third subplot are potentially consistent with the observation that $$f(x_1) < f(x_2)$$.
If we _knew_ we were in situation two, we could shrink the interval from $$[a,b]$$ down to $$[x_1, x_2]$$.
On the other hand, if we _knew_ were in situation three, we could shrink the interval to $$[a, x_1]$$.

Since we cannot distinguish between cases 2 and 3, we must retain all points consistent with either case. The union of these two intervals is simply  $$[a, x_2]$$.

Running through the same argument for when $$f(x_1)\ge f(x_2)$$ shows the interval can be reduced to $$[x_1, b]$$.

The algorithm to find the minimum is to initialize an interval $$[a, b]$$ to the smallest and largest values of kinks in the graph. Then compute $$x_1$$ and $$x_2$$ and evaluate the objective to determine whether the solution lies in $$[a, x_2]$$ or $$[x_1, b]$$. This becomes our new interval and we repeat the procedure.

This algorithm is shown in the code below

<details>
<summary>
Click for code
</summary>

{% highlight python %}

def interval_shrinking_minimize(
    obj_fn,
    a: float,
    b: float,
    rho: float,
    n_iters: int = 5
):
    assert b > a
    assert n_iters >= 0

    L = b - a
    x1 = b - rho * L
    x2 = a + rho * L

    assert a < x1 < x2 < b

    f_a = obj_fn(beta=a)
    f_x1 = obj_fn(beta=x1)
    f_x2 = obj_fn(beta=x2)
    f_b = obj_fn(beta=b)

    for _ in range(n_iters):
        if f_x1 < f_x2:
            b, f_b = x2, f_x2
        else:
            a, f_a = x1, f_x1

        L = b - a

        x1 = b - rho * L
        x2 = a + rho * L

        f_x1 = obj_fn(beta=x1)
        f_x2 = obj_fn(beta=x2)

    return {"a": (a, f_a), "x1": (x1, f_x1), "x2": (x2, f_x2) , "b":(b, f_b)}
{% endhighlight %}

</details>
<br>

Notice how in the code above, `obj_fn` is called twice per iteration, once for $$x_1$$ and again for $$x_2$$.
Each objective evaluation incurs an $$\mathcal{O}(N)$$ cost.
What if we could choose $$\rho$$ so that we could reuse the objective evaluation on the next iteration?

# Golden-Section

To avoid having to compute the objective function at _two_ new points each iteration, we can attempt to set $$\rho$$ so that one of our interior points ($$x_1$$ or $$x_2$$) becomes an interior point in the _next_ iteration.

At initialization we have

$$
\begin{align*}
x_1 = b - \rho (b-a)\\
x_2 = a + \rho (b-a)
\end{align*}
$$

Suppose, $$f(x_1) < f(x_2)$$ so that the interval in the next iteration is $$[a, x_2]$$.
The new interior points will be

$$
x'_1 = x_2 - \rho (x_2-a)\\
x'_2 = a + \rho (x_2-a)
$$

In order to ensure we can re-use $$x_1$$ as one of our interior points on this iteration, we can set $$x'_2 = x_1$$.
To find a $$\rho$$ satisfying this constraint, we can substitute the expressions on both sides

$$a + \rho (x_2 - a) = b - \rho(b-a).$$

Substituting for $$x_2$$ we have

$$a + \rho (a + \rho (b-a) - a) = b - \rho (b-a)$$

and with some rearrangement and cancellation, we have

$$\rho (\rho (b-a)) + \rho (b-a) - (b-a)= 0.$$

Dividing through by $$b-a$$ gives the following condition that $$\rho$$ must satisfy,

$$\rho^2 +\rho - 1= 0.$$

Using the quadratic formula and discarding the negative solution, we find

$$\rho = {-1 + \sqrt{5} \over 2} \approx 0.61803.$$

By setting $$\rho$$ to this value, we ensure we can reuse the objective value at $$f(x_1)$$ and only need to compute the objective at $$x'_1$$ saving us $$\mathcal{O}(N)$$ computation.

The widget in Figure 4 visually demonstrates the golden section algorithm and how points are reused from iteration to iteration.

<div class="widget-container" id="golden-section-widget">
  <div class="widget-controls">
    <label class="widget-label">Iteration: <span id="gs-iter-label">0</span></label>
    <input type="range" id="gs-iter-slider" class="widget-slider" min="0" max="5" step="1" value="0">
  </div>
  <svg class="widget-plot" id="golden-section-svg" viewBox="0 0 600 380">
    <rect x="0" y="0" width="600" height="380" fill="#0d1117"></rect>
    <g id="gs-grid"></g>
    <g id="gs-axes"></g>
    <g id="gs-curve"></g>
    <g id="gs-vlines"></g>
    <g id="gs-dots"></g>
    <g id="gs-labels"></g>
  </svg>
  <div class="widget-info" id="gs-info">
    <div class="widget-info-row">
      <span class="widget-info-label">a:</span>
      <span class="widget-info-value" id="gs-a-val">–</span>
    </div>
    <div class="widget-info-row">
      <span class="widget-info-label">x₁:</span>
      <span class="widget-info-value" id="gs-x1-val">–</span>
    </div>
    <div class="widget-info-row">
      <span class="widget-info-label">x₂:</span>
      <span class="widget-info-value" id="gs-x2-val">–</span>
    </div>
    <div class="widget-info-row">
      <span class="widget-info-label">b:</span>
      <span class="widget-info-value" id="gs-b-val">–</span>
    </div>
    <div class="widget-info-row">
      <span class="widget-info-label">Interval length:</span>
      <span class="widget-info-value" id="gs-interval-val">–</span>
    </div>
  </div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">
    Figure 4: Golden section search on f(β) = mean(|β·xᵢ − yᵢ|) with 7 data points.
    Dashed lines show: <span style="color:#f85149">a</span> (left), <span style="color:#58a6ff">x₁</span> (interior left), <span style="color:#3fb950">x₂</span> (interior right), <span style="color:#d29922">b</span> (right).
  </figcaption>
</div>

The code below is a sample implementation of the golden section algorithm.
Notice how there is now only one call to `obj_fn` per iteration.

<details>
<summary>
Click for code
</summary>

{% highlight python %}

def golden_section_minimize(obj_fn, a, b, n_iters: int = 5):
    assert b > a
    assert n_iters >= 0

    L = b - a
    rho = (-1 + 5**0.5) / 2
    x1 = b - rho * L
    x2 = a + rho * L

    assert a < x1 < x2 < b

    f_a = obj_fn(beta=a)
    f_x1 = obj_fn(beta=x1)
    f_x2 = obj_fn(beta=x2)
    f_b = obj_fn(beta=b)

    for i in range(n_iters):
        if f_x1 < f_x2:
            # When f(x1) < f(x2), x1 becomes the new x2 (reused from previous iteration)

            b, f_b = x2, f_x2
            L = b - a

            x2, f_x2 = x1, f_x1
            
            x1 = b - rho * L
            f_x1 = obj_fn(beta=x1)
        else:
            a, f_a = x1, f_x1
            L = b - a

            x1, f_x1 = x2, f_x2
            x2 = a + rho * L

            f_x2 = obj_fn(beta=x2)

        assert a < x1 < x2 < b, f"{a}, {x1}, {x2}, {b}, {i}"

    return [a, x1, x2, b]
{% endhighlight %}

</details>
<br>

# The Golden Ratio

One remaining question is why this algorithm is referred to as the "golden" section algorithm.
It turns out, it has a very close connection to the golden ratio $$\varphi$$.

Two line segments are said to be in a golden ratio when the ratio of the longer segment to the shorter segment is the same as that of the sum of the lengths of the segments to the longer segment. Concretely, suppose $$a$$ and $$b$$ are the lengths of two line segments with $$a < b$$, then they are said to be in the golden ratio when

$$
{a+b \over b} = {b \over a}.
$$

Denoting the ratio $$b/a$$ as $$\varphi$$, we have

$$
\begin{align*}
{1\over \varphi} + 1 &= \varphi\\
1 + \varphi &= \varphi^2\\
\end{align*}
$$

Solving for $$\varphi$$ using the quadratic formula and discarding the negative solution,

$$\varphi = {1 + \sqrt{5} \over 2}.$$

This looks very close to our "optimal" $$\rho$$ from the golden-section algorithm which was

$$\rho = {-1 + \sqrt{5} \over 2}.$$

If we assume $$b=1$$ and solve for $$a$$ in $$(a+b)/b = \varphi$$, we have $$a = \varphi  - 1= \rho$$!
Solving $$b/a=\varphi$$ for $$a$$, we can also see that $$a = 1/\varphi$$ further cementing the connection to the golden ratio.

# Conclusion

We've seen how the golden section search elegantly solves 1D convex optimization problems by reusing objective evaluations across iterations, halving the computational cost compared to naive interval shrinking.

Recall that solving least absolute deviations regression—the robust alternative to least squares—requires solving:

$$\underset{\beta}{\text{minimize}} \quad ||X\beta - y||_1$$

Since this is a convex objective without a closed-form solution, coordinate descent is a natural approach: cycle through each coordinate $\beta_k$ and solve the resulting 1D subproblem. For each coordinate, we're minimizing a sum of absolute value terms—a convex, unimodal function perfectly suited to this algorithm.

Remarkably, this classical algorithm from numerical optimization connects back to one of mathematics' most famous constants, the golden ratio.

# Footnotes

<a name="footnote1">1</a>: This formula only holds if $$X^TX$$ is invertible. More specifically, when $$X$$ is skinny (i.e. $$N>d$$) and full rank (i.e. $$\mathbf{rank}(X)=d$$)

<a name="footnote2">2</a>: The algorithm will also work for quasiconvex functions (a.k.a unimodal functions) like $$f(x) = -e^{-x^2}$$

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

{% include widget-scripts.html %}
<script>
(function() {
  'use strict';

  // Data from marimo notebook (seed=31)
  const X_DATA = [19, 24, 17, 3, 24, 7, 11];
  const Y_DATA = [21, -19, -7, -7, 0, -8, -2];
  const RHO = (-1 + Math.sqrt(5)) / 2; // ≈ 0.61803
  const COLORS = {
    a: '#f85149',
    x1: '#58a6ff',
    x2: '#3fb950',
    b: '#d29922'
  };

  // SVG layout
  const MARGIN = { top: 30, right: 30, bottom: 50, left: 60 };
  const SVG_W = 600;
  const SVG_H = 380;
  const PLOT_W = SVG_W - MARGIN.left - MARGIN.right;
  const PLOT_H = SVG_H - MARGIN.top - MARGIN.bottom;

  // DOM elements
  const widget = document.getElementById('golden-section-widget');
  if (!widget) return;

  const svg = document.getElementById('golden-section-svg');
  const slider = document.getElementById('gs-iter-slider');
  const iterLabel = document.getElementById('gs-iter-label');
  const gGrid = document.getElementById('gs-grid');
  const gAxes = document.getElementById('gs-axes');
  const gCurve = document.getElementById('gs-curve');
  const gVlines = document.getElementById('gs-vlines');
  const gDots = document.getElementById('gs-dots');
  const gLabels = document.getElementById('gs-labels');
  const aVal = document.getElementById('gs-a-val');
  const x1Val = document.getElementById('gs-x1-val');
  const x2Val = document.getElementById('gs-x2-val');
  const bVal = document.getElementById('gs-b-val');
  const intervalVal = document.getElementById('gs-interval-val');

  // Compute objective function
  function objectiveFunction(beta) {
    let sum = 0;
    for (let i = 0; i < X_DATA.length; i++) {
      sum += Math.abs(beta * X_DATA[i] - Y_DATA[i]);
    }
    return sum / X_DATA.length;
  }

  // Initialize the algorithm state
  const knots = Y_DATA.map((y, i) => y / X_DATA[i]);
  let initA = Math.min(...knots);
  let initB = Math.max(...knots);
  let initL = initB - initA;
  let initX1 = initB - RHO * initL;
  let initX2 = initA + RHO * initL;

  // Compute initial objective values
  let initFa = objectiveFunction(initA);
  let initFx1 = objectiveFunction(initX1);
  let initFx2 = objectiveFunction(initX2);
  let initFb = objectiveFunction(initB);

  // Pre-compute all algorithm states
  function computeStates() {
    const states = [];
    let a = initA, b = initB, x1 = initX1, x2 = initX2;
    let fa = initFa, fx1 = initFx1, fx2 = initFx2, fb = initFb;

    // Iteration 0: all points are new
    states.push({ a, b, x1, x2, fa, fx1, fx2, fb, newEval: null });

    for (let iter = 0; iter < 5; iter++) {
      if (fx1 < fx2) {
        // Shrink from right; x1 is reused as new x2
        b = x2;
        fb = fx2;
        let L = b - a;
        x2 = x1;
        fx2 = fx1;
        x1 = b - RHO * L;
        fx1 = objectiveFunction(x1);
        states.push({ a, b, x1, x2, fa, fx1, fx2, fb, newEval: 'x1' });
      } else {
        // Shrink from left; x2 is reused as new x1
        a = x1;
        fa = fx1;
        let L = b - a;
        x1 = x2;
        fx1 = fx2;
        x2 = a + RHO * L;
        fx2 = objectiveFunction(x2);
        states.push({ a, b, x1, x2, fa, fx1, fx2, fb, newEval: 'x2' });
      }
    }

    return states;
  }

  const allStates = computeStates();

  // Compute data range for plotting
  const betaMin = initA - 0.2 * initL;
  const betaMax = initB + 0.2 * initL;
  const betaRange = betaMax - betaMin;

  let maxF = 0;
  for (let i = 0; i <= 100; i++) {
    const beta = betaMin + (i / 100) * betaRange;
    maxF = Math.max(maxF, objectiveFunction(beta));
  }
  const fMax = maxF * 1.1;

  // Coordinate transforms
  function toSvgX(beta) {
    return MARGIN.left + ((beta - betaMin) / betaRange) * PLOT_W;
  }

  function toSvgY(f) {
    return MARGIN.top + (1 - f / fMax) * PLOT_H;
  }

  // Draw background grid
  function drawGrid() {
    gGrid.innerHTML = '';
    const xStep = betaRange / 5;
    const yStep = fMax / 5;

    for (let beta = betaMin; beta <= betaMax; beta += xStep) {
      const line = WidgetUtils.createSvgElement('line', {
        x1: toSvgX(beta),
        y1: MARGIN.top,
        x2: toSvgX(beta),
        y2: SVG_H - MARGIN.bottom,
        stroke: '#21262d',
        'stroke-width': '1'
      });
      gGrid.appendChild(line);
    }

    for (let f = 0; f <= fMax; f += yStep) {
      const line = WidgetUtils.createSvgElement('line', {
        x1: MARGIN.left,
        y1: toSvgY(f),
        x2: SVG_W - MARGIN.right,
        y2: toSvgY(f),
        stroke: '#21262d',
        'stroke-width': '1'
      });
      gGrid.appendChild(line);
    }
  }

  // Draw axes
  function drawAxes() {
    gAxes.innerHTML = '';

    // X axis
    const xAxis = WidgetUtils.createSvgElement('line', {
      x1: MARGIN.left,
      y1: toSvgY(0),
      x2: SVG_W - MARGIN.right,
      y2: toSvgY(0),
      stroke: '#6e7681',
      'stroke-width': '2'
    });
    gAxes.appendChild(xAxis);

    // Y axis
    const yAxis = WidgetUtils.createSvgElement('line', {
      x1: toSvgX(0),
      y1: MARGIN.top,
      x2: toSvgX(0),
      y2: SVG_H - MARGIN.bottom,
      stroke: '#6e7681',
      'stroke-width': '2'
    });
    gAxes.appendChild(yAxis);
  }

  // Draw objective curve
  function drawCurve() {
    gCurve.innerHTML = '';
    const nSamples = 300;
    const points = [];

    for (let i = 0; i <= nSamples; i++) {
      const beta = betaMin + (i / nSamples) * betaRange;
      const f = objectiveFunction(beta);
      points.push({ beta, f });
    }

    let pathData = `M ${toSvgX(points[0].beta)} ${toSvgY(points[0].f)}`;
    for (let i = 1; i < points.length; i++) {
      pathData += ` L ${toSvgX(points[i].beta)} ${toSvgY(points[i].f)}`;
    }

    const path = WidgetUtils.createSvgElement('path', {
      d: pathData,
      stroke: '#8b949e',
      'stroke-width': '2',
      fill: 'none'
    });
    gCurve.appendChild(path);
  }

  // Update vertical lines and dots for current iteration
  function updateState(iterIdx) {
    const state = allStates[iterIdx];
    const { a, b, x1, x2, newEval } = state;

    // Clear old lines and dots
    gVlines.innerHTML = '';
    gDots.innerHTML = '';
    gLabels.innerHTML = '';

    // Helper to draw vline + dot + label
    // isNew: true = newly computed in this iteration (dashed, lighter)
    //        false = reused from previous iteration (solid, full opacity)
    function drawKeyPoint(beta, fVal, name, color, isNew) {
      const strokeDasharray = isNew ? '5,5' : 'none';
      const opacity = isNew ? '0.6' : '1';

      // Vertical line
      const line = WidgetUtils.createSvgElement('line', {
        x1: toSvgX(beta),
        y1: MARGIN.top,
        x2: toSvgX(beta),
        y2: SVG_H - MARGIN.bottom,
        stroke: color,
        'stroke-width': '2',
        'stroke-dasharray': strokeDasharray,
        'opacity': opacity
      });
      gVlines.appendChild(line);

      // Dot on curve
      const dot = WidgetUtils.createSvgElement('circle', {
        cx: toSvgX(beta),
        cy: toSvgY(fVal),
        r: '4',
        fill: color,
        'opacity': opacity
      });
      gDots.appendChild(dot);

      // Label below x-axis
      const text = WidgetUtils.createSvgElement('text', {
        x: toSvgX(beta),
        y: SVG_H - MARGIN.bottom + 20,
        'text-anchor': 'middle',
        fill: color,
        'font-size': '12',
        'font-weight': 'bold',
        'opacity': opacity
      });
      text.textContent = name;
      gLabels.appendChild(text);
    }

    // Interior points: check if they're newly evaluated
    const x1IsNew = (newEval === 'x1');
    const x2IsNew = (newEval === 'x2');

    // Boundaries and interior points
    drawKeyPoint(a, objectiveFunction(a), 'a', COLORS.a, false);  // never new
    drawKeyPoint(x1, objectiveFunction(x1), 'x₁', COLORS.x1, x1IsNew);
    drawKeyPoint(x2, objectiveFunction(x2), 'x₂', COLORS.x2, x2IsNew);
    drawKeyPoint(b, objectiveFunction(b), 'b', COLORS.b, false);  // never new

    // Update info panel
    aVal.textContent = a.toFixed(4);
    x1Val.textContent = x1.toFixed(4);
    x2Val.textContent = x2.toFixed(4);
    bVal.textContent = b.toFixed(4);
    intervalVal.textContent = (b - a).toFixed(4);
  }

  // Initialize display
  drawGrid();
  drawAxes();
  drawCurve();
  updateState(0);

  // Slider event listener
  slider.addEventListener('input', function() {
    const iter = parseInt(this.value);
    iterLabel.textContent = iter;
    updateState(iter);
  });
})();
</script>
