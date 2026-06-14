---
title: "Golden Section Search for Robust Regression"
categories:
  - Optimization
date: 2026-02-17 19:00:00 +0000
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
\underset{\beta}{\text{minimize}} \quad {1 \over N}\sum_{i=1}^N (\beta^T x_i - y_i)^2
\end{equation}
$$

where $$N$$ is the number of samples and $$\beta \in \mathbf{R}^d$$.
Often this will be written in matrix form as

$$
\begin{equation}
\underset{\beta}{\text{minimize}} \quad {1 \over N}||X\beta - y||_2^2
\end{equation}
$$

where $$X\in \mathbf{R}^{N\times d}$$ and $$y\in \mathbf{R}^{N}$$.
In this formulation, least squares has a particularly nice closed form solution[^fn1]

$$\beta^\star = (X^TX)^{-1}X^Ty.$$

However, least squares is strongly affected by outlier data points.
A more robust fitting procedure is _least absolute deviations_ where we instead solve the optimization problem

$$
\begin{equation}
\underset{\beta}{\text{minimize}} \quad {1 \over N}\sum_{i=1}^N |\beta^T x_i - y_i|
\end{equation}.
$$

Or written in matrix form

$$
\begin{equation}
\underset{\beta}{\text{minimize}} \quad ||X\beta - y||_1
\end{equation}.
$$

However, unlike least squares, there is no succinct closed form solution.
The next section will introduce an iterative algorithm for solving this problem.

# Coordinate Descent

Since the objective is the composition of an affine function with the absolute value function (which is convex), the least absolute deviation objective is convex.
This means any local minimum we find will be a global minimum!

In typical applications, the $$\beta$$ we're solving for can be very high dimensional and hard to visualize.
What if we could instead reduce the multi-dimensional optimization down to a series of 1D optimization problems?

This is the idea of coordinate descent.
Instead of optimizing over all $$\beta_1,\ldots,\beta_d$$ jointly, we cycle through each variable holding the others fixed and optimize over just one variable.
For a specific coordinate $$k$$, since all other $$\beta_j (j\ne k)$$ are held fixed, the term $$\sum_{j\ne k} \beta_j x_{ij}$$ is constant with respect to $$\beta_k$$, so we can rewrite the residual as $$r_i = y_i - Σ_{j\ne k} \beta_j x_{ij}$$. Now the subproblem becomes a single-variable optimization:

$$
\begin{equation}
\underset{\beta_k}{\text{minimize}} \quad {1 \over N}\sum_{i=1}^N |\beta_k x_{ik} - r_i|.
\end{equation}
$$

Now that we have a simple 1D problem, we can graph an example to see what the objective looks like.
With $$N=1$$, the objective is just a single absolute value term with a kink at $$\beta = r_1/x_1$$.

<figure style="max-width: 75%; margin: 0 auto;">
    <a href="/assets/2026/golden-section/images/1d-abs-deviation-00.png"><img src="/assets/2026/golden-section/images/1d-abs-deviation-00.png"></a>
    <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 1: N = 1: the objective is a single absolute value term (grey) whose mean (black) is identical to it, with one kink.</figcaption>
</figure>

With $$N=2$$, the objective becomes the mean of two such terms.

<figure style="max-width: 75%; margin: 0 auto;">
    <a href="/assets/2026/golden-section/images/1d-abs-deviation-01.png"><img src="/assets/2026/golden-section/images/1d-abs-deviation-01.png"></a>
    <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 2: N = 2: two absolute value terms (grey) and their mean (black). The mean is piecewise linear with a kink at the zero of each term.</figcaption>
</figure>

One observation worth noting is that the graph of the average of the absolute values has non-differentiable points ("kinks") at the vertices of the original absolute value terms.
Figure 3 shows the objective with $$N=6$$ and $$N=7$$ data points and we can see that the kinks in the graph of the mean occur at exactly the non-differentiable points of each absolute value term.

<figure class="half">
    <a href="/assets/2026/golden-section/images/1d-abs-deviation-05.png"><img src="/assets/2026/golden-section/images/1d-abs-deviation-05.png"></a>
    <a href="/assets/2026/golden-section/images/1d-abs-deviation-06.png"><img src="/assets/2026/golden-section/images/1d-abs-deviation-06.png"></a>
    <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 3: N = 6 and N = 7: the kinks in the black curve exactly coincide with the vertices of the absolute value terms that make it up.</figcaption>
</figure>

Since these kinks occur at the non-differentiable points of each absolute value term, they occur exactly when $$\beta_k x_{ik} - r_i = 0$$.
More precisely, kinks occur at values when $$\beta_k = r_i / x_{ik}$$ for each $$i=1,\ldots, N$$.

From the figures, it should be clear that the minimum will always lie between $$\min\{r_1/x_1, \ldots, r_n/x_n\}$$ and $$\max\{r_1/x_1, \ldots, r_n/x_n\}$$.
If the objective is convex[^fn2] and we can bound the minimizer $$\beta_k^\star$$, can we come up with an algorithm to iteratively shrink the bounds on the minimizer?

# Three-Point Search

Since the minimum is guaranteed to lie between the smallest and largest knot, we initialize our search with

$$
\begin{align*}
a &=\min\{r_1/x_1, \ldots, r_n/x_n\}\\
b &=\max\{r_1/x_1, \ldots, r_n/x_n\}.
\end{align*}
$$

It is guaranteed that $$a \le \beta^\star_k \le b$$.

A natural first attempt is to place a single interior probe $$x_1$$ inside the bracket $$[a, b]$$.
Together with the two endpoints, this gives three points from which to infer where the minimum lies.
Suppose we evaluate the objective at $$a < x_1 < b$$ and observe their function values.

<figure style="max-width: 75%; margin: 0 auto;">
    <a href="/assets/2026/golden-section/images/three_points_probes_only.png"><img src="/assets/2026/golden-section/images/three_points_probes_only.png"></a>
    <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 4: Three probe points with observed function values. Since f(x₁) is the smallest, the minimum lies somewhere in [a, b].</figcaption>
</figure>

Since $$f(x_1) < f(a)$$ and $$f(x_1) < f(b)$$, we know the minimum is somewhere in $$[a, b]$$.
But do the the three values tell us how we can safely discard either $$[a, x_1]$$ or $$[x_1, b]$$?

As Figure 5 shows, the answer is no.
The same three probe values are consistent with a function whose minimum lies to the left of $$x_1$$ _and_ with a function whose minimum lies to the right of $$x_1$$.

<figure class="half">
    <a href="/assets/2026/golden-section/images/three_points_left_min.png"><img src="/assets/2026/golden-section/images/three_points_left_min.png"></a>
    <a href="/assets/2026/golden-section/images/three_points_right_min.png"><img src="/assets/2026/golden-section/images/three_points_right_min.png"></a>
    <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 5: Two different convex functions that agree on all three probe values yet have their minima in different subintervals.</figcaption>
</figure>

Because we cannot rule out either subinterval, three points are not enough to shrink the bracket past $$[a, b]$$.
To make progress we need a fourth evaluation point.

# Four-Point Search

In order to iteratively shrink our interval $$[a, b]$$ in which the solution lies, we need to evaluate the objective at two points inside of the interval.
We can define two new points to evaluate inside the interval by taking an offset from each of the endpoints.

Letting $$L=b-a$$, the length of the bounding interval, we can define

$$
\begin{align*}
x_1 &= b - \rho L\\
x_2 &= a + \rho L
\end{align*}
$$

where $$\rho \in (0.5, 1)$$ ensures $$a < x_1 < x_2 < b$$.

There are three possible configurations we can end up in.
Figure 6 shows the four probe values $$f(a)$$, $$f(x_1)$$, $$f(x_2)$$, $$f(b)$$ for each case.

<figure class="third">
    <a href="/assets/2026/golden-section/images/case_1_bare.png"><img src="/assets/2026/golden-section/images/case_1_bare.png"></a>
    <a href="/assets/2026/golden-section/images/case_2_bare.png"><img src="/assets/2026/golden-section/images/case_2_bare.png"></a>
    <a href="/assets/2026/golden-section/images/case_3_bare.png"><img src="/assets/2026/golden-section/images/case_3_bare.png"></a>
    <figcaption>Figure 6: The four probe values for three distinct placements of x₁ and x₂ within [a, b].</figcaption>
</figure>

In the left-most plot, the minimum must lie in $$[x_1, x_2]$$ or $$[x_2, b]$$.
In the middle plot, the minimum must lie in $$[x_1, x_2]$$ or $$[x_2, b]$$.
And in the right-most plot, the minimum must lie in $$[a, x_1]$$ or $$[x_1, x_2]$$

Figure 7 shows the same three cases with the underlying objective revealed.

<figure class="third">
    <a href="/assets/2026/golden-section/images/case_1.png"><img src="/assets/2026/golden-section/images/case_1.png"></a>
    <a href="/assets/2026/golden-section/images/case_2.png"><img src="/assets/2026/golden-section/images/case_2.png"></a>
    <a href="/assets/2026/golden-section/images/case_3.png"><img src="/assets/2026/golden-section/images/case_3.png"></a>
    <figcaption>Figure 7: The three distinct cases of the two interior points falling to the left of the minimum, straddling the minimum, and to the right of the minimum.</figcaption>
</figure>

At the start of the algorithm, we'll have found $$a$$ and $$b$$ in $$\mathcal{O}(N)$$ time.
For a given $$\rho$$, we can then compute $$x_1$$ and $$x_2$$.
Suppose we evaluate the objective at these new points, and find $$f(x_1) < f(x_2)$$.

Looking at the first subplot of Figure 6, we can see that this scenario would be impossible.
For a convex function, when both interior points are to the left of the minimum, the function is decreasing (negative subgradient), so $$f(x_1) > f(x_2)$$.

However, both the second and third subplot are potentially consistent with the observation that $$f(x_1) < f(x_2)$$.
If we _knew_ we were in situation two, we could shrink the interval from $$[a,b]$$ down to $$[x_1, x_2]$$.
On the other hand, if we _knew_ we were in situation three, we could shrink the interval to $$[a, x_1]$$.

Since we cannot distinguish between cases 2 and 3, we must retain all points consistent with either case, namely the union of the two intervals, $$[a, x_2]$$.

Running through the same argument for when $$f(x_1)\ge f(x_2)$$ shows the interval can be reduced to $$[x_1, b]$$.

The full algorithm is then to initialize an interval $$[a, b]$$ to the smallest and largest values of kinks in the graph.
Then compute $$x_1$$ and $$x_2$$ as described above and evaluate the objective to determine whether the solution lies in $$[a, x_2]$$ or $$[x_1, b]$$.
This becomes our new interval and we repeat the procedure.

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

Figure 8 shows that for an arbitrary $$\rho$$, we need to evaluate _two_ new interior points.

<figure class="half">
    <a href="/assets/2026/golden-section/images/rho_06.png"><img src="/assets/2026/golden-section/images/rho_06.png"></a>
    <a href="/assets/2026/golden-section/images/rho_07.png"><img src="/assets/2026/golden-section/images/rho_07.png"></a>
    <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 8: For ρ = 0.6 and ρ = 0.7, all probe positions in iteration 1 (diamonds) are new evaluations with none carrying over from iteration 0 (circles).</figcaption>
</figure>

We can see from the figure that there is a value of $$\rho$$ between 0.6 and 0.7 that results in the previous iteration's $$x_1$$ becoming the next iteration's $$x_2$$.

We can derive the value of $$\rho$$ that requires evaluating just one new interation point by noting that at initialization we have

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
Figure 9 confirms this: $$x_1$$ from iteration 0 (filled circle) reappears as $$x_2$$ in iteration 1 (faded circle), connected by the dotted line.

<figure style="max-width: 75%; margin: 0 auto;">
    <a href="/assets/2026/golden-section/images/rho_golden_section.png"><img src="/assets/2026/golden-section/images/rho_golden_section.png"></a>
    <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 9: For ρ = 1/φ ≈ 0.618, x₁ from iteration 0 is reused as x₂ in iteration 1 with only one new evaluation (diamond) needed per iteration.</figcaption>
</figure>

The widget in Figure 10 visually demonstrates the golden section algorithm and how points are reused from iteration to iteration.

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
    Figure 10: Golden section search on f(β) = mean(|β·xᵢ − yᵢ|) with 7 data points.
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

Two line segments are said to be in a golden ratio when the ratio of the longer segment to the shorter segment is the same as that of the sum of the lengths of the segments to that of the longer segment.

Concretely, suppose $$s$$ and $$\ell$$ are the lengths of two line segments with $$s < \ell$$, then they are said to be in the golden ratio when

$$
{s + \ell \over \ell} = {\ell \over s}.
$$

Denoting the ratio $$\ell/s$$ as $$\varphi$$, we have

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

If we assume $$\ell=1$$ and solve for $$s$$ in $$(s+\ell)/\ell = \varphi$$, we have $$s = \varphi  - 1= \rho$$!
Solving $$\ell/s=\varphi$$ for $$s$$, we can also see that $$s = 1/\varphi$$ further cementing the connection to the golden ratio.

# Conclusion

We've seen how the golden section search elegantly solves 1D convex optimization problems by reusing objective evaluations across iterations, halving the computational cost compared to naive interval shrinking.

Recall that solving least absolute deviations regression—the robust alternative to least squares—requires solving:

$$\underset{\beta}{\text{minimize}} \quad {1 \over N}||X\beta - y||_1$$

Since this is a convex objective without a closed-form solution, coordinate descent is a natural approach: cycle through each coordinate $$\beta_k$$ and solve the resulting 1D subproblem. For each coordinate, we're minimizing a sum of absolute value terms—a convex, unimodal function perfectly suited to this algorithm.

Surprisingly, this classical algorithm connects back to one of mathematics' most famous constants, the golden ratio.

[^fn1]: This formula only holds if $$X^TX$$ is invertible. More specifically, when $$X$$ is skinny (i.e. $$N>d$$) and full rank (i.e. $$\mathbf{rank}(X)=d$$)

[^fn2]: The algorithm will also work for quasiconvex functions (a.k.a unimodal functions) like $$f(x) = -e^{-x^2}$$

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
