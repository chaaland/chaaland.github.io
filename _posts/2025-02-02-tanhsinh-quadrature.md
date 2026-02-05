---
title: "Tanhsinh Quadrature"
categories:
  - Algorithms
date:   2025-01-17 14:33:00 +0100
mathjax: true
tags:
  - Numerical methods
  - Numerical integration
  - Quadrature
toc: true
# classes: wide
excerpt: "Tackle tricky integrals with endpoint singularities using a clever variable transformation."
header: 
  overlay_image: assets/2025/tanhsinh-quadrature/images/splash_image.png
  overlay_filter: 0.2
---

Tanhsinh quadrature is a powerful numerical integration technique for handling integrals with endpoint singularities, leveraging a clever variable transformation to cluster points where the function changes rapidly.

## Motivation

The fundamental theorem of calculus gives us a deceptively simple method of evaluating a definite integral as the difference of its antiderivative at each end point

$$\int_a^b f(x) dx = F(b) - F(a).$$

But what if $$F(x)$$ doesn't exist in closed form?
Or what if we're writing software where users provide arbitrary functions to integrate?
This is where numerical integration becomes essential.

In the release notes of [Scipy 1.15.0](https://docs.scipy.org/doc/scipy/release/1.15.0-notes.html) a new method called `tanhsinh` was added to the `scipy.integrate` subpackage.
In this post we'll see how this method can be used to accurately approximate especially tricky integrals.

## Riemann approximation

The simplest method we learn for approximating the integral is the Riemann sum.
This approximates the area under the curve as the sum of many small rectangles.
The more rectangles we have, the more accurate the approximation.

Mathematically, this translates to:

$$
\begin{align*}
\int_a^b f(x)dx &\approx \sum_{k=0}^{N-1} \left({b-a \over N}\right)f\left(a + {b-a \over N} k\right)\\
&\approx \sum_{k=0}^{N-1} hf\left(a + hk\right)
\end{align*}
$$

Where each term in the sum represents the area of one of our rectangles.
The width of each block is $$h={b-a \over N}$$ (the total interval divided into $$n$$ pieces), and the height is $$f\left(a + {b-a \over N} k\right)$$ (the function evaluated at the left edge of each block)<sup>[1](#footnote1)</sup>.

<div class="widget-container" id="riemann-widget">
  <div class="widget-controls">
    <label>
      Rectangles
      <input type="range" min="1" max="50" value="10" step="1" id="riemann-n-slider">
      <span class="widget-readout" id="riemann-n-readout">10</span>
    </label>
  </div>
  <svg class="widget-plot" id="riemann-svg" viewBox="0 0 450 440" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="450" height="440" fill="#0d1117"></rect>
    <g id="riemann-grid"></g>
    <g id="riemann-axes"></g>
    <g id="riemann-rects"></g>
    <path id="riemann-curve" fill="none" stroke="#58a6ff" stroke-width="2.5"></path>
    <g id="riemann-labels"></g>
  </svg>
  <div class="widget-footer">
    <span>Approximate area: <span class="widget-value" id="riemann-approx">0.7188</span></span>
    <span>Exact area (ln 2): <span class="widget-value" id="riemann-exact">0.6931</span></span>
    <span class="widget-error">Error: <span id="riemann-error">3.70%</span></span>
  </div>
</div>

<figcaption style="text-align: center; font-size: 0.9rem; color: #8b949e; margin-top: 0.5rem;">
Figure 1: Interactive left Riemann approximation.
</figcaption>

We can also approximate the area with rectangles measured from the top right corner just by incrementing the indices of summation.

$$
\int_a^b f(x)dx \approx \sum_{k=1}^{N} hf\left(a + hk\right)
$$

<div class="widget-container" id="right-riemann-widget">
  <div class="widget-controls">
    <label>
      Rectangles
      <input type="range" min="1" max="50" value="10" step="1" id="right-riemann-n-slider">
      <span class="widget-readout" id="right-riemann-n-readout">10</span>
    </label>
  </div>
  <svg class="widget-plot" id="right-riemann-svg" viewBox="0 0 450 440" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="450" height="440" fill="#0d1117"></rect>
    <g id="right-riemann-grid"></g>
    <g id="right-riemann-axes"></g>
    <g id="right-riemann-rects"></g>
    <path id="right-riemann-curve" fill="none" stroke="#58a6ff" stroke-width="2.5"></path>
    <g id="right-riemann-labels"></g>
  </svg>
  <div class="widget-footer">
    <span>Approximate area: <span class="widget-value" id="right-riemann-approx">0.6688</span></span>
    <span>Exact area (ln 2): <span class="widget-value" id="right-riemann-exact">0.6931</span></span>
    <span class="widget-error">Error: <span id="right-riemann-error">3.52%</span></span>
  </div>
</div>

<figcaption style="text-align: center; font-size: 0.9rem; color: #8b949e; margin-top: 0.5rem;">
Figure 2: Interactive right Riemann approximation.
</figcaption>

We can think of Riemann sums as approximating the function as piecewise constant<sup>[2](#footnote2)</sup>.

## Trapezoidal approximation

A slightly more sophisticated method is to approximate the area as the sum of trapezoids.
We can think of this as approximating the function as piecewise _linear_.

Recalling that the area of trapezoid is $${height \over 2}(base_1 + base_2)$$ we get the slightly more complicated approximation

$$
\int_a^b f(x)dx \approx \sum_{k=0}^{N-1} {h \over 2}\left[f\left(a + hk\right) + f\left(a + h(k+1)\right)\right].
$$

<div class="widget-container" id="trapezoid-widget">
  <div class="widget-controls">
    <label>
      Trapezoids
      <input type="range" min="1" max="50" value="10" step="1" id="trapezoid-n-slider">
      <span class="widget-readout" id="trapezoid-n-readout">10</span>
    </label>
  </div>
  <svg class="widget-plot" id="trapezoid-svg" viewBox="0 0 450 440" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="450" height="440" fill="#0d1117"></rect>
    <g id="trapezoid-grid"></g>
    <g id="trapezoid-axes"></g>
    <g id="trapezoid-shapes"></g>
    <path id="trapezoid-curve" fill="none" stroke="#58a6ff" stroke-width="2.5"></path>
    <g id="trapezoid-labels"></g>
  </svg>
  <div class="widget-footer">
    <span>Approximate area: <span class="widget-value" id="trapezoid-approx">0.6938</span></span>
    <span>Exact area (ln 2): <span class="widget-value" id="trapezoid-exact">0.6931</span></span>
    <span class="widget-error">Error: <span id="trapezoid-error">0.09%</span></span>
  </div>
</div>

<figcaption style="text-align: center; font-size: 0.9rem; color: #8b949e; margin-top: 0.5rem;">
Figure 3: Interactive trapezoidal approximation.
</figcaption>

## Quadrature

The Riemann and trapezoidal methods are special cases of quadrature which is a family of integral approximation methods which take the form

$$\int_a^b f(x)dx \approx \sum_{k=0}^N w_k f(x_k)$$

In the left handed Riemann approximation,

$$
\begin{align*}
w_k &=
\begin{cases}
    0 & k=N \\
    {b-a \over N} & \text{otherwise}
\end{cases}\\
x_k &= a + \left({b-a \over N}\right)k
\end{align*}
$$

The right handed Riemann approximation is the same except the first weight is zero rather than the last.

$$
\begin{align*}
w_k &=
\begin{cases}
    0 & k=0 \\
    {b-a \over N} & \text{otherwise}
\end{cases}\\
x_k &= a + \left({b-a \over N}\right)k
\end{align*}
$$

The trapezoidal rule has nearly the same weights except the endpoints are half the other weights.

$$
\begin{align*}
w_k &=
\begin{cases}
    {1 \over 2}{b-a \over N} & k=0 \text{ or } N\\
    {b-a \over N} & \text{otherwise}
\end{cases}\\
x_k &= a + \left({b-a \over N}\right)k
\end{align*}
$$

Quadrature then amounts to the following Python code

{% highlight python %}
def quadrature(f, a=-1, b=1, n_points=10):
    x_k, w_k = compute_points(a, b, n_points)
    return np.dot(w_k, f(x_k))
{% endhighlight %}

where `compute_points` implements one of the above formulae.

The exact value of $$\int_1^2 {1 \over x} dx$$ is $$\log 2$$.
The different quadrature method's accuracy is shown in the following table.

| method        | value     | % error  |
| ------------- | --------- | -------- |
| left riemann  | 0.7188    | 3.697%   |
| right riemann | 0.6688    | 3.517%   |
| trapezoidal   | 0.6938    | 0.09006% |
| exact         | 0.6931    | 0%       |

With only 10 function evaluations, that's not too bad!

But what if the function has an asymptote at one of the endpoints?
Consider integrating $${1 \over \sqrt{1-x}}$$ over the interval [-1, 1].

<figure class>
    <a href="/assets/2025/tanhsinh-quadrature/images/shifted_rsqrt.png"><img src="/assets/2025/tanhsinh-quadrature/images/shifted_rsqrt.png"></a>
    <figcaption>Figure 4: Function with an asymptote at its right endpoint.</figcaption>
</figure>

Applying our quadrature methods so far leads to summations in which the last term is infinite, completely destroying the approximation.

One thing we _could_ try to do if we were programming a library is to only sum the values of `f(x_k)` which are finite.
Our approximation is saved from blowing up to infinity but we still lose accuracy.

In the next section we'll see a method for taming these endpoint asymptotes.

## tanh-sinh quadrature

Why do our previous methods struggle with endpoint asymptotes?
The problem is that we're sampling points uniformly in our interval.
Near the asymptote, the function changes so rapidly that a reasonable number of uniform samples doesn't capture its behavior.

Tanh-sinh quadrature transforms our integral in a way that "tames" these asymptotes.

In this section we'll restrict our attention to integrals in the interval $$[-1,1]$$ for simplicity<sup>[3](#footnote3)</sup>.
We'll start by making the rather unusual variable substitution

$$x = \tanh\left(\sinh \left({\pi \over 2} t\right)\right)$$

This function is plotted in Figure 5 along with the regular hyperbolic tangent.

<figure class>
    <a href="/assets/2025/tanhsinh-quadrature/images/tanhsinh.png"><img src="/assets/2025/tanhsinh-quadrature/images/tanhsinh.png"></a>
    <figcaption>Figure 5: The tanhsinh function saturates much more quickly than the regular hyperbolic tangent.</figcaption>
</figure>

From the figure we can see that it maps the infinite interval $$(-\infty, \infty)$$ to $$(-1, 1)$$.
Also, because of how rapidly the function saturates, it will cluster our sampling points near $$\pm 1$$, exactly where we need them for handling endpoint behavior.

Recalling that the derivatives of the hyperbolic functions are essentially the same as their trigonometric counterparts, we can apply the chain rule to compute the derivative

$$
\begin{align*}
{dx \over dt} &= \text{sech}^2\left(\sinh \left({\pi \over 2} t\right)\right) {d\over dt}\left(\sinh \left({\pi \over 2} t\right)\right)\\
&= {\pi \over 2}\text{sech}^2\left(\sinh \left({\pi \over 2} t\right)\right) \cosh \left({\pi \over 2} t\right)
\end{align*}
$$

With this substitution, our integral becomes the doubly improper, and rather complicated looking,

$$
\begin{align*}
\int_{-1}^1 f(x) &\,dx = \\
&\int_{-\infty}^\infty f\left(\tanh\left(\sinh \left({\pi \over 2} t\right)\right)\right){\pi \over 2}\text{sech}^2\left(\sinh \left({\pi \over 2} t\right)\right) \cosh \left({\pi \over 2} t\right) dt
\end{align*}
$$

It's not immediately clear what we have gained by doing this.
In fact, it looks like we may have even made things significantly worse.

But let's look at what happens to the integrand $${1 \over \sqrt{1-x}}$$ using this substitution.

<figure class>
    <a href="/assets/2025/tanhsinh-quadrature/gifs/tanhsinh_transform_inf_loop.gif"><img src="/assets/2025/tanhsinh-quadrature/gifs/tanhsinh_transform_inf_loop.gif"></a>
    <figcaption>Figure 6: The original integrand being transformed after the tanhsinh substitution. The colors are just to keep track of where each segment of the original graph gets mapped to in the end.</figcaption>
</figure>

Have a look again at the integrand before and after the tanhsinh substitution.

<figure class="half">
    <a href="/assets/2025/tanhsinh-quadrature/images/integrand.png"><img src="/assets/2025/tanhsinh-quadrature/images/integrand.png"></a>
    <a href="/assets/2025/tanhsinh-quadrature/images/integrand_tanhsinh.png"><img src="/assets/2025/tanhsinh-quadrature/images/integrand_tanhsinh.png"></a>
    <figcaption>Figure 7: The left shows the original integrand while the right shows the integrand after the tanhsinh substitution</figcaption>
</figure>

The figure makes it clear that the asymptote at 1 disappears!
What's more, the function decays _extremely_ quickly to 0.
To get a sense of just how fast this function decays to 0, compare with the standard Gaussian $$e^{-x^2}$$ which is  $$1.23\times 10^{-4}$$ at $$x=-3$$.
Our integrand is down to $$9.60\times 10^{-13}$$ at the same value of $$x$$.
Over eight orders of magnitude smaller!

The exceedingly fast roll-off makes accurate numerical integration much easier since $$f(x_k)$$ will quickly go to 0 and contribute nothing to the sum.
This allows us to effectively truncate the summation with very negligible error.

We can now use Riemann sums on this transformed integrand to get an accurate approximation of the original integral. In the case of proper integrals, the number of points is explicitly defined by the width of the rectangle, but when $$a$$ or $$b$$ are infinite, we have to choose them independently.
Since the integral after the substitution is improper, we need to adapt our quadrature formula slightly:

$$\int_{-1}^1 f(x) dx \approx \sum_{k=-N}^{N-1} w_k f(x_k)$$

where

$$
\begin{align*}
w_k &=h{\pi \over 2}\text{sech}^2\left(\sinh \left({\pi \over 2} kh\right)\right) \cosh \left({\pi \over 2} kh\right) \\
x_k &=f\left(\tanh\left(\sinh \left({\pi \over 2} kh\right)\right)\right)
\end{align*}
$$

<figure class>
    <a href="/assets/2025/tanhsinh-quadrature/images/tanhsinh_riemann.png"><img src="/assets/2025/tanhsinh-quadrature/images/tanhsinh_riemann.png"></a>
    <figcaption>Figure 8: Left Riemann rectangles of the integrand after the tanhsinh substitution using 20 rectangles between -3 and 3.</figcaption>
</figure>

A simple python implementation of tanhsinh quadrature is shown below:

{% highlight python %}
def tanh_sinh_points(n, h=0.1):
    # Generate evaluation points
    k = np.arange(-n, n)
    t = h * k

    # Apply the tanh(sinh()) transformation
    sinh_t = np.sinh(t)
    cosh_t = np.cosh(t)

    # Compute the quadrature points (x values)
    x_k = np.tanh(np.pi / 2 * sinh_t)

    # Compute the weights using the derivative of the transformation
    cosh_term = np.cosh(np.pi / 2 * sinh_t)
    w_k = h * np.pi / 2 * cosh_t / (cosh_term * cosh_term)

    return x_k, w_k

def tanh_sinh_quadrature(f, n=30, h=0.1):
    points, weights = tanh_sinh_points(n, h)
    return np.sum(weights * f(points))
{% endhighlight %}

The true value is $$2\sqrt{2}\approx 2.828427$$. Using this code we can approximate the integral with $$N=10$$ and $$h=0.3$$ which gives $$2.828425$$, a $$7\times 10^{-5}%$$ error! Compare this with 20 left Riemann rectangles which gives $$2.40183$$, a 15% error.

### Implementation Challenges

The code above is actually numerically unstable and will break when $$n*h$$ gets a bit bigger than 3 (using double precision).
This is because of the rapid saturation of the tanhsinh substitution and the limits of floating point precision.
The $$x$$ values become indistinguishable from 1 for arguments just larger than 3.
This causes $$1/\sqrt{1-x}$$ to divide by zero and enters an infinity into the summation.

A more robust implementation of this would find the first value (based on the floating point precision) that becomes identically 1 under the tanhsinh transformation.<sup>[4](#footnote4)</sup>.
This effectively truncates the infinite summation and allows the user to only specify $$h$$ (since $$a$$ and $$b$$ become finite).

## Conclusion

We've seen that tanhsinh quadrature can be a powerful method for numerically integrating functions with endpoint singularities or rapid changes near the boundaries.

We saw how it can be viewed in two different ways

1. a clever weighting scheme with doubly exponential roll off combined with non-uniformly distributed points with extra points concentrated near the limits of integration.
2. Riemann rectangles applied to the integral after the substitution $$x=\tanh(\pi/2 \sinh(t))$$

The first corresponds to the view in $$x$$ space and using quadrature.
The second corresponds to uniform spacing in $$t$$ space and using Riemann rectangles of equal weight.

## Footnotes

<a name="footnote1">1</a>: The rectangles don't need to be uniformly spaced but it makes things simpler. In the limit as $$h\rightarrow 0$$, this becomes equality. It's more of a definition actually. An important caveat is that this does not work when $$a$$ or $$b$$ are infinite.

<a name="footnote2">2</a>: We may hear this called zero-order hold in time series contexts. That is, the value of the function in between sampling points is assumed constant (a zero order polynomial) until the next sampling point.

<a name="footnote3">3</a>: A simple change of variables $$u = a + x (b-a)$$ allows handling arbitrary limits of integration.

<a name="footnote4">4</a>: The easiest way to do this is to find where the complement of $$x$$ achieves the smallest normal floating point number, denoted $$\epsilon$$. Some algebra shows

$$1-x = {2 \over 1+e^{\pi/2 \sinh(t)}}.$$

We then just need to compute $$t_{max} = \text{asinh}({1 \over \pi}\log\left(2/\epsilon - 1\right))$$.
For single precision it's about 4.02 and for double it's around 6.11.

## References

1. [Scipy 1.15.0 release notes](https://docs.scipy.org/doc/scipy/release/1.15.0-notes.html)
2. [Wikipedia](https://en.wikipedia.org/wiki/Tanh-sinh_quadrature)
3. [Original paper](https://ems.press/content/serial-article-files/41766)

<!-- Widget Scripts -->
{% include widget-scripts.html %}
<script>
// Left Riemann Widget
(function() {
  const width = 450, height = 440;
  const margin = { top: 30, right: 30, bottom: 50, left: 60 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xMin = 0.8, xMax = 2.2, yMin = 0, yMax = 1.3;
  const a = 1, b = 2;
  const exactArea = Math.log(2);

  const slider = document.getElementById('riemann-n-slider');
  const readout = document.getElementById('riemann-n-readout');
  const gridG = document.getElementById('riemann-grid');
  const axesG = document.getElementById('riemann-axes');
  const rectsG = document.getElementById('riemann-rects');
  const curvePath = document.getElementById('riemann-curve');
  const labelsG = document.getElementById('riemann-labels');
  const approxSpan = document.getElementById('riemann-approx');
  const exactSpan = document.getElementById('riemann-exact');
  const errorSpan = document.getElementById('riemann-error');

  function toSvgX(x) { return margin.left + (x - xMin) / (xMax - xMin) * plotW; }
  function toSvgY(y) { return margin.top + (1 - (y - yMin) / (yMax - yMin)) * plotH; }
  function f(x) { return 1 / x; }

  function drawGrid() {
    gridG.innerHTML = '';
    for (let x = 1; x <= 2; x += 0.5) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', toSvgX(x)); line.setAttribute('y1', margin.top);
      line.setAttribute('x2', toSvgX(x)); line.setAttribute('y2', height - margin.bottom);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
    for (let y = 0; y <= 1.2; y += 0.2) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', margin.left); line.setAttribute('y1', toSvgY(y));
      line.setAttribute('x2', width - margin.right); line.setAttribute('y2', toSvgY(y));
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
  }

  function drawAxes() {
    axesG.innerHTML = '';
    labelsG.innerHTML = '';
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin.left); xAxis.setAttribute('y1', toSvgY(0));
    xAxis.setAttribute('x2', width - margin.right); xAxis.setAttribute('y2', toSvgY(0));
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(xAxis);
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', margin.left); yAxis.setAttribute('y1', margin.top);
    yAxis.setAttribute('x2', margin.left); yAxis.setAttribute('y2', height - margin.bottom);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(yAxis);
    for (let x = 1; x <= 2; x += 0.5) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', toSvgX(x)); label.setAttribute('y', height - margin.bottom + 20);
      label.setAttribute('fill', '#8b949e'); label.setAttribute('font-size', '12');
      label.setAttribute('text-anchor', 'middle'); label.textContent = x.toFixed(1);
      labelsG.appendChild(label);
    }
    for (let y = 0.2; y <= 1.2; y += 0.2) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - 10); label.setAttribute('y', toSvgY(y) + 4);
      label.setAttribute('fill', '#8b949e'); label.setAttribute('font-size', '12');
      label.setAttribute('text-anchor', 'end'); label.textContent = y.toFixed(1);
      labelsG.appendChild(label);
    }
    const xTitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xTitle.setAttribute('x', width / 2); xTitle.setAttribute('y', height - 10);
    xTitle.setAttribute('fill', '#8b949e'); xTitle.setAttribute('font-size', '14');
    xTitle.setAttribute('text-anchor', 'middle'); xTitle.textContent = 'x';
    labelsG.appendChild(xTitle);
    const yTitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yTitle.setAttribute('x', 20); yTitle.setAttribute('y', height / 2);
    yTitle.setAttribute('fill', '#8b949e'); yTitle.setAttribute('font-size', '14');
    yTitle.setAttribute('text-anchor', 'middle');
    yTitle.setAttribute('transform', `rotate(-90, 20, ${height / 2})`);
    yTitle.textContent = 'f(x) = 1/x';
    labelsG.appendChild(yTitle);
  }

  function drawCurve() {
    const points = [];
    for (let x = xMin + 0.01; x <= xMax; x += 0.01) points.push([x, f(x)]);
    let d = `M ${toSvgX(points[0][0])} ${toSvgY(points[0][1])}`;
    for (let i = 1; i < points.length; i++) d += ` L ${toSvgX(points[i][0])} ${toSvgY(points[i][1])}`;
    curvePath.setAttribute('d', d);
  }

  function drawRectangles(n) {
    rectsG.innerHTML = '';
    const dx = (b - a) / n;
    let approxArea = 0;
    for (let i = 0; i < n; i++) {
      const xLeft = a + i * dx;
      const h = f(xLeft);
      approxArea += dx * h;
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', toSvgX(xLeft));
      rect.setAttribute('y', toSvgY(h));
      rect.setAttribute('width', toSvgX(xLeft + dx) - toSvgX(xLeft));
      rect.setAttribute('height', toSvgY(0) - toSvgY(h));
      rect.setAttribute('fill', 'rgba(88, 166, 255, 0.3)');
      rect.setAttribute('stroke', '#58a6ff');
      rect.setAttribute('stroke-width', 1);
      rectsG.appendChild(rect);
    }
    return approxArea;
  }

  function update() {
    const n = parseInt(slider.value);
    readout.textContent = n;
    const approxArea = drawRectangles(n);
    const error = Math.abs((approxArea - exactArea) / exactArea) * 100;
    approxSpan.textContent = approxArea.toFixed(4);
    exactSpan.textContent = exactArea.toFixed(4);
    errorSpan.textContent = error.toFixed(2) + '%';
  }

  drawGrid(); drawAxes(); drawCurve(); update();
  slider.addEventListener('input', update);
})();

// Right Riemann Widget
(function() {
  const width = 450, height = 440;
  const margin = { top: 30, right: 30, bottom: 50, left: 60 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xMin = 0.8, xMax = 2.2, yMin = 0, yMax = 1.3;
  const a = 1, b = 2;
  const exactArea = Math.log(2);

  const slider = document.getElementById('right-riemann-n-slider');
  const readout = document.getElementById('right-riemann-n-readout');
  const gridG = document.getElementById('right-riemann-grid');
  const axesG = document.getElementById('right-riemann-axes');
  const rectsG = document.getElementById('right-riemann-rects');
  const curvePath = document.getElementById('right-riemann-curve');
  const labelsG = document.getElementById('right-riemann-labels');
  const approxSpan = document.getElementById('right-riemann-approx');
  const exactSpan = document.getElementById('right-riemann-exact');
  const errorSpan = document.getElementById('right-riemann-error');

  function toSvgX(x) { return margin.left + (x - xMin) / (xMax - xMin) * plotW; }
  function toSvgY(y) { return margin.top + (1 - (y - yMin) / (yMax - yMin)) * plotH; }
  function f(x) { return 1 / x; }

  function drawGrid() {
    gridG.innerHTML = '';
    for (let x = 1; x <= 2; x += 0.5) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', toSvgX(x)); line.setAttribute('y1', margin.top);
      line.setAttribute('x2', toSvgX(x)); line.setAttribute('y2', height - margin.bottom);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
    for (let y = 0; y <= 1.2; y += 0.2) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', margin.left); line.setAttribute('y1', toSvgY(y));
      line.setAttribute('x2', width - margin.right); line.setAttribute('y2', toSvgY(y));
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
  }

  function drawAxes() {
    axesG.innerHTML = '';
    labelsG.innerHTML = '';
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin.left); xAxis.setAttribute('y1', toSvgY(0));
    xAxis.setAttribute('x2', width - margin.right); xAxis.setAttribute('y2', toSvgY(0));
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(xAxis);
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', margin.left); yAxis.setAttribute('y1', margin.top);
    yAxis.setAttribute('x2', margin.left); yAxis.setAttribute('y2', height - margin.bottom);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(yAxis);
    for (let x = 1; x <= 2; x += 0.5) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', toSvgX(x)); label.setAttribute('y', height - margin.bottom + 20);
      label.setAttribute('fill', '#8b949e'); label.setAttribute('font-size', '12');
      label.setAttribute('text-anchor', 'middle'); label.textContent = x.toFixed(1);
      labelsG.appendChild(label);
    }
    for (let y = 0.2; y <= 1.2; y += 0.2) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - 10); label.setAttribute('y', toSvgY(y) + 4);
      label.setAttribute('fill', '#8b949e'); label.setAttribute('font-size', '12');
      label.setAttribute('text-anchor', 'end'); label.textContent = y.toFixed(1);
      labelsG.appendChild(label);
    }
    const xTitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xTitle.setAttribute('x', width / 2); xTitle.setAttribute('y', height - 10);
    xTitle.setAttribute('fill', '#8b949e'); xTitle.setAttribute('font-size', '14');
    xTitle.setAttribute('text-anchor', 'middle'); xTitle.textContent = 'x';
    labelsG.appendChild(xTitle);
    const yTitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yTitle.setAttribute('x', 20); yTitle.setAttribute('y', height / 2);
    yTitle.setAttribute('fill', '#8b949e'); yTitle.setAttribute('font-size', '14');
    yTitle.setAttribute('text-anchor', 'middle');
    yTitle.setAttribute('transform', `rotate(-90, 20, ${height / 2})`);
    yTitle.textContent = 'f(x) = 1/x';
    labelsG.appendChild(yTitle);
  }

  function drawCurve() {
    const points = [];
    for (let x = xMin + 0.01; x <= xMax; x += 0.01) points.push([x, f(x)]);
    let d = `M ${toSvgX(points[0][0])} ${toSvgY(points[0][1])}`;
    for (let i = 1; i < points.length; i++) d += ` L ${toSvgX(points[i][0])} ${toSvgY(points[i][1])}`;
    curvePath.setAttribute('d', d);
  }

  function drawRectangles(n) {
    rectsG.innerHTML = '';
    const dx = (b - a) / n;
    let approxArea = 0;
    for (let i = 0; i < n; i++) {
      const xLeft = a + i * dx;
      const xRight = xLeft + dx;
      const h = f(xRight);
      approxArea += dx * h;
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', toSvgX(xLeft));
      rect.setAttribute('y', toSvgY(h));
      rect.setAttribute('width', toSvgX(xRight) - toSvgX(xLeft));
      rect.setAttribute('height', toSvgY(0) - toSvgY(h));
      rect.setAttribute('fill', 'rgba(88, 166, 255, 0.3)');
      rect.setAttribute('stroke', '#58a6ff');
      rect.setAttribute('stroke-width', 1);
      rectsG.appendChild(rect);
    }
    return approxArea;
  }

  function update() {
    const n = parseInt(slider.value);
    readout.textContent = n;
    const approxArea = drawRectangles(n);
    const error = Math.abs((approxArea - exactArea) / exactArea) * 100;
    approxSpan.textContent = approxArea.toFixed(4);
    exactSpan.textContent = exactArea.toFixed(4);
    errorSpan.textContent = error.toFixed(2) + '%';
  }

  drawGrid(); drawAxes(); drawCurve(); update();
  slider.addEventListener('input', update);
})();

// Trapezoidal Widget
(function() {
  const width = 450, height = 440;
  const margin = { top: 30, right: 30, bottom: 50, left: 60 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xMin = 0.8, xMax = 2.2, yMin = 0, yMax = 1.3;
  const a = 1, b = 2;
  const exactArea = Math.log(2);

  const slider = document.getElementById('trapezoid-n-slider');
  const readout = document.getElementById('trapezoid-n-readout');
  const gridG = document.getElementById('trapezoid-grid');
  const axesG = document.getElementById('trapezoid-axes');
  const shapesG = document.getElementById('trapezoid-shapes');
  const curvePath = document.getElementById('trapezoid-curve');
  const labelsG = document.getElementById('trapezoid-labels');
  const approxSpan = document.getElementById('trapezoid-approx');
  const exactSpan = document.getElementById('trapezoid-exact');
  const errorSpan = document.getElementById('trapezoid-error');

  function toSvgX(x) { return margin.left + (x - xMin) / (xMax - xMin) * plotW; }
  function toSvgY(y) { return margin.top + (1 - (y - yMin) / (yMax - yMin)) * plotH; }
  function f(x) { return 1 / x; }

  function drawGrid() {
    gridG.innerHTML = '';
    for (let x = 1; x <= 2; x += 0.5) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', toSvgX(x)); line.setAttribute('y1', margin.top);
      line.setAttribute('x2', toSvgX(x)); line.setAttribute('y2', height - margin.bottom);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
    for (let y = 0; y <= 1.2; y += 0.2) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', margin.left); line.setAttribute('y1', toSvgY(y));
      line.setAttribute('x2', width - margin.right); line.setAttribute('y2', toSvgY(y));
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
  }

  function drawAxes() {
    axesG.innerHTML = '';
    labelsG.innerHTML = '';
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin.left); xAxis.setAttribute('y1', toSvgY(0));
    xAxis.setAttribute('x2', width - margin.right); xAxis.setAttribute('y2', toSvgY(0));
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(xAxis);
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', margin.left); yAxis.setAttribute('y1', margin.top);
    yAxis.setAttribute('x2', margin.left); yAxis.setAttribute('y2', height - margin.bottom);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(yAxis);
    for (let x = 1; x <= 2; x += 0.5) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', toSvgX(x)); label.setAttribute('y', height - margin.bottom + 20);
      label.setAttribute('fill', '#8b949e'); label.setAttribute('font-size', '12');
      label.setAttribute('text-anchor', 'middle'); label.textContent = x.toFixed(1);
      labelsG.appendChild(label);
    }
    for (let y = 0.2; y <= 1.2; y += 0.2) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - 10); label.setAttribute('y', toSvgY(y) + 4);
      label.setAttribute('fill', '#8b949e'); label.setAttribute('font-size', '12');
      label.setAttribute('text-anchor', 'end'); label.textContent = y.toFixed(1);
      labelsG.appendChild(label);
    }
    const xTitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xTitle.setAttribute('x', width / 2); xTitle.setAttribute('y', height - 10);
    xTitle.setAttribute('fill', '#8b949e'); xTitle.setAttribute('font-size', '14');
    xTitle.setAttribute('text-anchor', 'middle'); xTitle.textContent = 'x';
    labelsG.appendChild(xTitle);
    const yTitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yTitle.setAttribute('x', 20); yTitle.setAttribute('y', height / 2);
    yTitle.setAttribute('fill', '#8b949e'); yTitle.setAttribute('font-size', '14');
    yTitle.setAttribute('text-anchor', 'middle');
    yTitle.setAttribute('transform', `rotate(-90, 20, ${height / 2})`);
    yTitle.textContent = 'f(x) = 1/x';
    labelsG.appendChild(yTitle);
  }

  function drawCurve() {
    const points = [];
    for (let x = xMin + 0.01; x <= xMax; x += 0.01) points.push([x, f(x)]);
    let d = `M ${toSvgX(points[0][0])} ${toSvgY(points[0][1])}`;
    for (let i = 1; i < points.length; i++) d += ` L ${toSvgX(points[i][0])} ${toSvgY(points[i][1])}`;
    curvePath.setAttribute('d', d);
  }

  function drawTrapezoids(n) {
    shapesG.innerHTML = '';
    const dx = (b - a) / n;
    let approxArea = 0;
    for (let i = 0; i < n; i++) {
      const xLeft = a + i * dx;
      const xRight = xLeft + dx;
      const hLeft = f(xLeft);
      const hRight = f(xRight);
      approxArea += (dx / 2) * (hLeft + hRight);
      const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      const points = [
        `${toSvgX(xLeft)},${toSvgY(0)}`,
        `${toSvgX(xLeft)},${toSvgY(hLeft)}`,
        `${toSvgX(xRight)},${toSvgY(hRight)}`,
        `${toSvgX(xRight)},${toSvgY(0)}`
      ].join(' ');
      poly.setAttribute('points', points);
      poly.setAttribute('fill', 'rgba(88, 166, 255, 0.3)');
      poly.setAttribute('stroke', '#58a6ff');
      poly.setAttribute('stroke-width', 1);
      shapesG.appendChild(poly);
    }
    return approxArea;
  }

  function update() {
    const n = parseInt(slider.value);
    readout.textContent = n;
    const approxArea = drawTrapezoids(n);
    const error = Math.abs((approxArea - exactArea) / exactArea) * 100;
    approxSpan.textContent = approxArea.toFixed(4);
    exactSpan.textContent = exactArea.toFixed(4);
    errorSpan.textContent = error.toFixed(2) + '%';
  }

  drawGrid(); drawAxes(); drawCurve(); update();
  slider.addEventListener('input', update);
})();
</script>
