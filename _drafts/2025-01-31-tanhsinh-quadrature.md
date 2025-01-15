---
title: "Tanhsinh Quadrature"
categories:
  - Mathematics
date:   2025-01-11 14:33:00 +0100
mathjax: true
tags:
  - Numerical methods
  - Integration
toc: true
# classes: wide
excerpt: ""
header: 
  overlay_image: assets/tanhsinh-quadrature/images/splash_image.png
  overlay_filter: 0.2
---

## Motivation

The fundamental theorem of calculus gives us a deceptively simple method of evaluating a definite integral as the difference of its antiderivate at each end point

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

Where each term in the sum represents the area of one of our recatangles.
The width of each block is $$h={b-a \over N}$$ (the total interval divided into $$n$$ pieces), and the height is $$f\left(a + {b-a \over N} k\right)$$ (the function evaluated at the left edge of each block)<sup>[1](#footnote1)</sup>.

<figure class>
    <a href="/assets/tanhsinh-quadrature/images/left_riemann.png"><img src="/assets/tanhsinh-quadrature/images/left_riemann.png"></a>
    <figcaption>Figure 1: Left handed Riemann approximation using 10 rectangles.</figcaption>
</figure>

We can also approximate the area with rectangles measured from the top right corner just by incrementing the indices of summation.

$$
\int_a^b f(x)dx \approx \sum_{k=1}^{N} hf\left(a + hk\right)
$$

<figure class>
    <a href="/assets/tanhsinh-quadrature/images/right_riemann.png"><img src="/assets/tanhsinh-quadrature/images/right_riemann.png"></a>
    <figcaption>Figure 2: Right handed Riemann approximation using 10 rectangles.</figcaption>
</figure>

We can think of Riemann sums as approximating the function as piecewise constant<sup>[2](#footnote2)</sup>.

## Trapezoidal approximation

A slightly more sophisticated method is to approximate the area as the sum of trapezoids.
We can think of this as approximating the function as piecewise _linear_.

Recalling that the area of trapezoid is $${height \over 2}(base_1 + base_2)$$ we get the slightly more complicated approximation

$$
\int_a^b f(x)dx \approx \sum_{k=0}^{N-1} {h \over 2}\left[f\left(a + hk\right) + f\left(a + h(k+1)\right)\right].
$$

<figure class>
    <a href="/assets/tanhsinh-quadrature/images/trapezoid.png"><img src="/assets/tanhsinh-quadrature/images/trapezoid.png"></a>
    <figcaption>Figure 3: Trapezoidal approximation using 10 trapezoids.</figcaption>
</figure>

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
    <a href="/assets/tanhsinh-quadrature/images/shifted_rsqrt.png"><img src="/assets/tanhsinh-quadrature/images/shifted_rsqrt.png"></a>
    <figcaption>Figure 4: Function with an asymptote at its right endpoint.</figcaption>
</figure>

Applying our quadrature methods so far leads summations in which the last term is infinite, completely destroying the approximation.

One thing we _could_ try to do if we were programming a library is to only sum the values of `f(x_k)` which are finite.
Our approximation is saved from blowing up to infinity but we still lose accuracy.

In the next section we'll see a method for taming these endpoint asymptotes.

## tanh-sinh quadrature

Why do our previous methods struggle with endpoint asymptotes?
The problem is that we're sampling points uniformly in our interval.
Near the asymptote, the function changes so rapidly that a reasonable number of uniform samples doesn't capture its behavior.

Tanh-sinh quadrature transforms our integral in a way that "tames" these asymptotes.

In this section we'll restrict our attention to integrals in the interval $$[-1,1]$$ for simplicity<sup>[3](#footnote3)</sup>.
We'll start by making the rather unusual variable subsitution

$$x = \tanh\left(\sinh \left({\pi \over 2} t\right)\right)$$

This function is plotted in Figure 5 along with the regular hyperbolic tangent.

<figure class>
    <a href="/assets/tanhsinh-quadrature/images/tanhsinh.png"><img src="/assets/tanhsinh-quadrature/images/tanhsinh.png"></a>
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

With this subsitution, our integral becomes the doubly improper, and rather complicated looking,

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
    <a href="/assets/tanhsinh-quadrature/gifs/output-onlinegiftools.gif"><img src="/assets/tanhsinh-quadrature/gifs/output-onlinegiftools.gif"></a>
    <figcaption>Figure 6: The original integrand being transformed after the tanhsinh substitution. The colors are just to keep track of where each segment of the original graph gets mapped to in the end.</figcaption>
</figure>

Have a look again at the integrand before and after the tanhsinh substitution.

<figure class="half">
    <a href="/assets/tanhsinh-quadrature/images/integrand.png"><img src="/assets/tanhsinh-quadrature/images/integrand.png"></a>
    <a href="/assets/tanhsinh-quadrature/images/integrand_tanhsinh.png"><img src="/assets/tanhsinh-quadrature/images/integrand_tanhsinh.png"></a>
    <figcaption>Figure 7: The left shows the original integrand while the right shows the integrand after the tanhsinh substitution</figcaption>
</figure>

The figure makes it clear that the the asymptote at 1 disappears!
What's more, the function decays _extremely_ quickly to 0.
To get a sense of just how fast this function decays to 0, compare with the standard Gaussian $$e^{-x^2}$$ which is  $$1.23\times 10^{-4}$$ at $$x=-3$$.
Our integrand down to $$9.60\times 10^{-13}$$ at the same value of $$x$$.
Over eight orders of magnitude smaller!

The exceedingly fast rolloff makes accurate numerical integration much easier since $$f(x_k)$$ will quickly go to 0 and contribute nothing to the sum.
This allows us to effectively truncate the summation without very negligible error.

We can now use Riemann sums on this transformed integrand to get an accurate approximation of the original integral. In the case of proper integrals, the number of points explicitly defined the width of the rectangle, but when $$a$$ or $$b$$ are infinite, we have to choose them independently.
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
    <a href="/assets/tanhsinh-quadrature/images/tanhsinh_riemann.png"><img src="/assets/tanhsinh-quadrature/images/tanhsinh_riemann.png"></a>
    <figcaption>Figure 8: Left Riemann rectangles of the integrand after the tanhsinh substitution using 20 rectangles between -2 and 2</figcaption>
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

## Conclusion

## Footnotes

<a name="footnote1">1</a>: The rectangles don't need to be uniformly spaced but it makes things simpler. In the limit as $$h\rightarrow 0$$, this becomes equality. It's more of a definition actually. An important caveat is that this does not work when $$a$$ or $$b$$ are infinite.

<a name="footnote2">2</a>: We may hear this called zero-order hold in time series contexts. That is, the value of the function in between sampling points is assumed constant (a zero order polynomial) until the next sampling point.

<a name="footnote3">3</a>: A simple change of variables $$u = a + x (b-a)$$ allows handling arbitrary limits of integration.

## References

1. [Scipy 1.15.0 release notes](https://docs.scipy.org/doc/scipy/release/1.15.0-notes.html)
2. [tanhsinh quadrature](https://en.wikipedia.org/wiki/Tanh-sinh_quadrature)
