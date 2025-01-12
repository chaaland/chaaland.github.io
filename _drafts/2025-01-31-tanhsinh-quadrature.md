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

$$\int_a^b f(x) dx = F(b) - F(a)$$.

But what happens when the function has no nice antiderivative?
What do you if you want to write code that does integration but the user only provides function to integrate, $$f(x)$$?

In the release notes of [Scipy 1.15.0](https://docs.scipy.org/doc/scipy/release/1.15.0-notes.html) there is a new feature called `tanhsinh` in the `scipy.integrate` subpackage.
In this post we'll see how this method can be used to accurately approximate definite integrals.

## Riemann approximation

The simplest method you learn for approximating the integral is the Riemann sum.
This approximates the area under the curve as the sum of many small rectangles.

$$
\begin{align*}
\int_a^b f(x)dx &\approx \sum_{k=0}^{n-1} \left({b-a \over n}\right)f\left(a + {b-a \over n} k\right)\\
&\approx \sum_{k=0}^{n-1} hf\left(a + hk\right)
\end{align*}
$$

Where we've let $$h= {b-a \over n}$$ for ease of notation<sup>[1](#footnote1)</sup>.

<figure class>
    <a href="/assets/tanhsinh-quadrature/images/left_riemann.png"><img src="/assets/tanhsinh-quadrature/images/left_riemann.png"></a>
    <figcaption>Figure 1: Left handed Riemann approximation using 10 rectangles.</figcaption>
</figure>

We can also approximate the area with rectangles measured from the top right corner just by incrementing the indices of summation.

$$
\int_a^b f(x)dx \approx \sum_{k=1}^{n} hf\left(a + hk\right)
$$

<figure class>
    <a href="/assets/tanhsinh-quadrature/images/right_riemann.png"><img src="/assets/tanhsinh-quadrature/images/right_riemann.png"></a>
    <figcaption>Figure 2: Right handed Riemann approximation using 10 rectangles.</figcaption>
</figure>

You can think of Riemann sums as approximating the function as piecewise constant<sup>[2](#footnote2)</sup>. 

## Trapezoidal approximation

A slightly more sophisticated method is to approximate the area as the sum of trapezoids.
You can think of this as approximating the function as piecewise _linear_.

Recalling that the area of trapezoid is $${height \over 2}(base_1 + base_2)$$ we get the slightly more complicated approximation

$$
\int_a^b f(x)dx \approx \sum_{k=0}^{n-1} {h \over 2}\left[f\left(a + hk\right) + f\left(a + h(k+1)\right)\right].
$$

<figure class>
    <a href="/assets/tanhsinh-quadrature/images/trapezoid.png"><img src="/assets/tanhsinh-quadrature/images/trapezoid.png"></a>
    <figcaption>Figure 3: Trapezoidal approximation using 10 trapezoids.</figcaption>
</figure>

## Quadrature

The Riemann and trapezoidal methods are special cases of quadrature which is the following form of integral approximation

$$\int_a^b f(x)dx \approx \sum_{k=0}^n w_k f(x_k)$$

In the left handed Riemann approximation,

$$
\begin{align*}
w_k &= 
\begin{cases} 
    0 & k=n \\
    {b-a \over n} & \text{otherwise}
\end{cases}\\
x_k &= a + \left({b-a \over n}\right)k
\end{align*}
$$

The right handed Riemann approximation is the same except the first weight is zero rather than the last.

$$
\begin{align*}
w_k &=
\begin{cases}
    0 & k=0 \\
    {b-a \over n} & \text{otherwise}
\end{cases}\\
x_k &= a + \left({b-a \over n}\right)k
\end{align*}
$$

The trapezoidal rule has nearly the same weights except the endpoints are half the other weights.

$$
\begin{align*}
w_k &=
\begin{cases}
    {1 \over 2}{b-a \over n} & k=0 \text{ or } n\\
    {b-a \over n} & \text{otherwise}
\end{cases}\\
x_k &= a + \left({b-a \over n}\right)k
\end{align*}
$$

Quadrature then amounts to the following Python code

{% highlight python %}
def quadrature(f, a=-1, b=1, n_points=10):
    x_k, w_k = compute_points(a, b, n_points)
    return np.dot(w_k, f(x_k))
{% endhighlight %}

where `compute_points` implements one of the above formulae.

The exact value of the $$\int_1^2 {1 \over x} dx$$ is $$\log 2$$.
The different quadrature method's accuracy is shown in the following table.

| method        | value     | % error  |
| ------------- | --------- | -------- |
| left riemann  | 0.7188    | 3.697%   |
| right riemann | 0.6688    | 3.517%   |
| trapezoidal   | 0.6938    | 0.09006% |
| exact         | 0.6931    | 0%       |

With only 10 function evaluations, that's not too bad.

But what if the function has an asymptote at one of the endpoints?
Consider integrating $${1 \over \sqrt{1-x}}$$ over the interval [-1, 1].

<figure class>
    <a href="/assets/tanhsinh-quadrature/images/shifted_rsqrt.png"><img src="/assets/tanhsinh-quadrature/images/shifted_rsqrt.png"></a>
    <figcaption>Figure 4: Function with an asymptote at its right endpoint.</figcaption>
</figure>

Applying our quadrature methods so far leads summations in which the last term is infinite completely destroying the approximation.

One thing you _could_ try to do if you were programming a library is to only sum the values of `f(x_k)` which are finite.
Your approximation is saved from blowing up to infinity but you still lose accuracy.

In the next section we'll see a method for taming these endpoint asymptotes.

## tanh-sinh quadrature

In this section we'll restrict our attention to integrals in the interval [-1,1]<sup>[3](#footnote3)</sup> for simplicity.
We'll start by making the rather unusual variable subsitution

$$x = \tanh\left(\sinh \left({\pi \over 2} t\right)\right)$$

This function is plotted in Figure 5 along with regular hyperbolic tangent.

<figure class>
    <a href="/assets/tanhsinh-quadrature/images/tanhsinh.png"><img src="/assets/tanhsinh-quadrature/images/tanhsinh.png"></a>
    <figcaption>Figure 5: The tanhsinh function saturates much more quickly than the regular hyperbolic tangent.</figcaption>
</figure>

Recalling that the derivatives of the hyperbolic functions are essentially the same as their trig counterparts and applying the chain rule, we have

$$
\begin{align*}
{dx \over dt} &= \text{sech}^2\left(\sinh \left({\pi \over 2} t\right)\right) {d\over dt}\left(\sinh \left({\pi \over 2} t\right)\right)\\
&= {\pi \over 2}\text{sech}^2\left(\sinh \left({\pi \over 2} t\right)\right) \cosh \left({\pi \over 2} t\right)
\end{align*}
$$

With this subsitution, our integral becomes rather complicated looking

$$\int_{-1}^1 f(x) dx = \int_a^b f\left(\tanh\left(\sinh \left({\pi \over 2} t\right)\right)\right){\pi \over 2}\text{sech}^2\left(\sinh \left({\pi \over 2} t\right)\right) \cosh \left({\pi \over 2} t\right) dt
$$

Using this graph we see

$$
\begin{align*}
\lim_{t\rightarrow \infty} \tanh\left(\sinh \left({\pi \over 2} t\right)\right) &= 1\\
\lim_{t\rightarrow -\infty} \tanh\left(\sinh \left({\pi \over 2} t\right)\right) &= -1
\end{align*}
$$

The derivatives of the hyperbolic functions

## Conclusion

## Footnotes

<a name="footnote1">1</a>: The rectangles don't need to be uniformly spaced but it makes things simpler. In the limit as $$h\rightarrow 0$$, this becomes equality. It's more of a definition actually. An important caveat is that this does not work when $$a$$ or $$b$$ are infinite.

<a name="footnote2">2</a>: You may hear this called zero-order hold in time series contexts. That is, the value of the function in between sampling points is assumed constant (a zero order polynomial) until the next sampling point.

<a name="footnote3">2</a>: A simple change of variables $$u = a + x (b-a)$$ allows handling arbitrary limits of integration.

## References

1. [Lagrange Polynomial](https://en.wikipedia.org/wiki/Lagrange_polynomial)
