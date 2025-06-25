---
title: "Newton-Gregory Interpolation"
categories:
  - Mathematics
date:   2025-06-01 12:30:00 +0100
mathjax: true
tags:
  - Numerical methods
  - Interpolation
toc: true
# classes: wide
excerpt: ""
header: 
  overlay_image: assets/newton-gregory/images/splash_image.png
  overlay_filter: 0.2
---

This post explores an alternative algorithm for interpolation using polynomials.
We'll also see how it can be viewed as a discrete analogue of Taylor polynomials.

## Polynomial Interpolation Recap

As discussed in a [previous post]({{ site.baseurl }}/algorithms/lagrange-interpolation), we can interpolate a quadratic polynomial by finding the coefficients of the polynomial, $$a_i$$

$$y = a_2 x^2 + a_1 x + a_0.$$

Given 3 points $$(x_1, y_1), (x_2, y_2), (x_3, y_3)$$, this leads to the system of equations

$$
\begin{bmatrix}
y_0\\
y_1\\
y_2\\
\end{bmatrix}
=
\begin{bmatrix}
x_0^2 & x_0 & 1\\
x_1^2 & x_1 & 1\\
x_2^2 & x_2 & 1\\
\end{bmatrix}
\begin{bmatrix}
a_2\\
a_1\\
a_0\\
\end{bmatrix}
$$

But choosing the Lagrange basis, our polynomial could be written as

$$
\begin{align*}
y &= a_0 {(x-x_1)(x-x_2) \over (x_0 - x_1)(x_0 - x_2)} + a_1 {(x-x_0)(x-x_2) \over (x_1 - x_0)(x_1 - x_2)} + a_2 {(x-x_0)(x-x_1) \over (x_2 - x_0)(x_2 - x_1)}\\
&= a_0 \ell_0(x) + a_1 \ell_1(x) + a_2 \ell_2(x)\\
\end{align*}
$$

By evaluating the polynomial at $$x_0, x_1,$$ and $$x_2$$, we get the trivial system of equations

$$
\begin{bmatrix}
y_0\\
y_1\\
y_2\\
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1\\
\end{bmatrix}
\begin{bmatrix}
a_0\\
a_1\\
a_2\\
\end{bmatrix}
$$

## Newton-Gregory Interpolation

The subject of this post is yet another parameterisation of the interpolating polynomial, this time given by

$$ y = a_0 + a_1 (x-x_0) + a_2(x-x_0)(x-x_1)$$

Enforcing that this polynomial pass through the points $$(x_1, y_1), (x_2, y_2), (x_3, y_3)$$ we get the system of equations

$$
\begin{bmatrix}
y_0\\
y_1\\
y_2\\
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 0\\
1 & x_1-x_0 & 0\\
1 & x_2-x_0 & (x_2-x_0)(x_2 -x_1)\\
\end{bmatrix}
\begin{bmatrix}
a_0\\
a_1\\
a_2\\
\end{bmatrix}.
$$

If we make the further assumption that the $$x$$ coordinates are evenly spaced (as they might be when interpolating points from a lookup table), we get the following system

$$
\begin{bmatrix}
y_0\\
y_1\\
y_2\\
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 0\\
1 & h & 0\\
1 & 2h & 2h^2\\
\end{bmatrix}
\begin{bmatrix}
a_0\\
a_1\\
a_2\\
\end{bmatrix}
$$

This is of course a lower triangular matrix and can easily be solved with back substitution.
From the first equation we have immediately $$a_0 = y_0$$.
From the second equation we have

$$
\begin{align*}
y_1 &= a_0 + ha_1\\
&= y_0 + ha_1
\end{align*}
$$

which gives

$$a_1 = (y_1 - y_0) / h.$$

Denoting $$\Delta y_i = y_i - y_{i-1}$$, we can write this more compactly as $$a_1 = {\Delta y_1 \over h}$$.
The final equation is

$$
\begin{align*}
y_2 &= a_0 + 2 h a_1 + 2h^2 a_2\\
&= y_0 + 2(y_1 - y_0) + 2h^2 a_2
\end{align*}
$$

which gives

$$a_2 = {y_2 - 2y_1 + y_0 \over 2h^2}.$$

Denoting

$$
\begin{align*}
\Delta^2 y_i &= \Delta (\Delta y_i) \\
&= \Delta (y_i - y_{i-1}) \\
&= \Delta y_i - \Delta y_{i-1} \\
&= (y_i - y_{i-1}) - (y_{i-1} - y_{i-2})\\
&= y_i - 2y_{i-1} + y_{i-2}
\end{align*}
$$

we can write write this succinctly as $$a_2 = {\Delta^2 y_2 \over 2h^2}$$.
Our interpolating polynomial is then

$$
\begin{align*}
y &= y_0 + {y_1 - y_0 \over h}(x-x_0) + {y_2 - 2y_1 + y_0 \over 2h^2}(x - x_0)(x - x_1)\\
&= y_0 + {\Delta y_1 \over h}(x-x_0) + {\Delta^2 y_2 \over 2h^2}(x - x_0)(x - x_0 - h)\\
&= y_0 +{x-x_0 \over h} \Delta y_1 + {x - x_0 \over h} {x - x_0 - h \over h}{\Delta^2 y_2 \over 2}\\
\end{align*}
$$

Defining $$ u = {x-x_0 \over h}$$, the polynomial simplifies to

$$
y= y_0 + u \Delta y_1  + {u(u-1) \over 2} \Delta^2 y_2
$$

The pattern for the system of equations might not be entirely obvious from the quadratic case, so let's consider the case of a fourth degree polynomial

$$
\begin{align*}
y = &a_0 + a_1 (x-x_0) + a_2(x-x_0)(x-x_1) \\
&+ a_3(x-x_0)(x-x_1)(x-x_2) + a_4(x-x_0)(x-x_1)(x-x_2)(x-x_3) 
\end{align*}
$$

Plugging in evenly spaced points $$x_i = x_0 + i\cdot h$$ and their corresponding abscissae, we get the following system of equations

$$

\begin{bmatrix}
y_0\\
y_1\\
y_2\\
y_3\\
y_4
\end{bmatrix}
=

\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
1 & h & 0 & 0 & 0\\
1 & 2h & 2h^2 & 0 & 0\\
1 & 3h & 3 \cdot 2h^2 & 3\cdot 2 \cdot 1 h^3 & 0\\
1 & 4h & 4\cdot 3 h^2 & 4\cdot 3 \cdot 2 h^3 & 4 \cdot 3 \cdot 2 \cdot 1 h^4\\
\end{bmatrix}
\begin{bmatrix}
a_0\\
a_1\\
a_2\\
a_3\\
a_4
\end{bmatrix}.
$$
