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

## Polynomial Interpolation

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
\end{bmatrix}.
$$

This is of course a lower triangular matrix and can easily be solved with back substitution.