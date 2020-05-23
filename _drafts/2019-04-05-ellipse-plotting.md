---
title: "Plotting Ellipses in Numpy"
categories:
  - Mathematics
date:   2020-05-20 10:57:00 +0100
mathjax: true
tags:
  - matplotlib
  - python
toc: true
# classes: wide
excerpt: ""
header: 
  overlay_image: assets/images/shakespeare-zipf-param-surface-splash.png
  overlay_filter: 0.2
---

Kepler's laws of planetary motion, confidence intervals of multivariate normal distributions, level sets of least squares are just a few of the many instances in which ellipses appear in applied mathematics. This post will discuss a couple methods of how to plot arbitrary 2D ellipses.


## Conic Form
The most common definition of the ellipse is usually encountered when studying conic sections somewhere around 10th grade. The conic form of the ellipse is given by

$$
\begin{align*}
Ax^2 + Bxy + Cy^2 + Dx + Ey + F &= 0\\
B^2 - 4AC &< 0
\end{align*}
$$

where the quantity $$B^2 - 4AC$$ is called the _discriminant_ of the conic section. Evidently this form is not particularly amenable to plotting software such as `matplotlib` since $$y$$ is defined implicitly for each value of $$x$$ rather than being an explicit function of $$x$$.

## Quadratic Form
After a course in linear algebra, it is a simple exercise in matrix multiplication to verify the conic representation is equivalent to the following equation

$$
\begin{align*}
z^TQz + b^Tz + F = 0\\
Q\in \mathbf{S}^2_{++}
\end{align*}
$$

where

$$
\begin{align*}
Q = 
\begin{bmatrix}
A & B/2\\ 
B/2 & C
\end{bmatrix}, \,
b = \begin{bmatrix}
D \\
E
\end{bmatrix}, \,
z =  \begin{bmatrix}
x \\
y
\end{bmatrix} 
\end{align*}
$$

The constraint $$Q\in \mathbf{S}^2_{++}$$ is a more compact way of saying that $$Q\in \mathbf{R}^{2\times 2}$$, $$Q=Q^T$$, and $$\mathbf{det}(Q) > 0$$.<sup>[1](#footnote3)</sup>

Analogous to completing the square for a single variable quadratic $$ax^2+bx+c=0$$ in which $$\frac{b^2}{4a}$$ is added and substracted to eliminate the linear term

$$a(x - b/2a)^2 + c - b^2/4a = 0,$$

removing the linear term in the multidimensional case can be accomplished by adding and subtracting $$\frac{1}{4}b^TQ^{-1}b$$ to arrive at the following description of an ellipse

$$(z - Q^{-1}b)^TQ(z-Q^{-1}b) + F - \frac{1}{4}b^TQ^{-1}b = 0$$

Provided we don't have a degenerate case in which $$F - \frac{1}{4}b^TQ^{-1}b = 0$$, this can be rewritten in an even more compact form <sup>[2](#footnote3)</sup>

$$(z - c)^TP(z-c) = 1$$ 

This is an ellipse expressed as a _quadratic form_. Note that if $$P=r^{-2}I$$, this reduces to the familiar equation of a circle with radius $$r$$. When $$P=\mathbf{diag}(1/a^2, 1/b^2)$$, we have the familiar equation of an axis aligned ellipse

$$\frac{(x-x_c)^2}{a^2} + \frac{(y-y_c)^2}{b^2} = 1$$

Though mathematically simple, this equation is still not amenable to plotting as there is no simple way to get all the valid $$x,y$$ pairs satisfying it

## Parametric Form
Since $$P$$ is symmetric, the spectral theorem guarantees the matrix can be diagonalised as $$P=V\mathbf{diag}(\lambda_1, \lambda_2)V^T$$ where $$V^T=V^{-1}$$. Since an ellipse is the set of points 

$$\{z : (z-c)^TP(z-c)=1\}$$ 

we can substitute the factorised form for $$P$$ to arrive at a new form to express an ellipse

$$
\begin{align*}
\{z: (z-c)^TV\mathbf{diag}(\lambda_1, \lambda_2)V^T(z-c) = 1\}\\
\left\{z: \lVert\mathbf{diag}\left(\lambda_1^{1/2},\lambda_2^{1/2}\right)V^T(z-c)\rVert^2 = 1\right\}\\
\left\{z:  z = V\mathbf{diag}\left(\lambda_1^{-1/2}, \lambda_2^{-1/2}\right)u + c,\, ||u||^2 = 1\right\}\\
\end{align*}
$$

This description, though less elegant than the quadratic form, is actually an exact description of how to plot an ellipse. 

The _parametric form_ of the ellipse says that to create an ellipse, start with the unit circle $$\{u : ||u||^2=1\}$$. Then stretch the unit circle by $$\lambda_1^{-1/2}$$ in the $$x$$ direction and $$\lambda_2^{-1/2}$$ in the $$y$$ direction by applying the linear transformation $$\mathbf{diag}\left(\lambda_1^{-1/2}, \lambda_2^{-1/2}\right)$$. Then rotate this axis aligned ellipse to the appropriate orientation by multiplying by the matrix $$V$$ (recall that $$V^{-1}=V^T$$ so $$V$$ represents a rotation and/or reflection). Lastly, translate the ellipse to the appropriate center by adding the vector $$c$$

The steps in the transformation are illustrated in the animation below for the ellipse given by

$$
\left(
\begin{bmatrix}
x\\
y\\
\end{bmatrix} - 
\begin{bmatrix}
1\\
-1\\
\end{bmatrix}
\right)^T
\begin{bmatrix}
1 & -0.2\\
-0.2 & 0.4\\
\end{bmatrix}
\left(
\begin{bmatrix}
x\\
y\\
\end{bmatrix} -
\begin{bmatrix}
1\\
-1\\
\end{bmatrix}
\right) = 1
$$

<figure>
    <a href="/assets/images/shakespeare-levenberg-marquardt-fit.png"><img src="/assets/images/shakespeare-levenberg-marquardt-fit.png"></a>
    <figcaption>Figure 5</figcaption>
</figure>

Lastly, note that all 4 starting points converged to the same solution despite the non-convexity of the objective. Furthermore, the solution is identical to that of Gauss-Newton. 

## Model Comparison
Having solved both the NLLS and the log space OLS problem, we can compare the models that result. The first thing to notice is that the formulations do not yield the same $$(K^\star, \alpha^\star)$$. Indeed the optimisation problems are genuinely different and not just a simple change of variables (which would leave the optimum unaltered). 

We can plot both resulting models as in Figure 6 and notice the qualitative differences between them. The model resulting from NLLS does a much better job fitting the first few high ranking word frequencies compared to the OLS model which shows very large errors (as measured by the vertical distance between the point and the graph). In the lower ranking words however, the NLLS model shows a consistent overestimation of the word frequency that the OLS model does not. In log space, this pattern is even more pronounced

<figure class="half">
    <a href="/assets/images/shakespeare-zipf-fit.png"><img src="/assets/images/shakespeare-zipf-fit.png"></a>
    <a href="/assets/images/shakespeare-zipf-fit-loglog.png"><img src="/assets/images/shakespeare-zipf-fit-loglog.png"></a>
    <figcaption>Figure 6</figcaption>
</figure>

## Conclusion
Nonlinear models appear all the time in math and physics. Fitting parameters does not always require a reduction to a linear model (though this may be what you want) and we have seen two methods for handling the nonlinear case. In fact, we have seen the algorithms for nonlinear least squares use linear least squares as a subroutine to iteratively refine the solution. 

Lastly, it is worth noting that even though, we have written an implementation of an NLLS solver, in practice, you should always use something like `scipy.optimize`'s `least_squares` method. Thousands of man hours of work have gone into creating efficient solvers, the result of which is many clever optimisations and corner-case handling on top of the vanilla implementation. 

## Footnotes
<a name="footnote1">1</a>: More generally, $$Q\in \mathbf{S}^n_{++}$$, means that $$Q$$ is a positive semidefinite matrix. When $$n=2$$, a positive determinant is sufficient to ensure this. In fact, it is identical to the discriminant condition 
<a name="footnote2">2</a>: Here, $$c=Q^{-1}b$$ is the center of the ellipse and $$P = Q/(\frac{1}{4}b^TQ^{-1}b - F)$$ 
<a name="footnote3">3</a>: Since $$\frac{\partial ||r(x)||^2}{\partial x_j} = \sum_{i=1}^m \frac{\partial}{\partial x_j}(r_i^2(x)) =  \sum_{i=1}^m 2r_i(x)\frac{\partial r_i(x)}{\partial x_j}$$ it follows that $$\nabla_x ||r(x)||^2 = (Dr(x))^Tr(x)$$


## References
1. [Zipf's Law Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law)
2. [Gauss-Newton Algorithm Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)
3. [Stanford's Intro to Linear Dynamical Systems](http://ee263.stanford.edu/)
4. [scipy.optimize Notes on Least Squares Implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares)
5. [Boyd & Vandenberghe's Intro to Applied Linear Algebra](http://vmls-book.stanford.edu/)
