---
title: "Gaussian Process Intuition"
categories:
  - Mathematics
date:   2020-09-01 10:57:00 +0100
mathjax: true
tags:
  - Gaussian Processes
  - Regression
toc: true
# classes: wide
# excerpt: ""
# header: 
#   # make some GP with confidence interval shaded
#   overlay_image: assets/gaussian-processes/images/.png
#   overlay_filter: 0.2
---

This post is a light introduction to Gaussian Processes (GPs). The goal is to give an explanation of some of key results and most importantly, some intuition for how they come about.

## GP Intuition
In the case of regression, a GP features $$X\in \mathbf{R}^{n\times d}$$ and corresponding target $$y\in \mathbf{R}^n$$ and makes predictions for a new point $$z\in\mathbf{R}^d$$ by taking a weighted sum of $$y$$.
The weights are chosen so that points "nearer" to $$z$$ have their targets weighted higher than points futher away.
The underlying assumption is that the forecast for $$z$$ should be similar to the target values of "nearby" points.

<!-- A good starting point for discussing GPs is the interpolation problem. 
Given $$n$$ data points $$(x_i, y_i) \in \mathbf{R}^d \times \mathbf{R}$$, find a curve that passes through all $$n$$ points.
The following animation shows that there are many possible curves that can pass through $$n$$ points. -->

<figure class>
    <a href="/assets/gifs/gaussian-processes/1d-gpr.gif"><img src="/assets/gifs/gaussian-processes/1d-gpr.gif"></a>
</figure>


## Covariance Functions
The notion of "nearness" for GPs is given by a _covariance function_ $$k: \mathbf{R}^{d} \times \mathbf{R}^{d} \rightarrow \mathbf{R}$$. Some example covariance functions include the exponentiated quadratic 

$$k(x^{(1)}, x^{(2)}) = \exp\left(-\frac{||x^{(1)} - x^{(2)}||^2}{2\ell^2}\right)$$

the rational quadratic

$$k(x^{(1)}, x^{(2)}) = \left(1 + \frac{||x^{(1)} - x^{(2)}||^2}{2\alpha\ell^2}\right)^{-\alpha}$$

and the Matern covariance (of order 0)

$$k(x^{(1)}, x^{(2)}) = \sigma^2 \exp\left(-\frac{||x^{(1)} - x^{(2)}||}{\ell}\right)$$

The quantity $$\ell$$ is a hyperparameter called the _length scale_. Very roughly, it determines how close points need to be in order to be considered "near". In some situations it makes sense to have one length scale per feature dimension in which case the similarity is a function of the Mahalonobis distance

$$\left(x^{(1)}-x^{(2)}\right)^T\mathbf{diag}(\ell^{-2}_1, \ell^{-2}_2,\cdots, \ell^{-2}_n)\left(x^{(1)}-x^{(2)}\right)$$

which of course reduces to 
$$\frac{1}{\ell^2}||x^{(1)} - x^{(2)}||^2$$ 
in the special case $$\ell_1 = \ell_2 =\cdots=\ell_n$$.

The criteria for a function to be a valid covariance function is that the matrix with elements $$\Sigma_{ij} = k(x^{(i)}, x^{(j)})$$ form a valid covariance matrix (i.e. $$\Sigma \in \mathbf{S}_+^n$$)

## Gaussian Distributions
### Univariate Gaussian
The single variable Gaussian distribution has probability density function (pdf) 

$$f_X(x) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

where $$x\in \mathbf{R}$$ is a Gaussian random variable, $$\mu \in \mathbf{R}$$ is the mean of the distribution, and $$\sigma^2$$ is the variance. Typically a random variable with this distribution is written $$X \sim \mathcal{N}(\mu, \sigma^2)$$. The following animation demonstrates how the shape of the pdf changes as a function of $$\mu$$ and $$\sigma^2$$.

<figure class>
    <a href="/assets/gifs/gaussian-processes/1d-gaussian.gif"><img src="/assets/gifs/gaussian-processes/1d-gaussian.gif"></a>
</figure>

### Multivariate Gaussian
We can of course measure two independent Gaussian random variables. The  joint density of these two observations $$x_1$$ and $$x_2$$ is just the product of two single variable pdf's

$$
\begin{align*}
f_{X_1,X_2}(x_1,x_2) &= f_{X_1}(x_1)f_{X_2}(x_2) \\
 &= \frac{1}{\sqrt{2\pi \sigma_1^2}}\exp\left(-\frac{(x_1-\mu_1)^2}{2\sigma_1^2}\right)\frac{1}{\sqrt{2\pi \sigma_2^2}}\exp\left(-\frac{(x_2-\mu_2)^2}{2\sigma_2^2}\right)\\
 &= \frac{1}{2\pi\sqrt{\sigma_1^2 \sigma_2^2}}\exp\left(-\frac{(x_1-\mu_1)^2}{2\sigma_1^2}-\frac{(x_2-\mu_2)^2}{2\sigma_2^2}\right)\\
\end{align*}
$$

Denoting 

$$
\begin{align*}
\Sigma &= \mathbf{diag}(\sigma_{x_1}^2, \sigma_{x_2}^2) \in \mathbf{R}^{2\times 2}\\
x &= [x_1; x_2]\in \mathbf{R}^{2}\\
\mu &= [\mu_{x_1}; \mu_{x_2}]\in \mathbf{R}^2\\
\end{align*}
$$

the pdf can be written concisely as

$$f_X(x) = \frac{1}{2\pi \cdot \mathbf{det}(\Sigma)^{1/2}}\exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x - \mu)\right)$$

where $$x\in \mathbf{R}^2$$ is a Gaussian random vector, $$\mu \in \mathbf{R}^2$$ is the mean of the distribution, and $$\Sigma\in \mathbf{R}^{2\times 2}$$ is the covariance matrix<sup>[1](#footnote1)</sup>. 

This particular expression for the 2D Gaussian was derived for the special case where the components of the random vector were independent. Now consider the random vector given by

$$
x = Az
$$

where $$A\in \mathbf{R}^{n\times n}$$ and $$z\sim \mathcal{N}(0, I)$$. Gaussian variables have the very nice property that the sum of Gaussian variables is itself, another Gaussian random variable<sup>[1](#footnote1)</sup>. This almost makes intuitive sense considering the central limit theorem says that in the limit of a large number of i.i.d. random variables, their sum is normally distributed. 

The CLT holds for any bizarrely distributed random variables, so it's not too difficult to imagine that if the random variables are Gaussian, they just produce another Gaussian. Since each $$x_i$$ is just a linear combination of the elements of $$z$$, this means $$x$$ has a Gaussian joint pdf. So the form of the PDF holds more generally for cases in which the two variables are not independent.

More generally, the $$n$$-dimensional Gaussian distribution has pdf

$$f_X(x) = \frac{1}{(2\pi)^{n/2}\cdot \mathbf{det}(\Sigma)^{1/2}}\exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$

where $$\Sigma\in \mathbf{S}^{n}$$ is the covariance matrix. 

Rather than write the full form of the PDF each time, the more compact notation $$\mathcal{N}(\mu, \Sigma)$$ is used to denote a Gaussian Random Vector with mean $$\mu$$ and covariance $$\Sigma$$.

SHOW GIF HERE OF VARIOUS 2D GAUSSIAN PDFS

## Marginalisation
Given a Gaussian Random Vector $$X\in\mathbf{R}^{n}$$ with distribution $$\mathcal{N}(\mu, \Sigma)$$, it is often natural to ask what the distribution over just a subset of the components of the vector are, say the first $$k$$. This can be phrased as asking for the distribution of the following random vector

$$X_1 = 
\begin{bmatrix}
\mathbf{1}_k^T & 0_{n-k}^T
\end{bmatrix}
\begin{bmatrix}
X_1\\
X_2\\
\end{bmatrix} = AX
$$

Since this is just the linear transformation of a Gaussian Random Vector, $$X_1$$ is another Gaussian Random Vector with mean $$A\mu = \mu_1\in\mathbf{R}^k$$ and covariance matrix $$A\Sigma A^T = \Sigma_{11} \in \mathbf{R}^{k\times k}$$.

## Conditioning
## GP vs. Least Squares

## Conditional of a Gaussian
### Intuition
The formula for the conditional Gaussian can look impenetrable at first glance and impossible to remember. However, this section shows that the formula for the conditional mean and variance practically fall out from simple dimensional analysis and reasoning about the various matrix shapes.

To start, notice that the only parameters defining a zero mean Gaussian are $$\Sigma_{11}$$, $$\Sigma_{12}$$, and $$\Sigma_{22}$$. In particular, the conditional mean is only going to depend on these parameters and perhaps their inverses or transposes.

For the conditional mean $$\mathbf{E}[X_1|X_2]$$, 
we're given a vector $$x_2\in\mathbf{R}^{n_2}$$ and need to produce a vector $$\hat{x}_1\in \mathbf{R}^{n_1}$$. If you were to simply guess at a formula based only on the dimensions of the arrays, the most obvious candidate is
$$\Sigma_{12}x_2$$. Because $$\Sigma_{12} \in \mathbf{R}^{n_1 \times n_2}$$, this function correctly maps the $$n_2$$ dimensional vector of observations to a $$n_1$$ dimensional output. 

However, this formula cannot be correct solely on the basis of dimensional analysis. The conditional mean should have dimension $$units$$ yet since the elements of $$\Sigma_{12}$$ are measured in $$units^2$$ and $$x_2$$ is measured in $$units$$, the result of our hypothesis for the conditional mean, $$\Sigma_{12}x_2$$, will be measured in $$units^3$$. 

In order to get back to the correct units, we need something measured in $$units^{-2}$$. The simplest candidate based on this criteria is to insert a factor of $$\Sigma_{22}^{-1}$$. In particular,

$$\hat{x}_{1} = \Sigma_{12}\Sigma_{22}^{-1}x_{2}$$

 This formula clearly works on the basis of matrix shapes<sup>[1](#footnote1)</sup>., as did the previous one, but crucially the formula is measured in $$units^2 \cdot units^{-2} \cdot units = units$$. And indeed this is the formula for the conditional mean of a zero mean Gaussian.

In the event the mean is non-zero, simply demeaning the vector first allows application of the formula

$$
\hat{x}_1 - \mu_{1} = \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_{2})
$$

For the variance of the conditional Gaussian, if we weren't given any observations to condition on, the variance would just be that of the marginal distribution, namely $$\Sigma_{11}$$. Observing $$x_2$$ gives us some information (or at least doesn't take away any information) so intuitively this should decrease the variance<sup>[1](#footnote1)</sup>(ignore exactly what decrease means in the matrix sense and just pretend it's a scalar). In light of this, we hypothesise the conditional variance is of the form 

$$\Sigma_{11} - A$$ 

where $$A\in \mathbf{R}^{n_1 \times n_1}$$.

The simplest guess for $$A$$ that satisfies the shape criterion is $$\Sigma_{12}\Sigma_{21}$$. This is of course measured in $$units^4$$ while $$\Sigma_{11}$$ has $$units^2$$. We cannot subtract quartic units from quadratic ones, so this formula must be incorrect. 

To fix this formula, we need something with $$units^{-2}$$. We can further observe that the only term we haven't used is $$\Sigma_{22}$$. To make the units work, it has to appear as an inverse. To make the matrix dimensions conform, it must appear between the two matrices.  So our final hypothesis for the conditional variance is 

$$\Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$$ 

which is indeed the correct formula.


## Conclusion
Nonlinear models appear all the time in math and physics. Fitting parameters does not always require a reduction to a linear model (though this may be what you want) and we have seen two methods for handling the nonlinear case. In fact, we have seen the algorithms for nonlinear least squares use linear least squares as a subroutine to iteratively refine the solution. 

Lastly, it is worth noting that even though, we have written an implementation of an NLLS solver, in practice, you should always use something like `scipy.optimize`'s `least_squares` method. Thousands of man hours of work have gone into creating efficient solvers, the result of which is many clever optimisations and corner-case handling on top of the vanilla implementation. 

## Footnotes
<a name="footnote1">1</a>: The expectation obeys 

$$\mathbf{E}[aX + bY] = a\mathbf{E}[X] + b \mathbf{E}[Y].$$

This is true regardless of the relationship between $$X$$ and $$Y$$.

<a name="footnote2">2</a>: The inverse covariance matrix which often appears is called the _precision matrix_

<a name="footnote3">3</a>: Pay close attention to how the subscript indices come in pairs. In particular, this colorised version highlights the pattern $$\hat{y}_{\color{red}1} = \Sigma_{\color{red}1\color{orange}2}\Sigma_{\color{orange}2\color{yellow}2}^{-1}y_{\color{yellow}2}$$. This helps when needing to recall the formula from memory.


## References
1. [Zipf's Law Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law)
2. [Gauss-Newton Algorithm Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)
3. [Stanford's Intro to Linear Dynamical Systems](http://ee263.stanford.edu/)
4. [scipy.optimize Notes on Least Squares Implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares)
5. [Boyd & Vandenberghe's Intro to Applied Linear Algebra](http://vmls-book.stanford.edu/)
