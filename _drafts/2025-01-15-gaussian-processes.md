---
title: "Gaussian Process Intuition"
categories:
  - Mathematics
date:   2025-01-15 10:57:00 +0100
mathjax: true
tags:
  - Gaussian Processes
  - Regression
toc: true
# classes: wide
# excerpt: ""
header: 
  # make some GP with confidence interval shaded
  overlay_image: assets/gaussian-processes/images/splash_image.png
  overlay_filter: 0.2
---

This post is a light introduction to Gaussian Processes (GPs).
The goal is to give an explanation of some of key results and most importantly, some intuition for how they come about.

## GP Intuition

One of the simplest motivations for Gaussian Processes is interpolation.
We're given some features $$X\in \mathbf{R}^{n\times d}$$ and the corresponding targets $$y\in \mathbf{R}^{n}$$.
Can we find a curve such that $$f(x^{(i)}) = y^{(i)}$$ for $$i=1,\ldots,n$$?

Gaussian processes address this problem by forming a linear function of the _target_ values

$$f(x) = \sum_{i=1}^n w_i(x) y^{(i)}.$$

It's worthwhile comparing this to the classic linear regression model

$$f(x) = \sum_{i=1}^d w_i x_i.$$

Though they look very similar, there are a couple of key distinctions worth noting

- linear regression is a weighted function of the _features_ while GPs are weighted functions of the _targets_
- linear regression has constant weights, independent of the feature vector while the GP weights are a function of the input vector

The weight function in GPs is chosen so that points "nearer" to $$x$$ have their targets weighted higher than points futher away.
The underlying assumption is that the forecast for $$x$$ should be similar to the target values of "similar" points.

<figure class>
    <a href="/assets/gaussian-processes/gifs/1d-gpr.gif"><img src="/assets/gaussian-processes/gifs/1d-gpr.gif"></a>
</figure>

## Covariance Functions

But what does it mean to say that two feature vectors are "similar" or "near"?
In GPs this is codified mathematically by a _covariance function_

$$k: \mathbf{R}^{d} \times \mathbf{R}^{d} \rightarrow \mathbf{R}.$$

Some example covariance functions include the exponentiated quadratic

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

The criteria for a function to be a valid covariance function is that the matrix with elements $$\Sigma_{ij} = k(x^{(i)}, x^{(j)})$$ form a valid covariance matrix (i.e. positive semidefinite)<sup>[1](#footnote1)</sup>.

## GP weight function

With our notion of "nearness" between two feature vector defined concretely in terms of a covariance function, how does this relate to our weight function $$w_i(x)$$ that we need to make predictions?

Suppose we have $$n_1$$ training points (i.e. points for which we have a target label) and we want to make a prediction for a single new point.
First, let's define a covariance matrix $$\Sigma\in \mathbf{R}^{(n_1+1)\times(n_1+1)}$$ in terms of

- $$\Sigma_{11}\in \mathbf{R}^{n_1\times n_1}$$, the matrix of similarities between the training points
- $$\Sigma_{12}\in \mathbf{R}^{n_1 \times 1}$$, the matrix of similarities between the training points and the new point we want to predict
- $$\Sigma_{22}\in \mathbf{R}$$, the similarity between the new point and itself. For the kernels we list above, this will be 1.

$$
\Sigma =
\begin{bmatrix}
\Sigma_{11} & \Sigma_{12}\\
\Sigma_{21} & \Sigma_{22}\\
\end{bmatrix}
$$

We want to express the prediction, $$y_2$$, as a linear function of the target $$y_1$$

$$ y_2 = \sum_{i=1}^{n_1} w_i(x) y^{(i)}_1$$

Or in matrix notation,

$$y_2 = W(x) y_1.$$

For this to make sense

1. $$W$$ must be a function of the features $$x$$ of $$y_2$$
2. the dimensions of $$A$$ must be conformable. That is, $$A \in \mathbf{R}^{n_1 \times 1}$$.
3. $$A$$ must have no units. This is a consequence of $$y_1$$ and $$y_2$$ having the same units
4. $$A$$ has to be a function of the covariance matrix $$\Sigma$$, namely $$A = g(\Sigma)$$.

The most obvious guess for the linear function is

$$y_{\color{red}2} = \Sigma_{\color{red}2\color{orange}1}y_{\color{orange}1}.$$

We can see that the matrices conform and the prediction is a function of the covariance matrix.
However, since $$\Sigma_{12}$$ represents a variance, it will have $$units^2$$ leading to a prediction with $$units^3$$.

To get rid of the extra $$units^2$$, we can multiply by an inverse covariance matrix.
But which one?

$$\Sigma_{12}$$ isn't even guaranteed to be square, so an inverse isn't even defined.

$$\Sigma_{22}$$ is square so it at least has an inverse.
But imagine we were predicting multiple points and $$\Sigma_{22}\in\mathbf{n_2\times n_2}$$.
We do not want each prediction to depend on any of the other predictions we happen to be making at the time.
Otherwise, our prediction would vary depending on which of the other $$n_2 -1$$ points we were making predictions for.

That leaves just $$\Sigma_{11}$$ which is square and whose inverse has units of $$units^{-2}$$.
The constraints of matrix multiplication leave only one choice for the prediction

$$y_{\color{red}2} = \Sigma_{\color{red}2\color{orange}1}\Sigma_{\color{orange}1\color{orange}1}^{-1}y_{\color{orange}1}.$$

For the variance of the conditional Gaussian, if we weren't given any observations to condition on, the variance would just be that of the marginal distribution, namely $$\Sigma_{11}$$. Observing $$x_2$$ gives us some information (or at least doesn't take away any information) so intuitively this should decrease the variance<sup>[1](#footnote1)</sup>(ignore exactly what decrease means in the matrix sense and just pretend it's a scalar). In light of this, we hypothesise the conditional variance is of the form

$$\Sigma_{11} - A$$

where $$A\in \mathbf{R}^{n_1 \times n_1}$$.

The simplest guess for $$A$$ that satisfies the shape criterion is $$\Sigma_{12}\Sigma_{21}$$. This is of course measured in $$units^4$$ while $$\Sigma_{11}$$ has $$units^2$$. We cannot subtract quartic units from quadratic ones, so this formula must be incorrect.

To fix this formula, we need something with $$units^{-2}$$. We can further observe that the only term we haven't used is $$\Sigma_{22}$$. To make the units work, it has to appear as an inverse. To make the matrix dimensions conform, it must appear between the two matrices.  So our final hypothesis for the conditional variance is

$$\Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$$

## Conclusion

## Footnotes

<a name="footnote1">1</a>: Sometimes you'll see the notation $$\Sigma \in \mathbf{S}^n_+$$ which means a positive semidefinite $$n\times n$$ matrix or $$\Sigma \in \mathbf{S}^n_{++}$$ which means a positive definite $$n\times n$$ matrix.

<a name="footnote1">1</a>: The expectation obeys

$$\mathbf{E}[aX + bY] = a\mathbf{E}[X] + b \mathbf{E}[Y].$$

This is true regardless of the relationship between $$X$$ and $$Y$$.

<a name="footnote2">2</a>: The inverse covariance matrix which often appears is called the _precision matrix_

<a name="footnote3">3</a>: Pay close attention to how the subscript indices come in pairs. In particular, this colorised version highlights the pattern $$\hat{y}_{\color{red}1} = \Sigma_{\color{red}1\color{orange}2}\Sigma_{\color{orange}2\color{yellow}2}^{-1}y_{\color{yellow}2}$$. This helps when needing to recall the formula from memory.

## References

1. [Zipf's Law Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law)
