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

This post is a light introduction to Gaussian Processes (GPs) by focusing on the intuition rather than complicated mathematics and matrix algebra.
Though some of the formulas seem impenetrable, we'll see that they can be arrived at most though simple dimenionsal analysis.

## GP Intuition

Imagine you're measuring temperature at different locations.
If you know it's 20°C outside of your house but 17°C  at your friend's house 20 miles away, what might you expect the temperature be at your local grocery store 1 mile away?
How about your friend's local grocery store 1 mile from their house?
Intuitively, you'd expect the temperature to be more similar to 20°C than to 17°C closer to your house and closer to 17°C nearer your friend's house - but how can we formalise this reasoning mathematically?

Given some features $$X\in \mathbf{R}^{n\times d}$$ and their corresponding targets $$y\in \mathbf{R}^{n}$$, Gaussian processes address this problem by forming a linear function of the _target_ values

$$f(x) = \sum_{i=1}^n w_i(x) y^{(i)}.$$

Compare this with the classic linear regression model

$$f(x) = \sum_{i=1}^d w_i x_i.$$

Though they look very similar, there are a couple of key distinctions

- linear regression is a weighted function of the _features_ while GPs are weighted functions of the _targets_
- linear regression has constant weights, independent of the feature vector while the GP weights are a function of the input vector

The weight function in GPs is chosen so that points "closer" to $$x$$ have their targets weighted higher than points farther away.
The underlying assumption is that the prediction for $$x$$ should be similar to the target values of  points close by.

<figure class>
    <a href="/assets/gaussian-processes/gifs/1d-gpr.gif"><img src="/assets/gaussian-processes/gifs/1d-gpr.gif"></a>
</figure>

## Covariance Functions

But what does it mean to say that two feature vectors are "close"?
In GPs this is codified mathematically by a _covariance function_

$$k: \mathbf{R}^{d} \times \mathbf{R}^{d} \rightarrow \mathbf{R}.$$

There are a variety of covariance functions that are used to encode this similarity.
Some example covariance functions include the exponentiated quadratic

$$k(x^{(1)}, x^{(2)}) = \exp\left(-\frac{||x^{(1)} - x^{(2)}||^2}{2\ell^2}\right)$$

the rational quadratic

$$k(x^{(1)}, x^{(2)}) = \left(1 + \frac{||x^{(1)} - x^{(2)}||^2}{2\alpha\ell^2}\right)^{-\alpha}$$

and the Matern covariance (of order 0)

$$k(x^{(1)}, x^{(2)}) = \sigma^2 \exp\left(-\frac{||x^{(1)} - x^{(2)}||}{\ell}\right)$$

The criteria for a function to be a valid covariance function is that the matrix with elements $$\Sigma_{ij} = k(x^{(i)}, x^{(j)})$$ form a valid covariance matrix (i.e. positive semidefinite)<sup>[1](#footnote1)</sup>.

### Length scale

The quantity $$\ell$$ in the above covariance functions is a hyperparameter called the _length scale_. Very roughly, it determines how close points need to be in order to be considered "near". In some situations it makes sense to have one length scale per feature dimension in which case the similarity is a function of the Mahalonobis distance

$$\left(x^{(1)}-x^{(2)}\right)^T\mathbf{diag}(\ell^{-2}_1, \ell^{-2}_2,\cdots, \ell^{-2}_n)\left(x^{(1)}-x^{(2)}\right)$$

which of course reduces to
$$\frac{1}{\ell^2}||x^{(1)} - x^{(2)}||^2$$
in the special case $$\ell_1 = \ell_2 =\cdots=\ell_n$$.

Now that we've mathematically enshrined our measure of similarity between points, let's see how this leads to our prediction formula.

## Weight function

At the outset we stated our prediction function should take the form

$$f(x) = \sum_{i=1}^n w_i(x) y^{(i)}.$$

Our weight function $$w_i : \mathbf{R}^d\rightarrow \mathbf{R}$$ should have the following properties:

1. It needs to give us predictions in the same units as our original measurements
2. It should incorporate the notion of similarity between points
3. The prediction for a point should only depend on the training data, not on other points we might want to predict

These three requirements lead us almost inevitably to a formula for $$w_i$$.

Suppose we have $$n_1$$ training points (i.e. points for which we have a target label) and we want to make a prediction for $$n_2$$ new points.
First, we can define a covariance matrix $$\Sigma\in \mathbf{R}^{(n_1+n_2)\times(n_1+n_2)}$$ in terms of

- $$\Sigma_{11}\in \mathbf{R}^{n_1\times n_1}$$, the matrix of similarities between the training points
- $$\Sigma_{12}\in \mathbf{R}^{n_1 \times n_2}$$, the matrix of similarities between the training points and the new point we want to predict
- $$\Sigma_{22}\in \mathbf{R}$$, the similarity between the new point and itself. For the kernels we list above, this will be 1.

$$
\Sigma =
\begin{bmatrix}
\Sigma_{11} & \Sigma_{12}\\
\Sigma_{21} & \Sigma_{22}\\
\end{bmatrix}
$$

We want to express the prediction, $$y_2$$, as a linear function of the target $$y_1$$:

$$ y_2 = \sum_{i=1}^{n_1} w_i(x) y^{(i)}_1$$

Or in matrix notation,

$$y_2 = W(x) y_1.$$

The most obvious guess for the linear function is

$$y_{\color{red}2} = \Sigma_{\color{red}2\color{orange}1}y_{\color{orange}1}.$$

We can see that the matrices conform and the prediction is a function of the covariance matrix.
However, since $$\Sigma_{12}$$ represents a variance, it will have $$units^2$$ leading to a prediction with $$units^3$$.

To get rid of the extra $$units^2$$, we can multiply by an inverse covariance matrix.
But which one?

$$\Sigma_{12}$$ isn't even guaranteed to be square, so an inverse isn't even defined.

$$\Sigma_{22}$$ is square so it at least has an inverse.
But this would violate our third criteria of the weight function, making the predictions dependent on the other points being predicted.

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

## References

1. [Zipf's Law Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law)
