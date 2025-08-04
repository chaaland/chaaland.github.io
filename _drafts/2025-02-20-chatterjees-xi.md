---
title: "Chatterjee's Xi Coefficient"
categories:
  - Statistics
date:   2025-01-11 14:33:00 +0100
mathjax: true
tags:
  - Statistics
toc: true
# classes: wide
excerpt: ""
header: 
  overlay_image: assets/chatterjees-xi/images/splash_image.png
  overlay_filter: 0.2
---

## Motivation

In everyday parlance, we often use the words correlation, association, and relationship somewhat interchangeably.
When we say "lack of exercise is correlated with bad health outcomes", we mean something like, "there is a relationship between not exercising and having poor health".

In mathematics, the notion of correlation is more precise.
In this post, we'll see a few different ways correlation is expressed mathematically and how they match or clash with our intuitions.
We'll conclude with a recently developed measure of correlation published in 2019.

## Pearson's Correlation Coefficient

When most people talk about "correlation", they're usually thinking of Pearson's correlation coefficient.
Pearson's correlation measures the _linear_ relationship between two variables $$x, y\in \mathbf{R}^N$$.
It is defined as

$$ \rho = \mathbf{corr}(x, y) = {\mathbf{cov}(x, y) \over \sqrt{\mathbf{var}(x) \mathbf{var}(y)}}.$$

The numerator is the _covariance_ of the variables $$x$$ and $$y$$ defined as

$$ \mathbf{cov}(x, y) = {1 \over N} \sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})$$

The covariance measures the average product of $$x$$ and $$y$$'s deviations from their respective means.

The issue with covariance is that it can be large simply because the deviations of $$x$$ and $$y$$ from their means are large.
For example, we could increase the magnitude of the covariance simply by changing the units of measurement from kilometers to meters (i.e. scaling the variables by 1000).

The denominator is a normalisation to ensure $$-1 \le \rho \le 1$$.
This makes Pearson's correlation a unit-less quantity comparable across different data, or even the same data measured in different units.

The following Python code implements Pearson's correlation

{% highlight python %}
def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    x_bar = x.sum() / n
    y_bar = y.sum() / n

    x_deviation = x - x_bar
    y_deviation = y - y_bar

    # ignore Bessel's correction
    cov_xy = (x_deviation * y_deviation).sum() / n
    var_x = (x_deviation * x_deviation).sum() / n
    var_y = (y_deviation * y_deviation).sum() / n

    rho = cov_xy / (var_x * var_y) ** 0.5

    return rho
{% endhighlight %}

Figure 1 shows 4 different datasets with increasing correlation.
As the correlation approaches 1, you can see the data begin to lie on a line with positive slope.

<figure class>
    <a href="/assets/chatterjees-xi/images/pearson_corrs.png"><img src="/assets/chatterjees-xi/images/pearson_corrs.png"></a>
    <figcaption>Figure 1: Scatter plot of data with various Pearson correlations.</figcaption>
</figure>

It is important to emphasise that Pearson's correlation only measures the _linear_ relationship between two vectors.
Figure 2 shows points following a simple sigmoid pattern.
Despite there being a simple functional relationship between $$x$$ and $$y$$, the correlation is only 0.9.

<figure class>
    <a href="/assets/chatterjees-xi/images/pearson_sigmoid.png"><img src="/assets/chatterjees-xi/images/pearson_sigmoid.png"></a>
    <figcaption>Figure 2: Points with a sigmoidal relationship. </figcaption>
</figure>

In addition to being unable to capture simple non-linear relationships, Pearson's correlation also suffers in the presence outliers.
Figure 3 shows points with a simple linear relationship with the presence of a single outlier point

<figure class>
    <a href="/assets/chatterjees-xi/images/pearson_outlier.png"><img src="/assets/chatterjees-xi/images/pearson_outlier.png"></a>
    <figcaption>Figure 3: A single outlier point reduces the correlation to 0.64 in a dataset that would otherwise have a correlation of 1. </figcaption>
</figure>

Is there a more robust notion of correlation?
Is there a notion of correlation that can capture more complicated relationships between variables besides the purely linear?

## Spearman's Correlation

Spearman's correlation addresses these issues with Pearon's correlation.
Rather than measure the _linear_ relationship between $$x$$ and $$y$$, it tries to measure the _monotonicity_ of the variables.<sup>[1](#footnote1)</sup>

Spearman's correlation is defined as

$$\tau = \mathbf{corr}(\mathbf{rank}(x),\, \mathbf{rank}(y)).$$

Rather than work with the data values directly, Spearman's applies Pearson's formula to the _ranks_ of the data which has the added benefit of being robust to outliers.

When there are no ties, Spearman's correlation simplifies to

$$ \tau = 1 - \frac{6\sum_{i=1}^N \left(\mathbf{rank}(x_i) - \mathbf{rank}(y_i)\right)^2}{N(N^2-1)}$$

This makes it clear that when the ordering of $$x$$ and $$y$$ is the same (i.e. there's a monotonic relationship) that Spearman's correlation is 1.

Python code for computing Spearman's correlation with unique data is given below

{% highlight python %}

def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    # assert there are no ties
    assert len(x) == len(set(x))
    assert len(y) == len(set(y))

    n = x.size
    rank_x = np.empty_like(x, dtype=int)
    rank_x[np.argsort(x)] = np.arange(1, n + 1)

    rank_y = np.empty_like(y, dtype=int)
    rank_y[np.argsort(y)] = np.arange(1, n + 1)
    rank_diff = rank_x - rank_y
    sq_rank_diff = rank_diff * rank_diff

    corr = 1 - sq_rank_diff.sum() / (n * (n**2 - 1) / 6)

    return corr
{% endhighlight %}

Figure 4 shows how Pearson and Spearman compare on data from the same distribution as Figure 1.

<figure class>
    <a href="/assets/chatterjees-xi/images/spearman_corrs.png"><img src="/assets/chatterjees-xi/images/spearman_corrs.png"></a>
    <figcaption>Figure 4: Though Pearson and Spearman correlations yield different numbers, they are similar.</figcaption>
</figure>

Computing Spearman's on the sigmoid data gives $$\tau =1$$.
Perfect correlation!
This is because $$\mathbf{rank}(x_i) = \mathbf{rank}(y_i)$$.
In fact, this will be the case for any monotonic function

Figure 5, reproduces the same outlier dataset from Figure 3 and this time the
<figure class>
    <a href="/assets/chatterjees-xi/images/spearman_outlier.png"><img src="/assets/chatterjees-xi/images/spearman_outlier.png"></a>
    <figcaption>Figure 6: Though Pearson and Spearman correlations yield different numbers, they are similar.</figcaption>
</figure>

<figure class>
    <a href="/assets/chatterjees-xi/images/nonlinear_spearman_corrs.png"><img src="/assets/chatterjees-xi/images/nonlinear_spearman_corrs.png"></a>
    <figcaption>Figure 3: Spearman's correlation is low on both the quadratic and sine (as it was with Pearson's) but the sigmoid is exactly 1 because it is a monotone function. </figcaption>
</figure>

Like Pearson's correlation, Spearman's is basically 0 for the quadratic.
On the sigmoid, however, we get identically 1 even though the points do not have the linear relationship required by Pearson to achieve a correlation of 1.
 how the Spearman correlation is exactly 1 in the case of the sigmoid function, but still fails to capture the fact that in both the quadratic and sine cases, $$y$$ is a noiseless function of $$x$$.

Based on the formula, we can see that when large $$x_i$$ implies large $$y_i$$ (as measured by their ranks in the dataset), Spearman's correlation will be close to 1.
In the quadratic case, this does **not** hold.
For example $$x=-0.9$$ and $$x=1$$ both produce similarly ranked $$y$$ values (i.e. 0.81 and 1) yet the ranks of the $$x$$ values are far apart.

Similarly, for the sine wave, both $$x=-2\pi$$ and $$x=2\pi$$ yield the same $$y$$-value of 1.
Here, $$x=-2\pi$$ is assigned the lowest rank while its corresponding $$y$$-value has its highest rank.
This leads to a large squared difference and as a result, a low value of the correlation $$\tau$$.

Spearman's correlation is able to capture non-linear relationships, but how does it compare on the original data shown in Figure 1?
Figure 4 is sampled from the same distribution as Figure 1, but now includes the Spearman correlation as well

We can see from the figure that even when the data is nearly linear (as in the bottom right), the Spearman correlation indicates there is almost no monotonic relationship between these variables.

What if we want to capture more than just monotonic relationships?
What if we want to measure how likely it is that $$y$$ is a noiseless function of $$x$$?

Surprisingly, there is a very simple correlation coefficient that does exactly this.

## Chatterjee's Correlation

Chatterjee's Xi coefficient is defined by taking the data and first ordering them by their $$x$$-values (assuming they are unique).
The correlation is then given by

$$\xi = 1 - {3\sum_{i=1}^{N-1} |\mathbf{rank}(y_{i+1}) - \mathbf{rank}(y_i)| \over N^2-1},$$

where again, the $$y$$-values are assumed to be in increasing order of their corresponding $$x$$-values.

The following Python implements this formula in the case of unique $$x$$-values.

{% highlight python %}
def chatterjee_corr(x: np.ndarray, y: np.ndarray) -> float:
    # assert there are no ties

    assert len(x) == len(set(x))
    assert len(y) == len(set(y))

    n = x.size
    y_ordered_by_x =  y[np.argsort(x)]

    ranks = np.empty_like(x, dtype=int)
    ranks[np.argsort(y_ordered_by_x)] = np.arange(1, n + 1)
    abs_rank_diffs = np.abs(np.diff(ranks))

    xi_corr = 1 - abs_rank_diffs.sum() / ((n**2 - 1) / 3)

    return xi_corr

{% endhighlight %}

Similar to Spearman's, the values used to compute Chatterjee's correlation are rank based.

The major difference between Spearman's and Chatterjee's correlation coefficient is that  Spearman's relies on differences between the ranks of $$x$$ and $$y$$, while Chatterjee's relies only on differences between the ranks of $$y$$.

## Footnotes

<a name="footnote1">1</a>: Linearity, of course, being a specific type of monotonicity

## References

1. [Chatterjee's original paper introducing Xi](https://arxiv.org/abs/1909.10140)
2. [Scipy discussion on including Chatterjee's Xi in Scipy 1.15.0](https://discuss.scientific-python.org/t/new-function-scipy-stats-xi-correlation/1498)
3. [Scipy documentation for chatterjeexi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chatterjeexi.html)
