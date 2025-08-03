---
title: "Chaterjee's Xi Coefficient"
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

## Pearson's Correlation Coefficient

Pearson's correlation is the one most people mean when they talk about "correlation" without any qualifiers.
It is defined as

$$ \rho = \mathbf{corr}(x, y) = {\mathbf{cov}(x, y) \over \mathbf{var}(x) \mathbf{var}(y)}.$$

Pearson's correlation measures the _linear_ relationship between two variables $$x, y\in \mathbf{R}^N$$.

The numerator is the _covariance_ of the variables $$x$$ and $$y$$.
It is defined as

$$ \mathbf{cov}(x, y) = {1 \over N} \sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})$$

The covariance measures the average product of $$x$$ and $$y$$'s deviations from their respective means.

The issue with covariance is that it can be large simply because the deviations of $$x$$ and $$y$$ from their means are large.
For example, we could increase the magnitude of the covariance simply by changing units from kilometers to meters.

The denominator is a normalisation to ensure $$-1 \le \rho \le 1$$.
This makes Pearson's correlation comparable across different data, or even the same data measured in different units.

The following Python code implements Pearson's correlation

{% highlight python %}
def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    x_bar = x.sum() / n
    y_bar = y.sum() / n

    x_deviation = x - x_bar
    y_deviation = y - y_bar

    cov_xy = (x_deviation * y_deviation).sum() / n
    var_x = (x_deviation * x_deviation).sum() / n
    var_y = (y_deviation * y_deviation).sum() / n

    rho = cov_xy / (var_x * var_y) ** 0.5

    return rho
{% endhighlight %}

Figure 1 shows 4 different datasets with increasing correlation.
As the correlation approaches 1, you can see the data begin to line on a line with positive slope.

<figure class>
    <a href="/assets/chatterjees-xi/images/pearson_corrs.png"><img src="/assets/chatterjees-xi/images/pearson_corrs.png"></a>
    <figcaption>Figure 1: Scatter plot of data with various Pearson correlations.</figcaption>
</figure>

It is important to emphasise that Pearson's correlation only measures the _linear_ relationship between two vectors.
Figure 2 shows points that have a simple functional relationship along with their correlations.
Though in each case, $$y$$ is very clearly a function of $$x$$, the correlation is not 1.

<figure class>
    <a href="/assets/chatterjees-xi/images/nonlinear_pearson_corrs.png"><img src="/assets/chatterjees-xi/images/nonlinear_pearson_corrs.png"></a>
    <figcaption>Figure 2: From left to right, a quadratic, sigmoid, and sine function. </figcaption>
</figure>

## Spearman's Correlation

Spearman's correlation attempts to address some of the problems with Pearson correlation.
Rather than measure the _linear_ relationship between $$x$$ and $$y$$, it tries to measure the _monotonicity_ of the variables.<sup>[1](#footnote1)</sup>

Spearman's correlation is defined as

$$\tau = \mathbf{corr}(\mathbf{rank}(x),\, \mathbf{rank}(y)).$$

Rather than work with the data values directly, Spearman's applies the Pearson correlation to the _ranks_ of the data.
If there are no tied ranks, this simplifies to

$$ \tau = 1 - \frac{6\sum_{i=1}^N \left(\mathbf{rank}(x_i) - \mathbf{rank}(y_i)\right)^2}{N(N^2-1)}$$

This makes it clear that when the ordering of $$x$$ and $$y$$ is the same (i.e. there's a monotonic relationship) that Spearman's correlation is 1.

Python code for computing Spearman's correlation with unique data is given below

{% highlight python %}

def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    assert len(x) == len(set(x))
    assert len(y) == len(set(y))

    n = x.size
    rank_x = np.empty_like(x, dtype=int)
    rank_x[np.argsort(x)] = np.arange(1, n + 1)

    rank_y = np.empty_like(y, dtype=int)
    rank_y[np.argsort(y)] = np.arange(1, n + 1)
    d = rank_x - rank_y
    d_sq = d * d

    corr = 1 - d_sq.sum() / (n * (n**2 - 1) / 6)

    return corr
{% endhighlight %}

Figure 3 shows how the Spearman correlation is exactly 1 in the case of the sigmoid function, but still fails to capture the fact that in both the quadratic and sine cases, $$y$$ is a noiseless function of $$x$$.

<figure class>
    <a href="/assets/chatterjees-xi/images/nonlinear_spearman_corrs.png"><img src="/assets/chatterjees-xi/images/nonlinear_spearman_corrs.png"></a>
    <figcaption>Figure 3: Spearman's correlation is low on both the quadratic and sine (as it was with Pearson's) but the sigmoid is exactly 1 because it is a monotone function. </figcaption>
</figure>

Based on the formula, we can see that when large $$x_i$$ implies large $$y_i$$ (as measured by their ranks in the dataset), Spearman's correlation will be close to 1.
In the quadratic case, this does **not** hold.
For example $$x=-0.9$$ and $$x=1$$ both produce similarly ranked $$y$$ values (i.e. 0.81 and 1) yet the ranks of the $$x$$ values are far apart.

Similarly for the sine wave, both $$x=-2\pi$$ and $$x=2\pi$$ yield the same $$y$$-value of 1.
Here, $$x=-2\pi$$ is assigned the lowest rank while its corresponding $$y$$-value has its highest rank.
This leads to a large squared difference and as a result, a low value of the correlation $$\tau$$.

Spearman's correlation is able to capture non-linear relationships, but how does it compare on the original data shown in Figure 1?
Figure 4 is sampled from the same distribution as Figure 1, but now includes the Spearman correlation as well

<figure class>
    <a href="/assets/chatterjees-xi/images/spearman_corrs.png"><img src="/assets/chatterjees-xi/images/spearman_corrs.png"></a>
    <figcaption>Figure 4: Scatter plot of data with various Pearson/Spearman correlations.</figcaption>
</figure>

We can see from the figure that even when the data is nearly linear (as in the bottom right), the Spearman correlation indicates there is almost no monotonic relationship between these variables.

What if we want to capture more than just monotonic relationships?
What if we want to measure how likely it is that $$y$$ is a noiseless function $$x$$?

Surprisingly, there is a very simple correlation coefficient that does exactly this.

## Chatterjee's Correlation

Chaterjee's Xi is defined by taking the data and first ordering them by their $$x$$-values (assuming they are unique).
The correlation is then given by

$$\xi = 1 - {3\sum_{i=1}^{N-1} |\mathbf{rank}(Y_{i+1}) - \mathbf{rank}(Y_i)| \over N^2-1},$$

where again, the $$y$$-values are assumed to be in increasing order of their corresponding $$x$$-values.

The following python implements this formula in the case of unique $$x$$-values.

{% highlight python %}
def chatterjee_corr(x: np.ndarray, y: np.ndarray) -> float:
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

The major difference between Spearman's and Chaterjee's correlation coefficient is that Spearman's relies on computing the differences between ranks of $$x$$ and ranks of $$y$$, whereas Chatterjee's relies on differences only between ranks of $$y$$.

## Footnotes

<a name="footnote1">1</a>: Linearity, of course, being a specific type of monotonicity

## References

<https://arxiv.org/abs/1909.10140>
<https://discuss.scientific-python.org/t/new-function-scipy-stats-xi-correlation/1498>
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chatterjeexi.html>
