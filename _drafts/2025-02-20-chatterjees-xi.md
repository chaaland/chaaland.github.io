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
In this post, we'll see a few different ways correlation is expressed mathematically and how they match or clash with our everyday use of the word.
We'll conclude with a recently developed measure of correlation published in 2019.

## Pearson's Correlation Coefficient

When most people talk about "correlation" in a numerical discipline, they're usually talking about Pearson's correlation coefficient.
Pearson's correlation measures the _linear_ relationship between two variables $$x, y\in \mathbf{R}^N$$ and is defined as

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

Figure 2 shows points following a sigmoid pattern without any noise.
Despite a simple functional relationship existing between $$x$$ and $$y$$, the correlation is only 0.9.

<figure class>
    <a href="/assets/chatterjees-xi/images/pearson_sigmoid.png"><img src="/assets/chatterjees-xi/images/pearson_sigmoid.png"></a>
    <figcaption>Figure 2: Points with a sigmoidal relationship. </figcaption>
</figure>

In addition to being unable to capture simple non-linear relationships, Pearson's correlation also suffers in the presence of outliers.

Figure 3 shows points with a linear relationship with the presence of a single outlier point.

<figure class>
    <a href="/assets/chatterjees-xi/images/pearson_outlier.png"><img src="/assets/chatterjees-xi/images/pearson_outlier.png"></a>
    <figcaption>Figure 3: A single outlier point reduces the correlation to 0.64 in a dataset that would otherwise have a correlation of 1. </figcaption>
</figure>

Is there a more robust notion of correlation?
Is there a notion of correlation that can capture more complicated relationships between variables besides the purely linear?

## Spearman's Correlation

Spearman's correlation addresses these limitations with Pearson's correlation.
Spearman's correlation is defined as

$$\tau = \mathbf{corr}(\mathbf{rank}(x),\, \mathbf{rank}(y)).$$

Spearman’s correlation applies Pearson’s formula to the _ranks_ of the data instead of the raw values, which makes it naturally robust to outliers.

When there are no ties, Spearman's correlation simplifies to

$$ \tau = 1 - \frac{6\sum_{i=1}^N \left(\mathbf{rank}(x_i) - \mathbf{rank}(y_i)\right)^2}{N(N^2-1)}$$

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

If we now turn to the sigmoid data from Figure 2, computing Spearman's gives $$\tau =1$$.
Perfect correlation!
This is because $$\mathbf{rank}(x_i) = \mathbf{rank}(y_i)$$.
In fact, this will be the case for any monotonic function so Spearman's can better capture non-linear relationships in data such as square roots, logarithms, or exponentials.

As already mentioned, Spearman's is also naturally robust to outliers in the data.
Figure 5 reproduces the same outlier dataset from Figure 3, this time showing Spearman's correlation.

<figure class>
    <a href="/assets/chatterjees-xi/images/spearman_outlier.png"><img src="/assets/chatterjees-xi/images/spearman_outlier.png"></a>
    <figcaption>Figure 5: Spearman's correlation is 0.93 even in the presence of a large outlier.</figcaption>
</figure>

With $$\tau=0.93$$, it is clear Spearman's correlation is significantly less impacted by the outlier than Pearson's ($$\rho = 0.25$$) and is still able to detect the strong relationship between the variables.
This is because the correlation calculation doesn't depend on the actual values themselves, only the ranks.

Figure 6 shows Spearman's correlation on noiseless quadratic and sinusoidal data.

<figure class>
    <a href="/assets/chatterjees-xi/images/nonlinear_spearman_corrs.png"><img src="/assets/chatterjees-xi/images/nonlinear_spearman_corrs.png"></a>
    <figcaption>Figure 6: Spearman's correlation is low on both the quadratic and sine. </figcaption>
</figure>

Even though there is a simple relationship between $$x$$ and $$y$$, Spearman's correlation is low.
From the definition, we see that in order for Spearman's to be close to 1, each $$x_i$$ needs to be paired comparably ranked $$y_i$$. That is, small $$x_i$$ paired with small $$y_i$$ and large $$x_i$$ paired with large $$y_i$$.

In the quadratic case, this does **not** hold.
For example $$x=-0.9$$ and $$x=1$$ both produce similarly ranked $$y$$ values (i.e. 0.81 and 1) yet the ranks of the $$x$$ values are far apart.

Similarly, for the sine wave, both $$x=-2\pi$$ and $$x=2\pi$$ yield the same $$y$$-value of 1.
Here, $$x=-2\pi$$ is assigned the lowest rank while its corresponding $$y$$-value has its highest rank.

What if we want to capture more than just monotonic relationships?
What if we want to measure how likely it is that $$y$$ is a noiseless function of $$x$$?

Surprisingly, there is a very simple correlation coefficient that does exactly this.

## Chatterjee's Correlation

Chatterjee's Xi coefficient is defined by taking the data and first ordering them by their $$x$$-values (assuming they are unique) then computing the value

$$\xi = 1 - {3\sum_{i=1}^{N-1} |\mathbf{rank}(y_{i+1}) - \mathbf{rank}(y_i)| \over N^2-1}.$$

Let's apply this formula to the dataset below to understand how the coefficient is computed.

| x        | y          |
| -------- | ---------  |
| 3.14     | 0          |
| 2.36     | 0.70       |  
| 0.79     | 0.71       |
| 3.93     | -0.71      |
| 1.57     | 1.0        |

We first order the data by the $$x$$-values

| x        | y          |
| -------- | ---------  |
| 0.79     | 0.71       |
| 1.57     | 1.0        |
| 2.36     | 0.70       |  
| 3.14     | 0          |
| 3.93     | -0.71      |

Then we compute the ranks of the $$y$$-values

| x        | y          | rank(y) |
| -------- | ---------  | ------- |
| 0.79     | 0.71       | 4       |
| 1.57     | 1.0        | 5       |
| 2.36     | 0.70       | 3       |
| 3.14     | 0          | 2       |
| 3.93     | -0.71      | 1       |

We then compute the sum of the absolute differences in the ranks

$$d = |5 - 4| + |3 - 5| + |2 - 3| + |1 - 2| = 5.$$

Plugging this into the formula, we get

$$\xi = 1 - {3 \cdot 5 \over 5^2 -1} = 0.375$$

The following Python code implements this formula in the case of unique $$x$$-values.

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
However, rather than computing the differences in the ranks of $$x$$ and $$y$$, Chatterjee's correlation computes the differences between the ranks of only the $$y$$-values.

By first ordering the data

## Footnotes

<a name="footnote1">1</a>: Linearity, of course, being a specific type of monotonicity

## References

1. [Chatterjee's original paper introducing Xi](https://arxiv.org/abs/1909.10140)
2. [Scipy discussion on including Chatterjee's Xi in Scipy 1.15.0](https://discuss.scientific-python.org/t/new-function-scipy-stats-xi-correlation/1498)
3. [Scipy documentation for chatterjeexi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chatterjeexi.html)
