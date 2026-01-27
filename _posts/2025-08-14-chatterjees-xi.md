---
title: "Chatterjee's Xi Coefficient"
categories:
  - Statistics
date:   2025-08-14 14:33:00 +0100
mathjax: true
tags:
  - Statistics
toc: true
# classes: wide
excerpt: "Detect nonlinear relationships that Pearson and Spearman miss."
header: 
  overlay_image: assets/2025/chatterjees-xi/images/splash_image.png
  overlay_filter: 0.5
---

## Motivation

In everyday parlance, we often use the words correlation, association, and relationship somewhat interchangeably.
When we say "height is correlated with age", we mean something like, "there is a relationship between your age and your height".

In mathematics, the notion of correlation is more precise.
In this post, we'll see a few different ways correlation is expressed mathematically and how they match or clash with our everyday use of the word.
We'll conclude with a surprisingly recently developed measure of correlation published in 2019.

## Pearson's Correlation Coefficient

When most people talk about "correlation" in a numerical discipline, they're usually talking about Pearson's correlation coefficient.
Pearson's correlation measures the _linear_ relationship between two sets of observations $$x=(x_1,\ldots,x_N)$$ and $$y=(y_1,\ldots,y_N)$$ and is defined as

$$ \rho = \mathbf{corr}(x, y) = {\mathbf{cov}(x, y) \over \sqrt{\mathbf{var}(x) \mathbf{var}(y)}}.$$

The numerator is the _covariance_ of the variables $$x$$ and $$y$$ defined as

$$ \mathbf{cov}(x, y) = {1 \over N} \sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})$$

The covariance measures the average product of $$x$$ and $$y$$'s deviations from their respective means.

The issue with covariance is that it can be large simply because the deviations of $$x$$ and $$y$$ from their means are large.
For example, we could increase the magnitude of the covariance simply by changing the units of measurement from kilometers to meters (i.e. scaling the variables by 1,000).
To avoid this, we normalise the covariance by the product of the standard deviations, ensuring $$-1 \le \rho \le 1$$.

The following Python code implements Pearson's correlation

<details>
<summary>
Click for code
</summary>
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
</details>
<br>

Figure 1 shows data with varying correlations.
As the correlation approaches 1, you can see the data begin to lie on a line with positive slope.
Use the sliders below to explore how Pearson's correlation changes with different correlation values and sample sizes.

<div class="pearson-widget" id="pearson-widget">
  <div class="pearson-controls">
    <label>
      Correlation (ρ)
      <input type="range" min="-1" max="1" value="0.5" step="0.25" data-param="corr">
      <span class="pearson-readout" data-readout="corr">0.50</span>
    </label>
    <label>
      Points (N)
      <input type="range" min="20" max="100" value="60" step="20" data-param="npoints">
      <span class="pearson-readout" data-readout="npoints">60</span>
    </label>
    <button type="button" class="pearson-button" id="pearson-reset">Reset</button>
  </div>
  <svg class="pearson-plot" id="pearson-svg" viewBox="0 0 500 400" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="500" height="400" fill="#0d1117"></rect>
    <g id="pearson-grid"></g>
    <g id="pearson-axes"></g>
    <g id="pearson-points"></g>
  </svg>
  <div class="pearson-info" id="pearson-info">
    ρ = 0.50 | N = 60 | Sample correlation: 0.50
  </div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 1: Scatter plot of data with adjustable Pearson correlation. Adjust the sliders to see how different correlation values and sample sizes affect the scatter pattern.</figcaption>
</div>

It is important to emphasise that Pearson's correlation only measures the _linear_ relationship between two vectors.

Figure 2 shows points following a sigmoid pattern without any noise.
Despite a simple functional relationship existing between $$x$$ and $$y$$, the correlation is only 0.9.

<figure class>
    <a href="/assets/2025/chatterjees-xi/images/pearson_sigmoid.png"><img src="/assets/2025/chatterjees-xi/images/pearson_sigmoid.png"></a>
    <figcaption>Figure 2: Points with a sigmoidal relationship. </figcaption>
</figure>

In addition to being unable to capture simple non-linear relationships, Pearson's correlation also suffers in the presence of outliers.

In Figure 3 we see points with a linear relationship containing a single outlier point.

<figure class>
    <a href="/assets/2025/chatterjees-xi/images/pearson_outlier.png"><img src="/assets/2025/chatterjees-xi/images/pearson_outlier.png"></a>
    <figcaption>Figure 3: A single outlier point reduces the correlation to 0.25 in a dataset that would otherwise have a correlation of 1. </figcaption>
</figure>

Is there a more robust notion of correlation?
Is there a notion of correlation that can capture more complicated relationships between variables besides the purely linear?

## Spearman's Correlation


Spearman's correlation addresses the limitations with Pearson's correlation.
Spearman's correlation is defined as

$$r_s = \mathbf{corr}(\mathbf{rank}(x),\, \mathbf{rank}(y)).$$

When there are no ties<sup>[1](#footnote1)</sup>, Spearman's correlation simplifies to

$$ r_s = 1 - \frac{6\sum_{i=1}^N \left(\mathbf{rank}(x_i) - \mathbf{rank}(y_i)\right)^2}{N(N^2-1)}$$

Here’s a simple implementation of Spearman’s correlation for data without ties:

<details>
<summary>
Click for code
</summary>

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

</details>
<br>

Figure 4 illustrates how Pearson's and Spearman's compare on data from the same distribution as Figure 1.
Use the sliders to see how both correlation measures respond to different data configurations.

<div class="spearman-widget" id="spearman-widget">
  <div class="spearman-controls">
    <label>
      Correlation (ρ)
      <input type="range" min="-1" max="1" value="0.5" step="0.25" data-param="spearman-corr">
      <span class="spearman-readout" data-readout="spearman-corr">0.50</span>
    </label>
    <label>
      Points (N)
      <input type="range" min="20" max="100" value="60" step="20" data-param="spearman-npoints">
      <span class="spearman-readout" data-readout="spearman-npoints">60</span>
    </label>
    <button type="button" class="spearman-button" id="spearman-reset">Reset</button>
  </div>
  <svg class="spearman-plot" id="spearman-svg" viewBox="0 0 500 400" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="500" height="400" fill="#0d1117"></rect>
    <g id="spearman-grid"></g>
    <g id="spearman-axes"></g>
    <g id="spearman-points"></g>
  </svg>
  <div class="spearman-info" id="spearman-info">
    ρ = 0.50 | N = 60 | Pearson: 0.50 | Spearman: 0.50
  </div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 4: Scatter plot comparing Pearson and Spearman correlations. Though the two yield different numbers, they are typically similar for linear data.</figcaption>
</div>

If we now turn to the sigmoid data from Figure 2, computing Spearman's gives $$r_s =1$$.
Perfect correlation!
This is because $$\mathbf{rank}(x_i) = \mathbf{rank}(y_i)$$.
In fact, this will be the case for any monotonic function.
This means Spearman's can capture non-linear relationships in data such as square roots, logarithms, and exponentials.

Spearman's is also naturally robust to outliers in the data because it uses rank information rather than the values themselves.
Figure 5 reproduces the same outlier dataset from Figure 3, this time showing Spearman's correlation.

<figure class>
    <a href="/assets/2025/chatterjees-xi/images/spearman_outlier.png"><img src="/assets/2025/chatterjees-xi/images/spearman_outlier.png"></a>
    <figcaption>Figure 5: Spearman's correlation is 0.93 even in the presence of a large outlier.</figcaption>
</figure>

With $$r_s=0.93$$, it is clear Spearman's correlation is still able to detect the strong relationship between the variables and is significantly less impacted by the outlier than Pearson's ($$\rho = 0.25$$).
This is because the correlation calculation doesn't depend on the actual values themselves, only the ranks.

However, Spearman's correlation cannot capture more complicated relationships between data.
For example, the noiseless quadratic and sinusoidal data shown in figure 6 below.

<figure class>
    <a href="/assets/2025/chatterjees-xi/images/nonlinear_spearman_corrs.png"><img src="/assets/2025/chatterjees-xi/images/nonlinear_spearman_corrs.png"></a>
    <figcaption>Figure 6: Spearman's correlation is low on both the quadratic and sine. </figcaption>
</figure>

Even though there is a simple relationship between $$x$$ and $$y$$, Spearman's correlation is low.
From the definition, we see that in order for Spearman's to be close to 1, each $$x_i$$ needs to be paired with comparably ranked $$y_i$$. That is, small $$x_i$$ paired with small $$y_i$$ and large $$x_i$$ paired with large $$y_i$$, as measured by rank.

In the quadratic case, this does **not** hold.
For example $$x=-0.9$$ and $$x=1$$ both produce similarly ranked $$y$$ values (i.e. 0.81 and 1), yet the ranks of the $$x$$ values are far apart.

Similarly, for the sine wave, both $$x=-2\pi$$ and $$x=2\pi$$ yield the same $$y$$-value of 0.
Here, $$x=-2\pi$$ is assigned the lowest rank while its corresponding $$y$$-value has the highest rank.

What if we want to capture more than just monotonic relationships?
What if we want to measure how close $$y$$ is to being a noiseless function of $$x$$?

Surprisingly, in 2019, a very simple correlation coefficient was introduced that does exactly this.

## Chatterjee's Correlation

### Definition

If we first sort the data by their $$x$$-coordinates, Chatterjee's Xi coefficient is given by<sup>[2](#footnote2)</sup>

$$\xi(x,y) = 1 - {\sum_{i=1}^{N-1} |\mathbf{rank}(y_{i+1}) - \mathbf{rank}(y_i)| \over {(N^2-1)/3}}.$$

To better understand the formula's computation, let's apply it to the dataset below.

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

We can implement this logic in just a few lines of Python code (for the case of unique $$x$$-values).

<details>
<summary>
Click for code
</summary>

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

</details>
<br>

While this gives us a recipe for computing Chatterjee's Xi, we're still missing any actual intuition.
There is a whole list of questions we could ask at this point

- where does this formula come from?
- how is it measuring whether $$y$$ is a function of $$x$$?
- where does the $$(N^2-1)/3$$ in the denominator come from?
- why don't the $$x$$-values appear in the computation?

### Intuition

The main intuition behind Chatterjee's correlation is that if $$y$$ is a function of $$x$$, then we would expect a "small" change in $$x$$ to lead to a "small" change in $$y$$ (for "nice" functions).

Of course, small is a relative term and depends on units of measure and how fast the function is increasing/decreasing.
To avoid choosing a particular threshold for how "small" the change in $$x$$ needs to be, we can order our points $$(x_1, y_1), \ldots, (x_n, y_n)$$ by their $$x$$-coordinates.

Letting $$[i]$$ denote the index of the $$i^{th}$$ smallest $$x$$ value, our data becomes reordered as

$$(x_{[1]}, y_{[1]}), (x_{[2]}, y_{[2]}) \ldots, (x_{[N]}, y_{[N]}).$$

Now, a "small change in $$x$$" can be defined as a small change in the rank of $$x$$.
The smallest change would be a change of 1, which corresponds to choosing the neighbouring point on the $$x$$-axis.
Notice how this choice is completely independent of the magnitude or units of the $$x$$-values.

We're now left with the problem of how to mathematically encode the resulting "small change in $$y$$".
Intuitively, for a nicely behaved function we'd expect going from $$x_{[i]}$$ to $$x_{[i+1]}$$, not to change the corresponding $$y$$-value much.
In particular, we'd expect both points to have $$y$$-values with similar _rank_.
Mathematically,

$$\mathbf{rank}(y_{[i]}) \approx \mathbf{rank}(y_{[i+1]}).$$

We don't care if $$y_{[i]}$$ is bigger or smaller than $$y_{[i+1]}$$, just that they are "close" in rank.
The simplest way to measure this distance is

$$d_i = |\mathbf{rank}(y_{[i+1]}) - \mathbf{rank}(y_{[i]})|.$$

To measure if _all_ the points have neighbours with $$y$$-values similar in rank, we simply take the sum over all neighbouring pairs,

$$d = \sum_{i=1}^{N-1} |\mathbf{rank}(y_{[i+1]}) - \mathbf{rank}(y_{[i]})|.$$

The larger the value of $$d$$, the stronger the evidence that $$y$$ is just bouncing around with no relationship to the value of $$x$$.
But this leaves the question, "what is considered a large value of $$d$$?"

We can normalise $$d$$ by taking the ratio of $$d$$ and what we would get, on average, if we computed $$d$$ for a randomly shuffled version of $$y$$ (i.e. no relationship to $$x$$).

Since the average absolute rank difference is independent of the position $$i$$ (remember the $$y$$-values are uniformly shuffled), the normalising constant is

$$
\mathbf{E}[d] = (N-1)\cdot\mathbf{E} \left[|\mathbf{rank}(y_{[2]}) - \mathbf{rank}(y_{[1]})|\right]
$$

where $$\mathbf{E}[d]$$ means the _expected value_, or average value, of $$d$$.

We can estimate this quantity by taking many uniformly shuffled $$y$$-values and measuring the average absolute rank difference.
Repeating this experiment for various values of $$N$$ gives us an idea of what the expected absolute rank difference is as a function of $$N$$.

Figure 7 shows the result of running this for values of $$3\le N\le 25$$, each with 1500 trials to estimate the average.

<figure class>
    <a href="/assets/2025/chatterjees-xi/images/expected_rank_diff.png"><img src="/assets/2025/chatterjees-xi/images/expected_rank_diff.png"></a>
    <figcaption>Figure 7: Average absolute rank difference on uniform random data measured for various values of N using 1500 trials each.  </figcaption>
</figure>

From this figure, we can see that for a single pair of random ranks, the expected absolute difference is

$$\mathbf{E} \left[\lvert\mathbf{rank}(y_{[1]}) - \mathbf{rank}(y_{[2]})\rvert\right] = {N+1 \over 3}.$$

The total expected sum for a completely random relationship is

$$\mathbf{E}[d] = (N-1)(N+1)/3 = (N^2-1)/3.$$

The Xi coefficient is then constructed to measure how close our observed sum of differences is to this random baseline<sup>[3](#footnote3)</sup>.

$$
\begin{align*}
\xi(x,y) &= 1 - {d \over \mathbf{E}[d]} \\
&= 1 - {\sum_{i=1}^{N-1} |\mathbf{rank}(y_{i+1}) - \mathbf{rank}(y_i)| \over {(N^2-1)/3}}
\end{align*}
$$

When the relationship between $$x$$ and $$y$$ is completely random, $$d$$ will be close to the expected sum $$(N^2-1)/3$$, making $$\xi(x,y)\approx 0$$.
When there is a perfect functional relationship, the ranks of $$y$$ will change minimally between neighboring $$x$$ points, making the sum $$d$$ small and pushing $$\xi(x,y)\approx 1$$.

### How well does it work?

With the intuition for Xi in place, let's examine its performance on the nonlinear data we've discussed.

We'll start by revisiting the non-linear data from figure 6.
We saw that both Pearson's and Spearman's correlations failed to capture the clear functional relationships in the quadratic and sinusoidal datasets.
Figure 8 shows the same data, this time including Chatterjee's correlation.

<div class="chatterjee-widget" id="chatterjee-widget">
  <div class="chatterjee-controls">
    <label>
      Points (N)
      <input type="range" min="10" max="100" value="30" step="5" data-param="chatterjee-npoints">
      <span class="chatterjee-readout" data-readout="chatterjee-npoints">30</span>
    </label>
    <button type="button" class="chatterjee-button" id="chatterjee-reset">Reset</button>
  </div>
  <div class="chatterjee-plots">
    <div class="chatterjee-subplot">
      <svg class="chatterjee-plot" id="chatterjee-svg-quad" viewBox="0 0 320 280" preserveAspectRatio="xMidYMid meet">
        <rect x="0" y="0" width="320" height="280" fill="#0d1117"></rect>
        <g id="chatterjee-grid-quad"></g>
        <g id="chatterjee-axes-quad"></g>
        <g id="chatterjee-points-quad"></g>
      </svg>
      <div class="chatterjee-subplot-title">y = x²</div>
      <div class="chatterjee-info" id="chatterjee-info-quad">
        <span class="corr-pair">ρ: 0.00</span> | <span class="corr-pair">r<sub>s</sub>: 0.00</span> | <span class="corr-pair">ξ: 0.72</span>
      </div>
    </div>
    <div class="chatterjee-subplot">
      <svg class="chatterjee-plot" id="chatterjee-svg-sin" viewBox="0 0 320 280" preserveAspectRatio="xMidYMid meet">
        <rect x="0" y="0" width="320" height="280" fill="#0d1117"></rect>
        <g id="chatterjee-grid-sin"></g>
        <g id="chatterjee-axes-sin"></g>
        <g id="chatterjee-points-sin"></g>
      </svg>
      <div class="chatterjee-subplot-title">y = sin(x)</div>
      <div class="chatterjee-info" id="chatterjee-info-sin">
        <span class="corr-pair">ρ: 0.00</span> | <span class="corr-pair">r<sub>s</sub>: 0.00</span> | <span class="corr-pair">ξ: 0.63</span>
      </div>
    </div>
  </div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure 8: Chatterjee's Xi successfully detects the functional relationships that Pearson and Spearman miss. As the number of points increases, Chatterjee's Xi approaches 1.</figcaption>
</div>

From the figure, we can see that Chatterjee's coefficient is significantly higher than Pearson's and Spearman's, indicating that it succeeds in capturing the association between $$x$$ and $$y$$ even though the relationship is non-linear and non-monotonic.
But why are the correlations so much less than 1 even when there is a true functional relationship in both cases?

One important characteristic of Chatterjee's Xi is its dependence on the sample size, $$N$$.
For a function with a complex or "wiggly" shape, the coefficient's ability to detect the relationship improves with more data points.

As the number of points increases, Chatterjee's Xi approaches 1.
This is because Chatterjee's coefficient needs to detect arbitrary relationships $$y=f(x)$$<sup>[4](#footnote4)</sup>.
This requires a sufficient number of data points to distinguish a true functional relationship from random noise.
In general, the more "wiggly" the underlying function is, the more points will be required to get high Chatterjee's correlation

This also explains why, for a simple three-point line, Chatterjee's correlation is only 0.25.
Unlike Pearson's, which will have a correlation of 1, Chatterjee's Xi needs more evidence to be 'sure' that a functional relationship exists.

## Conclusion

In this post, we explored three different measures of correlation.

Pearson's correlation is the oldest, dating back to 1844.
Though it is a powerful and well understood tool for measuring linear relationships, it has limitations when measuring more complex non-linear data relationships and outliers.

Spearman's correlation proves to be a robust, rank-based alternative that captures monotonic relationships, as opposed to only linear ones.

Finally, over 175 years later, Stanford statistician Sourav Chatterjee introduced the xi coefficient in his 2019 paper "A new coefficient of correlation".
Chatterjee's Xi provides a mathematical formula for measuring the strength of general deterministic relationships between two variables.
This measure represents a significant leap forward, aligning the mathematical definition of correlation more closely with our intuitive, everyday understanding of a "relationship" between variables.

The simplicity of the formula for $$\xi(x,y)$$ makes it all the more surprising that it had remained undiscovered until 2019.
Chatterjee's coefficient is a truly remarkable example of how, even in a field as established as statistics, there are still simple and powerful ideas waiting to be discovered.

## Footnotes

<a name="footnote1">1</a>: In the case of ties, the $$\mathbf{corr}(\mathbf{rank}(x), \mathbf{rank}(y))$$ definition must be used.

<a name="footnote2">2</a>: In the case of ties in the $$y$$-values there is a more general form that can be found in the paper. We'll assume no ties in this post however.

<a name="footnote3">3</a>: Since the normalisation is done against the expected value of a random permutation, rather than the worst case, it is possible for Chatterjee's coefficient to be negative. Consider the $$y$$-ranks being $$[1,3,2]$$ after being ordered by $$x$$. This would yield $$d = 9/8$$ leading to $$\xi(x,y) = -1/8$$.

<a name="footnote4">4</a>: For this reason, $$\xi(x,y) \ne \xi(y,x)$$. Compare this with our other measures of correlation where $$x$$ and $$y$$ were completely interchangeable.

## References

1. [Chatterjee's original paper introducing Xi](https://arxiv.org/abs/1909.10140)
2. [Scipy discussion on including Chatterjee's Xi in Scipy 1.15.0](https://discuss.scientific-python.org/t/new-function-scipy-stats-xi-correlation/1498)
3. [Scipy documentation for chatterjeexi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chatterjeexi.html)

---

<style>
.pearson-widget {
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 12px;
  background: #161b22;
  margin: 1rem auto 1.5rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  max-width: 700px;
}

.pearson-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  align-items: center;
  margin-bottom: 10px;
}

.pearson-controls label {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.8rem;
  color: #c9d1d9;
  line-height: 1.1;
}

.pearson-controls input[type="range"] {
  width: 120px;
  height: 6px;
  accent-color: #58a6ff;
}

.pearson-readout {
  font-variant-numeric: tabular-nums;
  color: #8b949e;
  font-size: 0.85rem;
  min-width: 3em;
}

.pearson-plot {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 4px;
  background: #0d1117;
  border: 1px solid #30363d;
}

.pearson-info {
  font-size: 0.8rem;
  color: #8b949e;
  margin-top: 8px;
  padding: 8px 12px;
  background: rgba(88, 166, 255, 0.05);
  border-radius: 4px;
  border-left: 3px solid #58a6ff;
}

.pearson-button {
  padding: 5px 10px;
  border: 1px solid #30363d;
  border-radius: 6px;
  background: #0d1117;
  color: #c9d1d9;
  font-size: 0.8rem;
  font-family: inherit;
  cursor: pointer;
  transition: all 0.2s ease;
}

.pearson-button:hover {
  background: #21262d;
  border-color: #58a6ff;
}

@media (max-width: 600px) {
  .pearson-controls input[type="range"] {
    width: 90px;
  }
}

/* Spearman widget styles */
.spearman-widget {
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 12px;
  background: #161b22;
  margin: 1rem auto 1.5rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  max-width: 700px;
}

.spearman-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  align-items: center;
  margin-bottom: 10px;
}

.spearman-controls label {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.8rem;
  color: #c9d1d9;
  line-height: 1.1;
}

.spearman-controls input[type="range"] {
  width: 120px;
  height: 6px;
  accent-color: #58a6ff;
}

.spearman-readout {
  font-variant-numeric: tabular-nums;
  color: #8b949e;
  font-size: 0.85rem;
  min-width: 3em;
}

.spearman-plot {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 4px;
  background: #0d1117;
  border: 1px solid #30363d;
}

.spearman-info {
  font-size: 0.8rem;
  color: #8b949e;
  margin-top: 8px;
  padding: 8px 12px;
  background: rgba(88, 166, 255, 0.05);
  border-radius: 4px;
  border-left: 3px solid #58a6ff;
}

.spearman-button {
  padding: 5px 10px;
  border: 1px solid #30363d;
  border-radius: 6px;
  background: #0d1117;
  color: #c9d1d9;
  font-size: 0.8rem;
  font-family: inherit;
  cursor: pointer;
  transition: all 0.2s ease;
}

.spearman-button:hover {
  background: #21262d;
  border-color: #58a6ff;
}

@media (max-width: 600px) {
  .spearman-controls input[type="range"] {
    width: 90px;
  }
}

/* Chatterjee widget styles */
.chatterjee-widget {
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 12px;
  background: #161b22;
  margin: 1rem auto 1.5rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  max-width: 700px;
}

.chatterjee-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  align-items: center;
  margin-bottom: 10px;
}

.chatterjee-controls label {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.8rem;
  color: #c9d1d9;
  line-height: 1.1;
}

.chatterjee-controls input[type="range"] {
  width: 150px;
  height: 6px;
  accent-color: #58a6ff;
}

.chatterjee-readout {
  font-variant-numeric: tabular-nums;
  color: #8b949e;
  font-size: 0.85rem;
  min-width: 2.5em;
}

.chatterjee-plots {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
}

.chatterjee-subplot {
  flex: 1;
  min-width: 280px;
  max-width: 340px;
}

.chatterjee-plot {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 4px;
  background: #0d1117;
  border: 1px solid #30363d;
}

.chatterjee-subplot-title {
  text-align: center;
  font-size: 0.85rem;
  color: #c9d1d9;
  margin-top: 6px;
  font-style: italic;
}

.chatterjee-info {
  font-size: 0.8rem;
  color: #8b949e;
  margin-top: 8px;
  padding: 8px 12px;
  background: rgba(88, 166, 255, 0.05);
  border-radius: 4px;
  border-left: 3px solid #58a6ff;
}

.chatterjee-info .corr-pair {
  white-space: nowrap;
}

.chatterjee-button {
  padding: 5px 10px;
  border: 1px solid #30363d;
  border-radius: 6px;
  background: #0d1117;
  color: #c9d1d9;
  font-size: 0.8rem;
  font-family: inherit;
  cursor: pointer;
  transition: all 0.2s ease;
}

.chatterjee-button:hover {
  background: #21262d;
  border-color: #58a6ff;
}

@media (max-width: 600px) {
  .chatterjee-controls input[type="range"] {
    width: 100px;
  }
  .chatterjee-plots {
    flex-direction: column;
    align-items: center;
  }
  .chatterjee-subplot {
    max-width: 100%;
  }
}
</style>

<script>
(function() {
  const W = 500, H = 400;
  const margin = { left: 50, right: 30, top: 30, bottom: 40 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  const xMin = -3.5, xMax = 3.5;
  const yMin = -3.5, yMax = 3.5;

  const gridG = document.getElementById('pearson-grid');
  const axesG = document.getElementById('pearson-axes');
  const pointsG = document.getElementById('pearson-points');
  const infoDiv = document.getElementById('pearson-info');

  function toSvgX(x) { return margin.left + (x - xMin) / (xMax - xMin) * plotW; }
  function toSvgY(y) { return margin.top + (yMax - y) / (yMax - yMin) * plotH; }

  // Seeded random number generator (Mulberry32)
  function mulberry32(seed) {
    return function() {
      let t = seed += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // Box-Muller transform for normal distribution
  function normalRandom(rng) {
    const u1 = rng();
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  // Generate correlated data: y = rho * x + sqrt(1 - rho^2) * noise
  function generateData(n, rho, seed) {
    const rng = mulberry32(seed);
    const xs = [];
    const ys = [];

    for (let i = 0; i < n; i++) {
      const x = normalRandom(rng);
      const noise = normalRandom(rng);
      const y = rho * x + Math.sqrt(1 - rho * rho) * noise;
      xs.push(x);
      ys.push(y);
    }

    return { xs, ys };
  }

  // Calculate sample Pearson correlation
  function pearsonCorr(xs, ys) {
    const n = xs.length;
    const xMean = xs.reduce((a, b) => a + b, 0) / n;
    const yMean = ys.reduce((a, b) => a + b, 0) / n;

    let covXY = 0, varX = 0, varY = 0;
    for (let i = 0; i < n; i++) {
      const dx = xs[i] - xMean;
      const dy = ys[i] - yMean;
      covXY += dx * dy;
      varX += dx * dx;
      varY += dy * dy;
    }

    return covXY / Math.sqrt(varX * varY);
  }

  function drawGrid() {
    gridG.innerHTML = '';
    for (let v = -3; v <= 3; v += 1) {
      const sx = toSvgX(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', sx); line.setAttribute('y1', margin.top);
      line.setAttribute('x2', sx); line.setAttribute('y2', H - margin.bottom);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
    for (let v = -3; v <= 3; v += 1) {
      const sy = toSvgY(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', margin.left); line.setAttribute('y1', sy);
      line.setAttribute('x2', W - margin.right); line.setAttribute('y2', sy);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
  }

  function drawAxes() {
    axesG.innerHTML = '';
    // X axis
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin.left); xAxis.setAttribute('y1', toSvgY(0));
    xAxis.setAttribute('x2', W - margin.right); xAxis.setAttribute('y2', toSvgY(0));
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(xAxis);

    // Y axis
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', toSvgX(0)); yAxis.setAttribute('y1', margin.top);
    yAxis.setAttribute('x2', toSvgX(0)); yAxis.setAttribute('y2', H - margin.bottom);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(yAxis);

    // X axis labels
    for (let v = -3; v <= 3; v += 1) {
      if (v === 0) continue;
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', toSvgX(v)); label.setAttribute('y', H - margin.bottom + 18);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'middle');
      label.textContent = v;
      axesG.appendChild(label);
    }

    // Y axis labels
    for (let v = -3; v <= 3; v += 1) {
      if (v === 0) continue;
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - 10); label.setAttribute('y', toSvgY(v) + 4);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'end');
      label.textContent = v;
      axesG.appendChild(label);
    }

    // Axis labels
    const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xLabel.setAttribute('x', W / 2); xLabel.setAttribute('y', H - 5);
    xLabel.setAttribute('fill', '#8b949e'); xLabel.setAttribute('font-size', '13');
    xLabel.setAttribute('text-anchor', 'middle');
    xLabel.textContent = 'x';
    axesG.appendChild(xLabel);

    const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yLabel.setAttribute('x', 12); yLabel.setAttribute('y', H / 2);
    yLabel.setAttribute('fill', '#8b949e'); yLabel.setAttribute('font-size', '13');
    yLabel.setAttribute('text-anchor', 'middle');
    yLabel.setAttribute('transform', `rotate(-90, 12, ${H/2})`);
    yLabel.textContent = 'y';
    axesG.appendChild(yLabel);
  }

  function update() {
    const rho = parseFloat(document.querySelector('[data-param="corr"]').value);
    const nPoints = parseInt(document.querySelector('[data-param="npoints"]').value);

    document.querySelector('[data-readout="corr"]').textContent = rho.toFixed(2);
    document.querySelector('[data-readout="npoints"]').textContent = nPoints;

    // Use a fixed seed for reproducibility (changes with rho and n)
    const seed = Math.abs(Math.round(rho * 1000)) + nPoints * 7;
    const { xs, ys } = generateData(nPoints, rho, seed);

    // Draw points
    pointsG.innerHTML = '';
    for (let i = 0; i < xs.length; i++) {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', toSvgX(xs[i]));
      circle.setAttribute('cy', toSvgY(ys[i]));
      circle.setAttribute('r', 5);
      circle.setAttribute('fill', '#58a6ff');
      circle.setAttribute('fill-opacity', '0.7');
      pointsG.appendChild(circle);
    }

    // Calculate sample correlation
    const sampleCorr = pearsonCorr(xs, ys);
    infoDiv.innerHTML = `ρ = ${rho.toFixed(2)} | N = ${nPoints} | Sample correlation: ${sampleCorr.toFixed(2)}`;
  }

  drawGrid();
  drawAxes();
  update();

  document.querySelector('[data-param="corr"]').addEventListener('input', update);
  document.querySelector('[data-param="npoints"]').addEventListener('input', update);
  document.getElementById('pearson-reset').addEventListener('click', () => {
    document.querySelector('[data-param="corr"]').value = 0.5;
    document.querySelector('[data-param="npoints"]').value = 60;
    update();
  });
})();

// Spearman widget
(function() {
  const W = 500, H = 400;
  const margin = { left: 50, right: 30, top: 30, bottom: 40 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  const xMin = -3.5, xMax = 3.5;
  const yMin = -3.5, yMax = 3.5;

  const gridG = document.getElementById('spearman-grid');
  const axesG = document.getElementById('spearman-axes');
  const pointsG = document.getElementById('spearman-points');
  const infoDiv = document.getElementById('spearman-info');

  function toSvgX(x) { return margin.left + (x - xMin) / (xMax - xMin) * plotW; }
  function toSvgY(y) { return margin.top + (yMax - y) / (yMax - yMin) * plotH; }

  // Seeded random number generator (Mulberry32)
  function mulberry32(seed) {
    return function() {
      let t = seed += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // Box-Muller transform for normal distribution
  function normalRandom(rng) {
    const u1 = rng();
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  // Generate correlated data: y = rho * x + sqrt(1 - rho^2) * noise
  function generateData(n, rho, seed) {
    const rng = mulberry32(seed);
    const xs = [];
    const ys = [];

    for (let i = 0; i < n; i++) {
      const x = normalRandom(rng);
      const noise = normalRandom(rng);
      const y = rho * x + Math.sqrt(1 - rho * rho) * noise;
      xs.push(x);
      ys.push(y);
    }

    return { xs, ys };
  }

  // Calculate sample Pearson correlation
  function pearsonCorr(xs, ys) {
    const n = xs.length;
    const xMean = xs.reduce((a, b) => a + b, 0) / n;
    const yMean = ys.reduce((a, b) => a + b, 0) / n;

    let covXY = 0, varX = 0, varY = 0;
    for (let i = 0; i < n; i++) {
      const dx = xs[i] - xMean;
      const dy = ys[i] - yMean;
      covXY += dx * dy;
      varX += dx * dx;
      varY += dy * dy;
    }

    return covXY / Math.sqrt(varX * varY);
  }

  // Calculate ranks (1-based)
  function ranks(arr) {
    const indexed = arr.map((v, i) => ({ v, i }));
    indexed.sort((a, b) => a.v - b.v);
    const r = new Array(arr.length);
    for (let rank = 1; rank <= indexed.length; rank++) {
      r[indexed[rank - 1].i] = rank;
    }
    return r;
  }

  // Calculate Spearman's rank correlation
  function spearmanCorr(xs, ys) {
    const n = xs.length;
    const rankX = ranks(xs);
    const rankY = ranks(ys);

    let sumSqDiff = 0;
    for (let i = 0; i < n; i++) {
      const d = rankX[i] - rankY[i];
      sumSqDiff += d * d;
    }

    return 1 - (6 * sumSqDiff) / (n * (n * n - 1));
  }

  function drawGrid() {
    gridG.innerHTML = '';
    for (let v = -3; v <= 3; v += 1) {
      const sx = toSvgX(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', sx); line.setAttribute('y1', margin.top);
      line.setAttribute('x2', sx); line.setAttribute('y2', H - margin.bottom);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
    for (let v = -3; v <= 3; v += 1) {
      const sy = toSvgY(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', margin.left); line.setAttribute('y1', sy);
      line.setAttribute('x2', W - margin.right); line.setAttribute('y2', sy);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
  }

  function drawAxes() {
    axesG.innerHTML = '';
    // X axis
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin.left); xAxis.setAttribute('y1', toSvgY(0));
    xAxis.setAttribute('x2', W - margin.right); xAxis.setAttribute('y2', toSvgY(0));
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(xAxis);

    // Y axis
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', toSvgX(0)); yAxis.setAttribute('y1', margin.top);
    yAxis.setAttribute('x2', toSvgX(0)); yAxis.setAttribute('y2', H - margin.bottom);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(yAxis);

    // X axis labels
    for (let v = -3; v <= 3; v += 1) {
      if (v === 0) continue;
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', toSvgX(v)); label.setAttribute('y', H - margin.bottom + 18);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'middle');
      label.textContent = v;
      axesG.appendChild(label);
    }

    // Y axis labels
    for (let v = -3; v <= 3; v += 1) {
      if (v === 0) continue;
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - 10); label.setAttribute('y', toSvgY(v) + 4);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '11');
      label.setAttribute('text-anchor', 'end');
      label.textContent = v;
      axesG.appendChild(label);
    }

    // Axis labels
    const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xLabel.setAttribute('x', W / 2); xLabel.setAttribute('y', H - 5);
    xLabel.setAttribute('fill', '#8b949e'); xLabel.setAttribute('font-size', '13');
    xLabel.setAttribute('text-anchor', 'middle');
    xLabel.textContent = 'x';
    axesG.appendChild(xLabel);

    const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yLabel.setAttribute('x', 12); yLabel.setAttribute('y', H / 2);
    yLabel.setAttribute('fill', '#8b949e'); yLabel.setAttribute('font-size', '13');
    yLabel.setAttribute('text-anchor', 'middle');
    yLabel.setAttribute('transform', `rotate(-90, 12, ${H/2})`);
    yLabel.textContent = 'y';
    axesG.appendChild(yLabel);
  }

  function update() {
    const rho = parseFloat(document.querySelector('[data-param="spearman-corr"]').value);
    const nPoints = parseInt(document.querySelector('[data-param="spearman-npoints"]').value);

    document.querySelector('[data-readout="spearman-corr"]').textContent = rho.toFixed(2);
    document.querySelector('[data-readout="spearman-npoints"]').textContent = nPoints;

    // Use a fixed seed for reproducibility (changes with rho and n)
    const seed = Math.abs(Math.round(rho * 1000)) + nPoints * 7 + 12345;
    const { xs, ys } = generateData(nPoints, rho, seed);

    // Draw points
    pointsG.innerHTML = '';
    for (let i = 0; i < xs.length; i++) {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', toSvgX(xs[i]));
      circle.setAttribute('cy', toSvgY(ys[i]));
      circle.setAttribute('r', 5);
      circle.setAttribute('fill', '#58a6ff');
      circle.setAttribute('fill-opacity', '0.7');
      pointsG.appendChild(circle);
    }

    // Calculate both correlations
    const samplePearson = pearsonCorr(xs, ys);
    const sampleSpearman = spearmanCorr(xs, ys);
    infoDiv.innerHTML = `ρ = ${rho.toFixed(2)} | N = ${nPoints} | Pearson: ${samplePearson.toFixed(2)} | Spearman: ${sampleSpearman.toFixed(2)}`;
  }

  drawGrid();
  drawAxes();
  update();

  document.querySelector('[data-param="spearman-corr"]').addEventListener('input', update);
  document.querySelector('[data-param="spearman-npoints"]').addEventListener('input', update);
  document.getElementById('spearman-reset').addEventListener('click', () => {
    document.querySelector('[data-param="spearman-corr"]').value = 0.5;
    document.querySelector('[data-param="spearman-npoints"]').value = 60;
    update();
  });
})();

// Chatterjee widget
(function() {
  const W = 320, H = 280;
  const margin = { left: 45, right: 20, top: 20, bottom: 35 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  // Quadratic plot bounds
  const quadXMin = -1.2, quadXMax = 1.2;
  const quadYMin = -0.2, quadYMax = 1.3;

  // Sinusoidal plot bounds
  const sinXMin = -7, sinXMax = 7;
  const sinYMin = -1.3, sinYMax = 1.3;

  // Coordinate transforms for quadratic
  function toSvgXQuad(x) { return margin.left + (x - quadXMin) / (quadXMax - quadXMin) * plotW; }
  function toSvgYQuad(y) { return margin.top + (quadYMax - y) / (quadYMax - quadYMin) * plotH; }

  // Coordinate transforms for sinusoidal
  function toSvgXSin(x) { return margin.left + (x - sinXMin) / (sinXMax - sinXMin) * plotW; }
  function toSvgYSin(y) { return margin.top + (sinYMax - y) / (sinYMax - sinYMin) * plotH; }

  // Generate evenly spaced array
  function linspace(start, end, n) {
    const step = (end - start) / (n - 1);
    return Array.from({length: n}, (_, i) => start + i * step);
  }

  // Calculate ranks (1-based)
  function ranks(arr) {
    const indexed = arr.map((v, i) => ({ v, i }));
    indexed.sort((a, b) => a.v - b.v);
    const r = new Array(arr.length);
    for (let rank = 1; rank <= indexed.length; rank++) {
      r[indexed[rank - 1].i] = rank;
    }
    return r;
  }

  // Pearson correlation
  function pearsonCorr(xs, ys) {
    const n = xs.length;
    const xMean = xs.reduce((a, b) => a + b, 0) / n;
    const yMean = ys.reduce((a, b) => a + b, 0) / n;

    let covXY = 0, varX = 0, varY = 0;
    for (let i = 0; i < n; i++) {
      const dx = xs[i] - xMean;
      const dy = ys[i] - yMean;
      covXY += dx * dy;
      varX += dx * dx;
      varY += dy * dy;
    }

    if (varX === 0 || varY === 0) return 0;
    return covXY / Math.sqrt(varX * varY);
  }

  // Spearman correlation
  function spearmanCorr(xs, ys) {
    const n = xs.length;
    const rankX = ranks(xs);
    const rankY = ranks(ys);

    let sumSqDiff = 0;
    for (let i = 0; i < n; i++) {
      const d = rankX[i] - rankY[i];
      sumSqDiff += d * d;
    }

    return 1 - (6 * sumSqDiff) / (n * (n * n - 1));
  }

  // Chatterjee's Xi correlation
  function chatterjeeCorr(xs, ys) {
    const n = xs.length;

    // Sort by x values
    const indexed = xs.map((x, i) => ({ x, y: ys[i] }));
    indexed.sort((a, b) => a.x - b.x);
    const ysSorted = indexed.map(p => p.y);

    // Get ranks of y values (after sorting by x)
    const yRanks = ranks(ysSorted);

    // Sum of absolute rank differences
    let sumAbsDiff = 0;
    for (let i = 0; i < n - 1; i++) {
      sumAbsDiff += Math.abs(yRanks[i + 1] - yRanks[i]);
    }

    const expected = (n * n - 1) / 3;
    return 1 - sumAbsDiff / expected;
  }

  // Draw grid
  function drawGrid(gridG, toSvgX, toSvgY, xTicks, yTicks) {
    gridG.innerHTML = '';
    for (const v of xTicks) {
      const sx = toSvgX(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', sx); line.setAttribute('y1', margin.top);
      line.setAttribute('x2', sx); line.setAttribute('y2', H - margin.bottom);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
    for (const v of yTicks) {
      const sy = toSvgY(v);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', margin.left); line.setAttribute('y1', sy);
      line.setAttribute('x2', W - margin.right); line.setAttribute('y2', sy);
      line.setAttribute('stroke', '#21262d'); line.setAttribute('stroke-width', 1);
      gridG.appendChild(line);
    }
  }

  // Draw axes with labels
  function drawAxes(axesG, toSvgX, toSvgY, xTicks, yTicks, xLabel, yLabel) {
    axesG.innerHTML = '';

    // X axis (at y=0)
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin.left); xAxis.setAttribute('y1', toSvgY(0));
    xAxis.setAttribute('x2', W - margin.right); xAxis.setAttribute('y2', toSvgY(0));
    xAxis.setAttribute('stroke', '#484f58'); xAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(xAxis);

    // Y axis (at x=0)
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', toSvgX(0)); yAxis.setAttribute('y1', margin.top);
    yAxis.setAttribute('x2', toSvgX(0)); yAxis.setAttribute('y2', H - margin.bottom);
    yAxis.setAttribute('stroke', '#484f58'); yAxis.setAttribute('stroke-width', 1.5);
    axesG.appendChild(yAxis);

    // X axis tick labels
    for (const v of xTicks) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', toSvgX(v)); label.setAttribute('y', H - margin.bottom + 15);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '10');
      label.setAttribute('text-anchor', 'middle');
      label.textContent = v;
      axesG.appendChild(label);
    }

    // Y axis tick labels
    for (const v of yTicks) {
      if (v === 0) continue;
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - 8); label.setAttribute('y', toSvgY(v) + 3);
      label.setAttribute('fill', '#6e7681'); label.setAttribute('font-size', '10');
      label.setAttribute('text-anchor', 'end');
      label.textContent = v;
      axesG.appendChild(label);
    }

    // X axis label
    const xLabelEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xLabelEl.setAttribute('x', W / 2); xLabelEl.setAttribute('y', H - 5);
    xLabelEl.setAttribute('fill', '#8b949e'); xLabelEl.setAttribute('font-size', '11');
    xLabelEl.setAttribute('text-anchor', 'middle');
    xLabelEl.textContent = xLabel;
    axesG.appendChild(xLabelEl);

    // Y axis label
    const yLabelEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yLabelEl.setAttribute('x', 10); yLabelEl.setAttribute('y', H / 2);
    yLabelEl.setAttribute('fill', '#8b949e'); yLabelEl.setAttribute('font-size', '11');
    yLabelEl.setAttribute('text-anchor', 'middle');
    yLabelEl.setAttribute('transform', `rotate(-90, 10, ${H/2})`);
    yLabelEl.textContent = yLabel;
    axesG.appendChild(yLabelEl);
  }

  // Convert to SVG path
  function toPath(xs, ys, toSvgX, toSvgY) {
    let d = '';
    for (let i = 0; i < xs.length; i++) {
      const sx = toSvgX(xs[i]);
      const sy = toSvgY(ys[i]);
      if (i === 0) {
        d = `M ${sx} ${sy}`;
      } else {
        d += ` L ${sx} ${sy}`;
      }
    }
    return d;
  }

  // Draw sample points
  function drawPoints(pointsG, xs, ys, toSvgX, toSvgY) {
    pointsG.innerHTML = '';
    for (let i = 0; i < xs.length; i++) {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', toSvgX(xs[i]));
      circle.setAttribute('cy', toSvgY(ys[i]));
      circle.setAttribute('r', 5);
      circle.setAttribute('fill', '#58a6ff');
      circle.setAttribute('fill-opacity', '0.7');
      pointsG.appendChild(circle);
    }
  }

  // Initialize grids and axes
  const quadGridG = document.getElementById('chatterjee-grid-quad');
  const quadAxesG = document.getElementById('chatterjee-axes-quad');
  const quadPointsG = document.getElementById('chatterjee-points-quad');
  const quadInfoDiv = document.getElementById('chatterjee-info-quad');

  const sinGridG = document.getElementById('chatterjee-grid-sin');
  const sinAxesG = document.getElementById('chatterjee-axes-sin');
  const sinPointsG = document.getElementById('chatterjee-points-sin');
  const sinInfoDiv = document.getElementById('chatterjee-info-sin');

  // Draw static elements
  drawGrid(quadGridG, toSvgXQuad, toSvgYQuad, [-1, 0, 1], [0, 0.5, 1]);
  drawAxes(quadAxesG, toSvgXQuad, toSvgYQuad, [-1, 0, 1], [0, 0.5, 1], 'x', 'y');

  drawGrid(sinGridG, toSvgXSin, toSvgYSin, [-6, -3, 0, 3, 6], [-1, 0, 1]);
  drawAxes(sinAxesG, toSvgXSin, toSvgYSin, [-6, -3, 0, 3, 6], [-1, 0, 1], 'x', 'y');

  function update() {
    const nPoints = parseInt(document.querySelector('[data-param="chatterjee-npoints"]').value);
    document.querySelector('[data-readout="chatterjee-npoints"]').textContent = nPoints;

    // Generate sample points for quadratic
    const quadXs = linspace(-1, 1, nPoints);
    const quadYs = quadXs.map(x => x * x);

    // Generate sample points for sinusoidal
    const sinXs = linspace(-2 * Math.PI, 2 * Math.PI, nPoints);
    const sinYs = sinXs.map(x => Math.sin(x));

    // Draw points
    drawPoints(quadPointsG, quadXs, quadYs, toSvgXQuad, toSvgYQuad);
    drawPoints(sinPointsG, sinXs, sinYs, toSvgXSin, toSvgYSin);

    // Calculate correlations for quadratic
    const quadPearson = pearsonCorr(quadXs, quadYs);
    const quadSpearman = spearmanCorr(quadXs, quadYs);
    const quadXi = chatterjeeCorr(quadXs, quadYs);

    // Calculate correlations for sinusoidal
    const sinPearson = pearsonCorr(sinXs, sinYs);
    const sinSpearman = spearmanCorr(sinXs, sinYs);
    const sinXi = chatterjeeCorr(sinXs, sinYs);

    // Update info displays
    quadInfoDiv.innerHTML = `<span class="corr-pair">ρ: ${quadPearson.toFixed(2)}</span> | <span class="corr-pair">r<sub>s</sub>: ${quadSpearman.toFixed(2)}</span> | <span class="corr-pair">ξ: ${quadXi.toFixed(2)}</span>`;
    sinInfoDiv.innerHTML = `<span class="corr-pair">ρ: ${sinPearson.toFixed(2)}</span> | <span class="corr-pair">r<sub>s</sub>: ${sinSpearman.toFixed(2)}</span> | <span class="corr-pair">ξ: ${sinXi.toFixed(2)}</span>`;
  }

  update();

  document.querySelector('[data-param="chatterjee-npoints"]').addEventListener('input', update);
  document.getElementById('chatterjee-reset').addEventListener('click', () => {
    document.querySelector('[data-param="chatterjee-npoints"]').value = 30;
    update();
  });
})();
</script>
