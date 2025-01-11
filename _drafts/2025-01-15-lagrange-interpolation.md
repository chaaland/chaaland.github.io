---
title: "Lagrange Interpolation"
categories:
  - Mathematics
date:   2025-10-15 10:57:00 +0100
mathjax: true
tags:
  - Numerical methods
  - Regression
toc: true
# classes: wide
excerpt: ""
header: 
  overlay_image: ../assets/images/shakespeare-zipf-param-surface-splash.png
  overlay_filter: 0.2
---

This post is a light introduction to Gaussian Processes (GPs). The goal is to give an explanation of some of key results and most importantly, some intuition for how they come about.

## Motivation

Imagine back in the days of yore.
Before the internet, before calculators.
How would you compute a number's logarithm?
Historically, someone would arduously work through the arithmetic to compute the logarithm for a variety of values up to a certain precision and then publish a table.

The table might look something like this
┌──────┬────────────┐
│ x    ┆ y          │
╞══════╪════════════╡
│ 0.8  ┆ -0.223144  │
│ 1.0  ┆ 0.0        │
│ 1.2  ┆ 0.182322   │
│ 1.4  ┆ 0.336472   │
│ 1.6  ┆ 0.470004   │
│ 1.8  ┆ 0.587787   │
└──────┴────────────┘

But what if you want the value of the logarithm at a value _not_ in the table?
What if you really wanted to know the logarithm of 1.35?

The easiest thing to do is a weighted average to take into account that 1.35 is closer to 1.4 than to 1.2 (both of which have logarithms in the table).
$$\log(1.35) \approx 0.25 \log(1.2) + 0.75 \log(1.4)$$

This amounts to using a _linear_ interpolation between 1.2 and 1.4.
The figure below shows the line passing passing through both points and the corresponding estimate of $$\log(1.35)$$.

<figure class>
    <a href="/assets/lagrange-duality/images/logarithm_lerp.png"><img src="/assets/lagrange-duality/images/logarithm_lerp.png"></a>
    <figcaption>Figure 1</figcaption>
</figure>

$$
\begin{align*}
f_{X_1,X_2}(x_1,x_2) &= f_{X_1}(x_1)f_{X_2}(x_2) \\
 &= \frac{1}{\sqrt{2\pi \sigma_1^2}}\exp\left(-\frac{(x_1-\mu_1)^2}{2\sigma_1^2}\right)\frac{1}{\sqrt{2\pi \sigma_2^2}}\exp\left(-\frac{(x_2-\mu_2)^2}{2\sigma_2^2}\right)\\
 &= \frac{1}{2\pi\sqrt{\sigma_1^2 \sigma_2^2}}\exp\left(-\frac{(x_1-\mu_1)^2}{2\sigma_1^2}-\frac{(x_2-\mu_2)^2}{2\sigma_2^2}\right)\\
\end{align*}
$$

## Footnotes

<a name="footnote1">1</a>: The expectation obeys 

$$\mathbf{E}[aX + bY] = a\mathbf{E}[X] + b \mathbf{E}[Y].$$

This is true regardless of the relationship between $$X$$ and $$Y$$.

<a name="footnote2">2</a>: The inverse covariance matrix which often appears is called the _precision matrix_

<a name="footnote3">3</a>: Pay close attention to how the subscript indices come in pairs. In particular, this colorised version highlights the pattern $$\hat{y}_{\color{red}1} = \Sigma_{\color{red}1\color{orange}2}\Sigma_{\color{orange}2\color{yellow}2}^{-1}y_{\color{yellow}2}$$. This helps when needing to recall the formula from memory.


## References

1. [Zipf's Law Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law)
