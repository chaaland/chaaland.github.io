---
title: "Better Contour and Surface Plots Using Non-Rectangular Sampling Grids"
categories:
  - Mathematics
date:   2019-09-29 13:32:45 +0100
mathjax: true
tags:
  - Optimization
  - Nonlinear modeling  
  - Gauss-Newton
  - Least Squares
  - Regression
toc: true
toc_label: 
# classes: wide
excerpt: ""
header: 
  overlay_image: assets/images/shakespeare-zipf-param-surface-splash.png
  overlay_filter: 0.2
---




# The Problem with Rectangular Grids
The standard way of creating contour and surface plots of a function $$f:\mathbf{R}^2 \rightarrow \mathbf{R}$$ is first creating a rectangular grid of $$(x,y)$$ coordinates (using something like `meshgrid`), then evaluating the function $$f$$ elementwise at each $$(x,y)$$ pair.

Fitting linear models using least squares is so ubiquitous you would be hard pressed to find a field in which it has not found application. A large part of the reason ordinary least squares (OLS) is so prevalent is that many simply aren't familiar with non-linear methods. Historically, solving nonlinear least squares (NLLS) problems was computationally expensive, but with modern computing power the barrier is less computation and moreso people's familarity with the methods. As we'll see, solving NLLS problems is just as simple as OLS.

## Zipf's Law
Before moving on to the main focus of this post, it helps to have a concrete problem to motivate the material. Consider modeling the frequency of a word's appearance in a corpus with vocabulary $$V$$, against its rank (by frequency). For this example, I used the text of Shakespeare's _Hamlet_ which was downloaded using python's `requests` library to get the raw html, then parsed using `BeautifulSoup` as shown in the following script 

{% highlight python %}
import os
import re
import requests
from bs4 import BeautifulSoup
pjoin = os.path.join


def download_text(url: str) -> str:
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, "html.parser")
    text = soup.find_all(text=True)

    return text


if __name__ == "__main__":
    if "hamlet.txt" not in os.listdir(pjoin("..", "txt")):
        hamlet_url = "http://shakespeare.mit.edu/hamlet/full.html"
        hamlet_text = download_text(hamlet_url)
        output = ""
        for t in hamlet_text:
            if t.parent.name.lower() == "a":
                output += f"{t} "

        with open(pjoin("..", "txt", "hamlet.txt"), "wt") as f:
            f.writelines(output)
{% endhighlight %}

With the text downloaded, we can easily write a function that computes the overall frequencies of each word in the text. Python even has a specialised `dict` called `Counter` in the `collections` module that does almost exactly this (except it gives raw counts instead of frequencies). The below code uses a simple regex to remove non alphanumeric characters, before splitting the string on whitespace. A more specialised library for tokenisation like `nltk` would be better suited for anything other than a simple example like this

{% highlight python %}
def word_freqs(topk: int = 25):
    word_counts = Counter() 
    with open(pjoin("..", "txt", "hamlet.txt"), "rt") as f:
        text = " ".join(f.readlines())
        text = re.sub("[^0-9a-zA-Z ]+", "", text)
        all_words = [word.lower() for word in text.split() if len(word) > 0]
        n_words = len(all_words)
        word_freqs = Counter(all_words)
        for word in word_freqs.keys():
            word_freqs[word] /= n_words

    words, freqs = zip(*word_freqs.most_common(topk))
    return words, np.asarray(freqs)
{% endhighlight %}

We can now use this data to get a plot of top 100 word's frequencies. Plotting freq vs. rank as in Figure 1 gives a good indication that a _power law_ of the form 

$$ f(r) = Kr^\alpha $$

where $$f :\{1,2, \ldots, \lvert V \rvert\} \rightarrow \mathbf{R}$$, could be a reasonable fit to this data

As it turns out, this observation is not a specific characteristic of _Hamlet_, but is one example of a broader phenomenon known as _Zipf's Law_. From Wikipedia,
> Zipf's law is an empirical law formulated using mathematical statistics that refers to the fact that many types of data studied in the physical and social sciences can be approximated with a Zipfian distribution, one of a family of related discrete power law probability distributions

Though Zipf's Law gives a parametric form for the model, what remains is to find a suitable $$K$$ and $$\alpha$$ that best fit the data.

## Ordinary Least Squares
If you've taken any linear algebra, you'll most likely recall the least squares optimisation problem

$$
\begin{equation}
\underset{x}{\text{minimize}} \quad ||Ax - y||^2
\end{equation}
$$

where $$y\in \mathbf{R}^{m}$$ and $$A\in \mathbf{R}^{m\times n}$$ with $$m \ge n$$ and having _linearly independent_ columns. The vector inside the Euclidean norm is often called the _residual vector_ and denoted $$r \in \mathbf{R}^m$$. Ordinary least squares problems are ideal for a number of reasons but the most salient are 
- objective is differentiable
- objective is _convex_ 
- closed form solution given by the _normal equations_

$$A^TAx^{\star} = A^T y$$

The power law model proposed to fit the empirical word frequency data is very clearly nonlinear in the parameters $$K$$ and $$\alpha$$. Ideally we'd like to find a $$K$$ and $$\alpha$$ that solve the following optimisation problem

$$\underset{K,\, \alpha}{\text{minimize}}\quad \sum_{i=1}^{|V|} \left(f_i - Kr_i^\alpha\right)^2$$

where $$f_i\in\mathbf{R}_+$$ is the empircal frequency of the word with rank $$r_i$$. If instead of fitting the empirical frequencies with a power law, we fit the logarithm of the frequencies<sup>[1](#footnote1)</sup>, the optimisation problem becomes

$$\underset{K,\, \alpha}{\text{minimize}}\quad \sum_{i=1}^{|V|} \left(\log f_i - \log K - \alpha \log r_i\right)^2$$

This logarithmic transform moves the data from the $$f-r$$ space to the $$\log f-\log r$$ space where we can more clearly see the problem as a simple linear regression.

<figure class="half">
    <a href="/assets/images/shakespeare-freq-scatter.png"><img src="/assets/images/shakespeare-freq-scatter.png"></a>
    <a href="/assets/images/shakespeare-zipf-transformed-param-scatter.png"><img src="/assets/images/shakespeare-zipf-transformed-param-scatter.png"></a>
    <figcaption>Figure 1</figcaption>
</figure>

Since the logarithm makes the model linear with respect to the parameters $$\log K$$ and $$\alpha$$, we can write this OLS problem in standard form with $$A\in \mathbf{R}^{|V| \times 2}$$, $$x\in\mathbf{R}^2$$, and $$y\in \mathbf{R}^{|V|}$$ as below.

$$
A =
\begin{bmatrix}
1 & \log r_1\\
1 & \log r_2\\
\vdots & \vdots\\
1 & \log r_{|V|}\\
\end{bmatrix},\,
x = 
\begin{bmatrix}
\log K\\
\alpha \\
\end{bmatrix}, \,
y = 
\begin{bmatrix}
\log f_1\\
\log f_2\\
\vdots\\
\log f_{|V|}\\
\end{bmatrix}
$$

<figure class="half">
    <a href="/assets/images/shakespeare-zipf-transformed-param-contours.png"><img src="/assets/images/shakespeare-zipf-transformed-param-contours.png"></a>
    <a href="/assets/images/shakespeare-zipf-transformed-param-surface.png"><img src="/assets/images/shakespeare-zipf-transformed-param-surface.png"></a>
</figure>

The plots of the objective function above confirm it is convex quadratic with a unique minimum. From the plot we can see the minimum is achieved in the region where $$\log K\approx -2.5$$  and $$\alpha\approx -0.8$$. Using python's `scipy.optimize` module to fit a linear model on the Hamlet data gives

$$f_{OLS}(r) = 0.079 r^{-0.83}$$

{% highlight python %}
import numpy as np
from scipy.optimize import least_squares

def fit_zipf_ols(freq_counts_desc: np.ndarray):
    ranks = np.arange(1, 1 + freq_counts_desc.size)
    A = np.stack([np.ones_like(freq_counts_desc), np.log(ranks)], axis=1)
    y = np.log(freq_counts_desc)
    f = lambda x: A @ x - y
    opt_result = least_squares(f, x0=np.random.randn(2), loss="linear")

    K = np.exp(opt_result.x[0])
    alpha = opt_result.x[1]
    mse = np.sqrt(np.mean(2 * opt_result.cost)) # least_squares has 0.5 in objective

    return K, alpha, mse
{% endhighlight %}

## Nonlinear Least Squares
In the previous section we conventiently avoided solving the actual problem we wanted to solve

$$\underset{K,\, \alpha}{\text{minimize}}\quad \sum_{i=1}^{|V|} \left(f_i - Kr_i^\alpha\right)^2$$

by noticing we could solve a transformed version of the problem instead. The difficulty with this original formulation is that it a non-convex function of $$K$$ and $$\alpha$$. The consequence of this is of course that there are potentially many local extrema in addition to the global minimum. Figure 2 shows a plot of the objective function $$J(K,\alpha)$$ and its non-convexity

<figure class="half">
    <a href="/assets/images/shakespeare-zipf-param-contours.png"><img src="/assets/images/shakespeare-zipf-param-contours.png"></a>
    <a href="/assets/images/shakespeare-zipf-param-surface-best-angle.png"><img src="/assets/images/shakespeare-zipf-param-surface-best-angle.png"></a>
    <figcaption>Figure 2</figcaption>
</figure>

One approach to solving this problem is to set the gradient to 0 and find all the solutions of the system of equations. We could then find the globabl optimum by plugging all solutions into the objective to find the one that yields the lowest value. For our particular problem the system of equations is

$$
\nabla J =
\begin{bmatrix}
\frac{\partial J}{\partial K}\\
\frac{\partial J}{\partial \alpha}\\
\end{bmatrix} =
\begin{bmatrix}
-2\sum_{i=1}^{|V|} (f_i - Kr_i^\alpha)r_i^\alpha\\
-2\sum_{i=1}^{|V|} (f_i - Kr_i^\alpha)K(\log r_i) r_i^\alpha\\
\end{bmatrix} = 
\begin{bmatrix}
0\\
0\\
\end{bmatrix} 
$$

Far from the simplicity of the normal equations, the above is a nonlinear system of equations, which is hardly more tractable than the original optimisation problem. 

Rather than trying to solve the nonlinear system of equations resulting from the stationarity condition of the gradient, most solvers intstead implement the _Levenberg-Marquardt_ algorithm. In fact, this is what's implemented by the `least_squares` function inside of `scipy.optimize`. In order to understand this method, it's easiest to first understand a slight simplification of the algorithm.

### Gauss-Newton Method
Both the OLS and NNLS problem can be written in the form

$$\underset{x}{\text{minimize}}\quad ||r(x)||^2$$

The only difference is that in OLS the residual function $$r: \mathbf{R}^n \rightarrow \mathbf{R}^{m}$$, is a linear function of $$x$$. The Gauss-Newton algorithm solves the case when $$r(x)$$ is nonlinear by reducing the problem to a series of OLS problems. 

The core idea behind the algorithm is simple, since OLS problems are solvable, we can first linearise $$r(x)$$, solve the resulting OLS problem and use the solution as the next point around which to linearise $$r(x)$$. We can repeatedly iterate linearising around a point and finding a new point to linearise around until convergence.


### Python Implementation

## Conclusion


## References
1. [Zipf's Law Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law)
2. [Gauss-Newton Algorithm Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)
3. [Stanford's Intro to Linear Dynamical Systems]()

## Footnotes
<a name="footnote1">1</a>: Since the ranks and frequencies are all positive, the logarithm is well defined.
