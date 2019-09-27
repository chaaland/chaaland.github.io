---
title: "Nonlinear Least Squares"
categories:
  - Mathematics
date:   2019-09-18 22:12:45 +0100
mathjax: true
tags:
  - Optimization
  - Nonlinear modeling  
  - Gauss-Newton
  - Least Squares
  - Regression
toc: true
classes: wide
excerpt: ""
header: 
  overlay_image: assets/images/shakespeare-zipf-param-surface copy.png
  overlay_filter: 0.2
---

Fitting linear models using least squares is so ubiquitous you would be hard pressed to find a field in which it has not found application. A large part of the reason ordinary least squares (OLS) is so prevalent is that many (most?) simply aren't familiar with non-linear methods. Historically, solving nonlinear least squares (NNLS) problems was computationally expensive, but with modern computing power the barrier is less computation and moreso people's familarity with the methods. As we'll see in the following post, solving NNLS problems is just as simple as OLS.

## Zipf's Law
Before moving on to the main focus of this post, it helps to have a concrete problem to motivate the material that follows. Consider modeling the frequency of a word's appearance in a corpus with vocabulary $$V$$, against its rank (by frequency). We start by downloading the text of Shakespeare's _Hamlet_ using python's `requests` library to get the raw html, then parsing out the raw text using `BeautifulSoup` as shown in the following script 

{% highlight python %}
import os
import re
import requests
from bs4 import BeautifulSoup
pjoin = os.path.join


def download_text(url: str):
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

With this, we can easily write a function computing the frequencies of each word in the text using a specialised `dict` contained in python's `collections` module

{% highlight python %}
import os
import re
import numpy as np
from collections import Counter
pjoin = os.path.join


def word_freqs(topk: int = 25):
    word_counts = Counter() 
    with open(pjoin("..", "txt", "hamlet.txt"), "rt") as f:
        text = " ".join(f.readlines())
        text = re.sub("[^0-9a-zA-Z ]+", "", text)
        all_words = [word.lower() for word in text.split(" ")]
        word_counts = Counter(all_words)

    n_words = len(all_words)
    words, counts = zip(*word_counts.most_common(topk))
    freqs = np.asarray([c / n_words for c in counts])
    return words, freqs
{% endhighlight %}

We can now use this data to get a plot of top 100 word's frequencies. A quick glance at the plot below should give a good indication that a _power law_ of the form 

$$ f(r) = Kr^\alpha $$

where $$f :\{1,2, \ldots, \lvert V \rvert\} \rightarrow \mathbf{R}$$, could be a reasonable fit to this data

<figure class="third">
    <img src="">
    <a href="/assets/images/shakespeare-freq-scatter.png"><img src="/assets/images/shakespeare-freq-scatter.png"></a>
    <img src="">
</figure>

As it turns out, this observation is not a specific characteristic _Hamlet_, but is one instance of a broader phenomenon known as _Zipf's Law_. From Wikipedia,
> Zipf's law is an empirical law formulated using mathematical statistics that refers to the fact that many types of data studied in the physical and social sciences can be approximated with a Zipfian distribution, one of a family of related discrete power law probability distributions

Now that we've established a parametric form for the model along with 100 data points, what remains is to find a suitable $$K$$ and $$\alpha$$ that best fit the data.

## Ordinary Least Squares
If you've taken any linear algebra, you'll most likely recall the least squares optimisation problem

$$\underset{x}{\text{minimize}} \quad ||Ax - y||^2$$

where $$y\in \mathbf{R}^{m}$$ and $$A\in \mathbf{R}^{m\times n}$$ with $$m \ge n$$ and having _linearly independent_ columns. The vector inside the Euclidean norm is often called the _residual vector_ and denoted $$r \in \mathbf{R}^m$$. Ordinary least squares problems are ideal for a number of reasons but the most salient are 
- objective is differentiable
- objective is _convex_ 
- closed form solution given by the _normal equations_

$$A^TAx^{\star} = A^T y$$


## Derivation of the Gauss-Newton Method

## Python Implementation

## Conclusion

### References
1. [Zipf's Law Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law)
2. [Gauss-Newton Algorithm Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)
3. [Stanford's Intro to Linear Dynamical Systems]()