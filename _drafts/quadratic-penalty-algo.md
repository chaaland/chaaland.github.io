---
title: "Quadratic Penalty Algo"
categories:
  - Mathematics
date:   2019-10-20 22:12:45 +0100
mathjax: true
toc: true
# classes: wide
excerpt: ""
header: 
  overlay_image: assets/images/shakespeare-zipf-param-surface-splash.png
  overlay_filter: 0.2
tags:
  - Optimisation
---

# Constrained Optimisation
# Softmax classifier
maximize probability of joint probability subject to contraint the probabilities add to 1

# Dominant Eigenvalue
Maximize ||Ax||^2 subject to ||x|| = 1

# Quadratic Penalty Method
Fitting linear models using least squares is so ubiquitous you would be hard pressed to find a field in which it has not found application. A large part of the reason ordinary least squares (OLS) is so prevalent is that many simply aren't familiar with nonlinear methods. 

Historically, solving nonlinear least squares (NLLS) problems was computationally expensive, but with modern computing power the barrier is less computation and moreso people's familarity with the methods. As we will see, solving NLLS problems is just as simple as OLS.
You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out this math 

$$e^{\pi i} = -1$$


Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].


[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/