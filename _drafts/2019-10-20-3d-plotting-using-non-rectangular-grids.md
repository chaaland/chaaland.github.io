---
title: "Better Contour and Surface Plots Using Non-Rectangular Sampling Grids"
categories:
  - Mathematics
date:   2019-09-29 13:32:45 +0100
mathjax: true
tags:
  - matplotlib
  - python
toc: true
toc_label: 
# classes: wide
excerpt: ""
header: 
  overlay_image: assets/images/shakespeare-zipf-param-surface-splash.png
  overlay_filter: 0.2
---

TODO:
  - Change first plots to be consistent with coarser plot shown at the end
  - Add plot at the end including the scatter grid
  - Some examples. Gaussian, sinc function

# The Problem with Rectangular Grids
The standard way of creating contour and surface plots of a function $$f:\mathbf{R}^2 \rightarrow \mathbf{R}$$ is first creating a rectangular grid of $$(x,y)$$ coordinates (using something like `meshgrid`), then evaluating the function $$f$$ elementwise at each $$(x,y)$$ pair. For the most part this is fine, but there are some situations where it can really distort your view of the surface's true shape. 

Consider graphing the following quadratic form

$$f(x,y) = 
\begin{bmatrix}
x\\
y
\end{bmatrix}^T
\begin{bmatrix}
1 & 1\\
1 & 4\\
\end{bmatrix}
\begin{bmatrix}
x\\
y
\end{bmatrix}
$$

Plotting this on a rectangular grid we get a 3D plot like left image in Figure 1. From the definition of the function, we know the graph should be an upward facing paraboloid. Looking at the figure however, it is not obvious that this is what is plotted. What we would like to see is something more like the image on the right. The rest of this post will explain how this surface is plotted and propose some other cases it might be useful.

<figure class="half">
    <a href="/assets/images/grid-sampling-rectangular-paraboloid.png"><img src="/assets/images/grid-sampling-rectangular-paraboloid.png"></a>
    <a href="/assets/images/grid-sampling-elliptic-paraboloid.png"><img src="/assets/images/grid-sampling-elliptic-paraboloid.png"></a>
    <figcaption>Figure 1</figcaption>
</figure>

# Radially Symmetric Meshes
The traditional `meshgrid` function is only suitable for generating rectangular grids. Given a range of $$x$$ values $$(x_1,\ldots,x_n)$$ and $$y$$ values $$(y_1,\ldots,y_m)$$, two matrices $$X\in\mathbf{R}^{m\times n}$$ and $$Y\in \mathbf{R}^{m\times n}$$ are returned:
$$
X = 
\begin{bmatrix}
x_1 & x_2 & \ldots & x_n\\
x_1 & x_2 & \ldots & x_n\\
\vdots & \vdots & \ddots & \vdots\\
x_1 & x_2 & \ldots & x_n\\
\end{bmatrix},\quad
Y = 
\begin{bmatrix}
y_1 & y_1 & \ldots & y_1\\
y_2 & y_2 & \ldots & y_2\\
\vdots & \vdots & \ddots & \vdots\\
y_m & y_m & \ldots & y_m\\
\end{bmatrix}
$$

Effectively each $$X_{ij}$$ is paired with the corresponding $$Y_{ij}$$ forming the cartesian product of $$x$$ and $$y$$. This rectangular grid is useful but there are many functions that have a radial symmetry. Rather than even spacing along $$x$$ or $$y$$, a more natural sampling would be even angular and radial spacing. The polar coordinate transform will allow uniform sampling in $$(r,\theta)$$ space

$$
\begin{align*}
x &= r \cos (\theta)\\
y &= r \sin (\theta)\\
\end{align*}
$$

Figure 2 shows how using this polar transform we can get from a rectangular grid in $$(r,\theta)$$ space to a radially symmetric grid in $$(x,y)$$.

<figure class="half">
    <a href="/assets/images/grid-sampling-rectangular-horizontal.png"><img src="/assets/images/grid-sampling-rectangular-horizontal.png"></a>
    <a href="/assets/images/grid-sampling-rectangular-vertical.png"><img src="/assets/images/grid-sampling-rectangular-vertical.png"></a>
</figure>
<figure class="half">
    <a href="/assets/images/grid-sampling-circular-grid-radial.png"><img src="/assets/images/grid-sampling-circular-grid-radial.png"></a>
    <a href="/assets/images/grid-sampling-circular-grid-angular.png"><img src="/assets/images/grid-sampling-circular-grid-angular.png"></a>
    <figcaption>Figure 2</figcaption>
</figure>

To understand how the radially symmetric sampling image is generated, we start first with a rectangular grid. Using `numpy`'s `meshgrid` function we get a grid like the ones in the first row of Figure 2. Rather than use these as $$(x,y)$$ coordinates directly, we can instead treat them as sampling $$(r,\theta)$$ and then transform them to polar as in the second row of Figure 2. As shown in red, a horizontal line in $$(r,\theta)$$ space maps to a radius of fixed angle in $$(x,y)$$ space. Similarly, in green we see a vertical line is mapped to a circle of fixed radius. This sampling schmee is implemented below

{% highlight python %}
import numpy as np

def generate_radial_grid(r_low, r_high, n_r, n_theta):
    r = np.linspace(r_low, r_high, n_r)
    theta = np.linspace(0, 2*np.pi, n_theta)
    r_mesh, theta_mesh = np.meshgrid(r, theta)
    x_mesh = r_mesh * np.cos(theta_mesh)
    y_mesh = r_mesh * np.sin(theta_mesh)

    return x_mesh, y_mesh
{% endhighlight %}


# Elliptic Grid Sampling
Sampling a radially symmetric grid is useful when the function is isotropic. However, in the more general case we would like to sample concentric ellipses of points. First, recall the parametric equations for an axis alligned ellipse centered at the origin are

$$
\begin{bmatrix}
x(\theta) \\
y(\theta)\\
\end{bmatrix} =
\begin{bmatrix}
a & 0\\
0 & b\\
\end{bmatrix}
\begin{bmatrix}
\cos(\theta)\\
\sin(\theta)\\
\end{bmatrix}
$$

This case is very similar to the circularly symmetric case and requires a tiny change to the code. Using the same grid as in the previous figure we see the vertical lines are mapped to concentric ellipses

<figure class="half">
    <a href="/assets/images/grid-sampling-centered-ellipse-radial.png"><img src="/assets/images/grid-sampling-centered-ellipse-radial.png"></a>
    <a href="/assets/images/grid-sampling-centered-ellipse-angular.png"><img src="/assets/images/grid-sampling-centered-ellipse-angular.png"></a>
    <figcaption>Figure 3</figcaption>
</figure>

{% highlight python %}
import numpy as np

def generate_axis_aligned_ellipse_grid(r_low, r_high, n_r, n_theta, a, b):
    r = np.linspace(r_low, r_high, n_r)
    theta = np.linspace(0, 2*np.pi, n_theta)
    r_mesh, theta_mesh = np.meshgrid(r, theta)
    x_mesh = a * r_mesh * np.cos(theta_mesh)
    y_mesh = b * r_mesh * np.sin(theta_mesh)

    return x_mesh, y_mesh
{% endhighlight %}

But to handle the more general case of ellipse (rotated axes) we must recall from linear algebra the _rotation matrix_. The rotation matrix<sup>[1](#footnote1)</sup> for rotating by an angle $$\alpha$$ anti-clockwise measured from the positive $$x$$-axis is given by

$$
R_\alpha = 
\begin{bmatrix}
\cos(\alpha) & -\sin(\alpha)\\
\sin(\alpha) & \cos(\alpha)\\
\end{bmatrix}
$$

Generating a grid points on a rotated ellipse is now just as simple as generating points for an axis aligned ellipse, then applying the rotation matrix. The parametric equations for this more general case is then

$$
\begin{bmatrix}
x(\theta) \\
y(\theta)\\
\end{bmatrix} =
\begin{bmatrix}
\cos(\alpha) & -\sin(\alpha)\\
\sin(\alpha) & \cos(\alpha)\\
\end{bmatrix}
\begin{bmatrix}
a & 0\\
0 & b\\
\end{bmatrix}
\begin{bmatrix}
\cos(\theta)\\
\sin(\theta)\\
\end{bmatrix}
$$

<figure class="half">
    <a href="/assets/images/grid-sampling-ellipse-rotated-radial.png"><img src="/assets/images/grid-sampling-ellipse-rotated-radial.png"></a>
    <a href="/assets/images/grid-sampling-ellipse-rotated-angular.png"><img src="/assets/images/grid-sampling-ellipse-rotated-angular.png"></a>
    <figcaption>Figure 4</figcaption>
</figure>

And lastly, it is often the case that the surface you want to plot is not centered at the origin in which case we can simply translate the entire sampling grid to the appropriate center coordinates. Adapting the parametric equations for this is as simple as adding a vector of the coordinates of the desired ellipse center

$$
\begin{bmatrix}
x(\theta) \\
y(\theta)\\
\end{bmatrix} =
\begin{bmatrix}
x_c\\
y_c\\
\end{bmatrix} + 
\begin{bmatrix}
\cos(\alpha) & -\sin(\alpha)\\
\sin(\alpha) & \cos(\alpha)\\
\end{bmatrix}
\begin{bmatrix}
a & 0\\
0 & b\\
\end{bmatrix}
\begin{bmatrix}
\cos(\theta)\\
\sin(\theta)\\
\end{bmatrix}
$$

<figure class="half">
    <a href="/assets/images/grid-sampling-ellipse-offset-radial.png"><img src="/assets/images/grid-sampling-ellipse-offset-radial.png"></a>
    <a href="/assets/images/grid-sampling-ellipse-offset-angular.png"><img src="/assets/images/grid-sampling-ellipse-offset-angular.png"></a>
    <figcaption>Figure 5</figcaption>
</figure>

{% highlight python %}
import numpy as np

def generate_ellipse_grid(
  r_low, 
  r_high, 
  n_r, 
  n_theta = 50, 
  a = 1, 
  b = 1, 
  alpha=0, 
  center=np.zeros((2,1)),
):
    r = np.linspace(r_low, r_high, n_r)
    theta = np.linspace(0, 2*np.pi, n_theta)
    r_mesh, theta_mesh = np.meshgrid(r, theta)
    x_mesh = a * r_mesh * np.cos(theta_mesh).reshape((-1,1))
    y_mesh = b * r_mesh * np.sin(theta_mesh).reshape((-1,1))

    xy_stacked = np.hstack([x_mesh, y_mesh]).T 

    rot_mat = np.asarray([
      [np.cos(angle), -np.sin(angle)],
      [np.sin(angle), np.cos(angle)],
    ])
    plot_grid = rot_mat @ xy_stacked + center.reshape((2,1))
    X = plot_grid[0,:].reshape(x_mesh.shape)
    Y = plot_grid[1,:].reshape(x_mesh.shape)

    return X, Y
{% endhighlight %}

# Conclusion


# References
1. [Zipf's Law Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law)
2. [Gauss-Newton Algorithm Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)
3. [Stanford's Intro to Linear Dynamical Systems]()

## Footnotes
<a name="footnote1">1</a>: As with any linear transformation, you can derive this matrix by concatenating the result of rotating each of the standard basis vectors.
