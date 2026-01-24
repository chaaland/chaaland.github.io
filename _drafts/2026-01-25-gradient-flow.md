---
title: "Gradient Flow and Gradient Descent"
categories:
  - Optimization
date: 2026-01-09 19:00:00 +0000
mathjax: true
tags:
  - Optimization
  - Gradient Descent
  - Machine Learning
toc: true
classes: wide
excerpt: "Understand the effect of sharpness on gradient descent dynamics."
---

This post explores the relationship between gradient flow (the continuous-time limit of gradient descent) and gradient descent, with a focus on how the learning rate and loss landscape curvature determine convergence behavior.

## Motivation

Gradient descent is the workhorse of modern machine learning optimization. Given a differentiable loss function $$f: \mathbb{R}^d \to \mathbb{R}$$, gradient descent iteratively updates parameters according to

$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

where $$\eta > 0$$ is the learning rate (or step size). But how should we choose $$\eta$$? Too small and convergence is slow; too large and the iterates may diverge.

To understand this, it helps to study **gradient flow**—the continuous-time limit of gradient descent obtained as $$\eta \to 0$$:

$$\frac{dx}{dt} = -\nabla f(x)$$

This ODE describes a particle flowing downhill on the loss surface at a rate proportional to the gradient magnitude.

## The 1D Quadratic Case

Consider the simplest case: minimizing a 1D quadratic $$f(x) = \frac{S}{2}x^2$$ where $$S > 0$$ is the **sharpness** (curvature) of the function. Larger $$S$$ means a steeper, narrower parabola.

{% include plotly_figure.html
   src="/assets/2026/gradient-flow/plots/sharpness_1d.html"
   height="500px"
   caption="Quadratic functions with different sharpness values S. Larger S means steeper curvature."
%}

The gradient is $$\nabla f(x) = Sx$$, so gradient descent becomes

$$x_{k+1} = x_k - \eta \cdot S x_k = (1 - S\eta) x_k$$

This is a simple geometric sequence! Starting from $$x_0$$, we have $$x_k = (1-S\eta)^k x_0$$.

For convergence to zero, we need $$|1 - S\eta| < 1$$. Since $$S, \eta > 0$$, this means:

$$0 < S\eta < 2 \quad \Rightarrow \quad \eta < \frac{2}{S}$$

This is a fundamental result: **the maximum stable learning rate is inversely proportional to the sharpness**.

## Gradient Flow Solution

For the quadratic $$f(x) = \frac{S}{2}x^2$$, gradient flow gives the ODE

$$\frac{dx}{dt} = -Sx$$

with solution $$x(t) = x_0 e^{-St}$$. This exponentially decays to zero for any positive sharpness—gradient flow never diverges on a quadratic!

The loss along the trajectory is

$$f(x(t)) = \frac{S}{2}x_0^2 e^{-2St}$$

which shows exponential decay at rate $$2S$$. Sharper functions have faster gradient flow convergence.

## The 2D Quadratic Case

In higher dimensions, the picture becomes richer. Consider a 2D quadratic

$$f(x) = \frac{1}{2}x^T A x$$

where $$A$$ is a symmetric positive definite matrix. The eigenvalues of $$A$$ determine the curvature along the principal axes.

For an elliptical paraboloid with semi-axes $$a$$ and $$b$$, rotated by angle $$\theta$$, the level sets form ellipses. The matrix $$A$$ has eigenvalues $$\lambda_1 = 1/a^2$$ and $$\lambda_2 = 1/b^2$$.

The gradient flow solution is

$$x(t) = e^{-At}x_0$$

which traces a smooth curve toward the origin. In contrast, gradient descent with a fixed learning rate takes discrete steps that may overshoot along the high-curvature directions.

{% include plotly_figure.html
   src="/assets/2026/gradient-flow/plots/contours_simple.html"
   height="700px"
   caption="Contours of a 2D quadratic with gradient flow (blue) and gradient descent (red) trajectories."
%}

## Loss vs Step: The Effect of Learning Rate

The learning rate $$\eta$$ dramatically affects convergence behavior. The critical threshold is

$$\eta_{\text{crit}} = \frac{2}{\lambda_{\max}}$$

where $$\lambda_{\max}$$ is the largest eigenvalue of $$2A$$ (i.e., twice the sharpness along the steepest direction).

- For $$\eta < \eta_{\text{crit}}$$: Loss monotonically decreases
- For $$\eta > \eta_{\text{crit}}$$: Iterates diverge to infinity

{% include plotly_figure.html
   src="/assets/2026/gradient-flow/plots/loss_vs_step.html"
   height="550px"
   caption="Loss vs optimization step for different learning rates. Rates above the critical threshold cause divergence."
%}

## Interactive Exploration

The following interactive visualization lets you explore how the ellipse parameters ($$a$$, $$b$$, $$\theta$$) and learning rate ($$\eta$$) affect gradient descent behavior. The blue curve shows gradient flow (the continuous limit), while red points show gradient descent iterates.

{% include plotly_figure.html
   src="/assets/2026/gradient-flow/plots/quadratic_interactive.html"
   height="750px"
   caption="Interactive visualization of gradient flow and gradient descent. Adjust the sliders to explore different configurations."
%}

Key observations to make:
- When $$a = b$$, the contours are circular and gradient descent moves directly toward the origin
- When $$a \neq b$$ (elliptical contours), gradient descent can exhibit "zigzag" behavior
- Larger $$\eta$$ causes more aggressive steps that may overshoot the minimum
- The rotation angle $$\theta$$ changes the principal axes but not the convergence rate

## Connection to Sharpness in Deep Learning

In deep learning, the "sharpness" of the loss landscape—measured by the largest eigenvalue of the Hessian—plays a crucial role in:

1. **Determining maximum learning rate**: The stability bound $$\eta < 2/S$$ applies locally
2. **Generalization**: Flatter minima are often associated with better generalization
3. **Adaptive methods**: Algorithms like Adam implicitly adapt to local curvature

Recent research has shown that during training, neural networks often operate near the "edge of stability" where $$\eta \approx 2/S$$, with the sharpness dynamically adjusting to maintain this balance.

## Conclusion

Gradient flow provides a clean theoretical framework for understanding gradient descent:

- The continuous limit removes discretization artifacts
- Stability analysis reveals the critical role of sharpness
- The maximum learning rate is $$\eta_{\max} = 2/\lambda_{\max}$$

For quadratic functions, gradient flow always converges while gradient descent requires careful step size selection. In the non-quadratic case, these insights apply locally near critical points, explaining why adaptive learning rate methods are so effective in practice.

## References

1. [Gradient Flow](https://en.wikipedia.org/wiki/Gradient_descent#Gradient_flow)
2. Cohen, J., et al. "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability." ICLR 2021.
3. Boyd, S. & Vandenberghe, L. "Convex Optimization." Cambridge University Press, 2004.
