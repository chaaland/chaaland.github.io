---
title: "Fixed Point vs Floating Point: A Numerical Representation Deep Dive"
categories:
  - Algorithms
date: 2026-03-01 19:00:00 +0000
mathjax: true
tags:
  - Fixed-point arithmetic
  - Floating point
  - Numerical methods
  - Numerical computing
toc: true
classes: wide
excerpt: "Fixed point arithmetic is fast, predictable, and deterministic — so why does floating point dominate? A ground-up comparison of how computers represent real numbers."
---

Most computer programming languages have two classes of numeric types: integers for whole number arithmetic and floating point numbers for approximating real numbers.
This post introduces an alternative representation of the real numbers using fixed-point along with the advantages and disadvantages compared to floating point precision.

## Review of floating point

Floating point numbers are the ones most computer programmers are familiar with since they're broadly supported in both hardware and software.

As a reminder, floating point numbers are very similar to scientific notation with base 2 instead of base 10.
Roughly, you should think of a floating point number representing something like this

$$
(-1)^{s}\times m \times 2^e
$$

It consists of

- a single sign bit $$s$$ which determines whether the number is positive or negative
- mantissa bits $$m$$ which determine the number's precision
- exponent bits $$e$$ which determine the dynamic range of numbers that can be represented

Recall that in scientific notation,

- there must be only one digit before the decimal
- the digit before the decimal must be nonzero

For example, the number $$6.022 \times 10^{23}$$ is correctly in scientific notation while $$0.6022\times 10^{24}$$ and $$602.2\times 10^{21}$$ are not.

Since $$m$$ is represented as a binary number, it must _always_ begin with a 1 before the decimal.
For this reason, in floating point, the mantissa only contains the numbers to the right of the decimal and the leading one is implicit.[^denormal]
So a slightly more accurate depiction of floating point is the following representation

$$
(-1)^{s}\times 1.m \times 2^e.
$$

The exponent is also stored as an unsigned integer in binary.
If the exponent were 4 bits, it could be any integer between 0 and 15 ($$2^4-1$$) for example.
But if we were to do this, we would be left unable to have negative exponents to represent small fractional numbers.

To solve this, the exponent is instead interpreted as a _biased_ unsigned integer.[^twoscomp]
Returning to the 4 bit exponent example, rather than representing integers between 0 and 15, we can add a bias of -8 to have it represent integers between -8 and 7.

So a floating point number actually looks a lot more like

$$
(-1)^{s} \times 1.m\times 2^{e - b}.
$$

Floating point numbers also have special bit patterns to represent positive and negative infinity, as well as NaNs. Another quirk of floating point is there is both a positive and negative zero.

### Multiplication

Consider multiplying two floating point numbers

$$
(-1)^{\text{sign}_1} \times 1.{\text{mantissa}_1}\times 2^{\text{exp}_1 - \text{bias}}\cdot (-1)^{\text{sign}_2} \times 1.{\text{mantissa}_2}\times 2^{\text{exp}_2 - \text{bias}}
$$

Rearranging we have

$$
(-1)^{\text{sign}_1 + \text{sign}_2} \times (1.{\text{mantissa}_1} \times 1.{\text{mantissa}_2})\times 2^{\text{exp}_1 + \text{exp}_2 - 2\cdot\text{bias}}
$$

We can see that to write our new number in floating point format
- the new sign bit must be the XOR of $$\text{sign}_1$$ and $$\text{sign}_2$$
- the mantissa bits need to be elementwise multiplied
- the exponent bits need to be added together along with an additional bias (to keep the exponent of the form $$\text{exp} - \text{bias}$$). We may also need to increment the exponent if the multiplication of the mantissa bits overflows and leads to a carry.

### Addition

Now consider the addition of two floating numbers

$$
(-1)^{\text{sign}_1} \times 1.{\text{mantissa}_1}\times 2^{\text{exp}_1 - \text{bias}} + (-1)^{\text{sign}_2} \times 1.{\text{mantissa}_2}\times 2^{\text{exp}_2 - \text{bias}}
$$


## Fixed-point

### Addition
### Multiplication

## Summary

[^twoscomp]: Two's complement is the natural alternative, but has an undesirable property: the most significant bit flips from 0 to 1 crossing zero, so larger bit patterns no longer mean larger values. Biasing preserves the unsigned ordering, keeping larger bit patterns as larger exponents.

[^denormal]: This holds for normalized floating point numbers. An exception exists for _denormal_ (or _subnormal_) numbers, which have a special all-zero exponent pattern and no implicit leading 1. Denormals allow the representation of values closer to zero than normalized numbers can reach, at the cost of reduced precision.
