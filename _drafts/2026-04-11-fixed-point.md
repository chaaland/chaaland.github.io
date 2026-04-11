---
title: "Fixed Point vs Floating Point: A Numerical Representation Deep Dive"
categories:
  - Algorithms
date: 2026-04-11 08:00:00 +0000
mathjax: true
tags:
  - Fixed-point arithmetic
  - Floating point
  - Numerical methods
  - Numerical computing
toc: true
classes: wide
excerpt: "A ground-up look at how floating-point and fixed-point numbers are represented and how arithmetic works for each"
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

It's of course a bit more complicated than this because of special cases like positive/negative infinity, NaNs, and subnormal numbers but this is a good enough mental model for "ordinary" floating point numbers.

### Multiplication

Consider multiplying two floating point numbers

$$
(-1)^{s_1} \times 1.{m_1}\times 2^{e_1 - b}\cdot (-1)^{s_2} \times 1.{m_2}\times 2^{e_2 - b}
$$

Rearranging we have

$$
(-1)^{s_1 + s_2} \times (1.m_1 \times 1.m_2)\times 2^{(e_1 + e_2 - b) - b}
$$

We can see that to write our new number in floating point format

- the new sign bit must be the XOR of $$s_1$$ and $$s_2$$[^signxor]
- the mantissa bits need to be multiplied
- the exponent bits need to be added together with the bias subtracted (to keep the exponent of the form $$e - b$$)

One extra subtlety is when the multiplication of the mantissae overflows.
Since $$1.m_1$$ and $$1.m_2$$ are between 1 and 2, their product is between 1 and 4.
In the event $$1.m_1 \times 1.m_2 \ge 2$$, the mantissa needs to be bit shifted right and compensating by incrementing the exponent (i.e., multiplying the overall value by 2 via the exponent)

### Addition

Now consider the addition of two floating-point numbers

$$
(-1)^{s_1} \times 1.{m_1}\times 2^{e_1 - b} + (-1)^{s_2} \times 1.{m_2}\times 2^{e_2 - b}.
$$

In order to add these numbers, they first need to have the same exponent (by matching the larger of the two exponents).
Directly increasing the exponent from $$e_1$$ to $$e_2$$ would multiply the represented value by $$2^{e_2-e_1}$$, so we simultaneously right-shift the mantissa by $$e_2 - e_1$$ bits to divide by $$2^{e_2-e_1}$$, keeping the value unchanged.

Once the numbers have a common exponent, the mantissae are added or subtracted depending on the sign bits.
If the new mantissa overflows (i.e. $$\ge 2$$) then it needs to be bit shifted right and the exponent incremented by 1.
In the case of subtraction, it's possible that the mantissa underflows (i.e. $$< 1$$) in which case it needs to be bit shifted left until the first one appears and the exponent decremented by the same amount as the shift.

## Fixed-point

One major drawback of floating-point format is that many numbers in base 10 are not exactly representable (e.g. 0.3).
In applications like finance or accounting, it's crucial to have these fractional numbers represented accurately.
In these applications we're also less concerned about representing both astronomically large and microscopically small numbers as we might be in other domains.
Similarly, the need for infinities and NaNs are significantly less important.

An alternative way for computers to represent numbers is fixed-point decimals.
In contrast to floating point, a fixed-point decimal is essentially just an integer plus some metadata about where to place the decimal point.
There's no exponent, bias, or implicit leading 1 digit.
There are a few different ways to implement fixed-point but one of the simplest ways is as the dataclass below

```python
from dataclasses import dataclass

@dataclass
class DecimalFixedPoint:
    value: int    # the integer mantissa
    scale: int    # number of decimal places
```

which is interpreted as the number

$$\text{value}\times 10^{-\text{scale}}.$$

For example, `DecimalFixedPoint(value=12345, scale=2)` represents the number $$123.45$$.[^int128][^sql]
For other applications in scientific computing and numerical methods, a base 2 fixed-point is used which takes the form

$$\text{value}\times 2^{-\text{scale}}.$$

These have the nice property that common operations like multiplying or dividing by 2 correspond to simple bit shifts but they have the same drawback that numbers like 0.3 are not exactly representable.

### Addition

In order to add two fixed-point numbers, they must have the same scale.
This is very similar to how adding two floating point numbers requires them to have the same exponent!

Suppose you want to add $$3.1415$$ and $$0.577$$.
In fixed-point this would be $$31415 \times 10^{-4} + 577 \times 10^{-3}$$.
We first need to align the scales to the _maximum_ of each of the numbers (4 in this case) which becomes $$31415 \times 10^{-4} + 5770 \times 10^{-4}$$.
Now the numbers are simply added in the usual way

$$
\begin{array}{r|c|c|c||c|c|c|c|}
 & 0 & 0 & 3 & 1 & 4 & 1 & 5 \\
\hline
+ & 0 & 0 & 0 & 5 & 7 & 7 & 0 \\
\hline
= & 0 & 0 & 3 & 7 & 1 & 8 & 5 \\
\hline
\end{array}
$$

and the result is $$37185 \times 10^{-4} = 3.7185$$.
In actual fact, the integer value component is represented in base 2 rather than base 10 as depicted above but the logic is the same.
So adding or subtracting fixed-point decimals becomes a simple matter of regular integer arithmetic and incrementing scales.

### Multiplication

The multiplication of two fixed-point decimals is similarly straightforward but there are some design choices.
From the fixed-point representation outlined above, the product of two fixed-point numbers is

$$
\text{value}_1 \times 10^{-\text{scale}_1} \cdot \text{value}_2 \times 10^{-\text{scale}_2} = (\text{value}_1\cdot \text{value}_2 )\times 10^{-(\text{scale}_1 +\text{scale}_2)}
$$

which might make it seem obvious that the scale of the resulting fixed-point number should be the sum of the scales.
This does work in theory but in practice, you end up with "exploding scales" when multiplying even just a handful of fixed-point numbers. It also makes it harder to reason about what scale your result will be since addition will be the maximum of the two scales and multiplication is the sum. This can lead to lots of extra casts to control the scale exactly.

Instead, libraries like `polars` implement a `Decimal` type using a scale that doesn't "float".
Multiplying two numbers of the same scale results in a number of the _same_ scale rather than twice the scale.
If the numbers have different scales, the result is a number with the _maximum_ of the two scales, just like addition.
This is accomplished by multiplying the values and then downscaling to the desired scale (dividing by the appropriate power of 10) and rounding.

## Summary

Though floating-point formats are the de-facto standard for numerical computing, fixed-point remains important in domains where its trade-offs are more favorable.

The key trade-off is **dynamic range vs. exactness**.
Floating-point achieves its range by letting precision vary with magnitude which is a useful property for scientific computing where quantities can span many orders of magnitude.
Fixed-point sacrifices that range for uniform, predictable precision, which is what finance and accounting applications require: a rounding error of $$10^{-4}$$ is equally unacceptable on a \$1.00 transaction as on a \$1,000,000.00 one.

[^twoscomp]: Two's complement is the natural alternative, but has an undesirable property: the most significant bit flips from 0 to 1 crossing zero, so larger bit patterns no longer mean larger values. Biasing preserves the unsigned ordering, keeping larger bit patterns as larger exponents.

[^denormal]: This holds for normalized floating point numbers. An exception exists for _denormal_ (or _subnormal_) numbers, which have a special all-zero exponent pattern and no implicit leading 1. Denormals allow the representation of values closer to zero than normalized numbers can reach, at the cost of reduced precision.

[^signxor]: The exponent $$s_1 + s_2$$ from $$(-1)^{s_1+s_2}$$ only needs to be tracked modulo 2, since $$(-1)^0 = (-1)^2 = 1$$ and $$(-1)^1 = (-1)^3 = -1$$. Reduction mod 2 is exactly XOR: $$1 \oplus 1 = 0$$, $$0 \oplus 0 = 0$$, and $$0 \oplus 1 = 1 \oplus 0 = 1$$.

[^sql]: SQL's `DECIMAL(p, s)` and `NUMERIC(p, s)` types follow the same model: `p` is the total number of significant digits (precision) and `s` is the scale.

[^int128]: In practice, `value` is typically a wide integer such as `Int128`. Multiplication doubles the bit-width of the result before any rescaling, so two 64-bit values produce a 128-bit intermediate product. A 64-bit `value` would overflow before that rescaling step, losing significant bits.
