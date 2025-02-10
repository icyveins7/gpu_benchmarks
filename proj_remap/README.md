# Bilinear Interpolation Remapping

This subproject deals with the task of bilinear interpolation (also commonly known as _bilerp_). Here we design a custom kernel to perform this,
and extend it to a special scenario of wrap-around.

# Motivation

The bilinear interpolation function provided by NPP appears to be flawed. Currently, it does not handle the edge cases properly, and instead extends
the valid range in a non-symmetric way: half a pixel on the top and left edge, and more on the right and bottom edge.

This is clearly not intended, as IPP (the CPU library by Intel) performs the range checks correctly. See [my other library for the IPP test](https://github.com/icyveins7/ipp_ext).

# Background

Bilinear interpolation takes in 4 corner pixels and interpolates them, and is akin to using a 3-step linear interpolation in succession.

We assume $y$ increases in the downwards direction, as is common with image processing computations. Without loss of generality, take the 4 corner pixels to be
$A = (x_0, y_0)$, $B = (x_0, y_1)$, $C = (x_1, y_0)$, and $D = (x_1, y_1)$. The requested pixel is defined as $z = (x, y)$. These are arranged in the following way, for example:

```
A     C
   z

B     D
```

Assume the values at each pixel are simply given by $v$, e.g. $v_A, v_B$ etc.
We can then perform the interpolation as follows:

$$
v_{top} = v_A + (v_C - v_A) * (x - x_0)
$$
$$
v_{btm} = v_B + (v_D - v_B) * (x - x_0)
$$
$$
v_{z} = v_{top} + (v_{btm} - v_{top}) * (y - y_0)
$$

# Handling edge cases

The above operation is well-defined for corner pixels which exist, but is not well-defined for coordinates of $p$ which reside on the edge of the image or outside it. Libraries like IPP and NPP sometimes have edge-smoothing options which either allow you to repeat the edge pixels or perform some soft gradient-like descent outwards, but the standard scenario is to simply ignore these invalid coordinates.

Concretely, for an $m$ rows $\times$ $n$ columns image, this is given by the following criteria:

1. $x \in [0, n-1]$
2. $y \in [0, m-1]$

Note that the $-1$ in each upper limit is important, as it clearly denotes the final pixel's index in each direction.

This of course also means that, strictly speaking, _the valid region rectangle is $m-1 \times n-1$ rather than $m \times n$_. This is despite the fact that the number of pixels is $m \times n$. Drawing it out should make it clear:

```
              <----- Length n-1 ------->
              0     1            n-2   n-1
              --------------------------
 /\       0 | x     x    ...     x     x
  |       1 | x     x    ...     x     x
Length      | .........................P
 m-1        | ..........................
  |     m-2 | x     x    ...     x     x
 \/     m-1 | x  Q  x    ...     x     x
```

## Pixels _along_ the edges are deemed valid

In the above diagram, we consider points like $P$ and $Q$ to be valid, and should be interpolated.
This respects the maximum limit inclusion in the $x$ and $y$ range conditions above.

Checking the validity of the requested pixel coordinates for both conditions is _necessary and sufficient_. Once a requested pixel is deemed valid for interpolation, we can go on to determine the 4 corner pixels around it. Obviously, this presents a problem for cases like $P$ and $Q$ on the edge, where 2 or more pixels may not _exist_.

In these edge scenarios, it is important to note that a single 1D linear interpolation is sufficient and produces the correct value. For example, for point Q, only the values at $(0,m-1)$ and $(1, m-1)$ are required.
__This is equivalent to setting the two 'out-of-bounds' pixel values to 0, as $v_{btm}$ would then be equal to 0, and $v_{Q} = v_{top}$__.

Hence, handling the edge cases for this standard scenario is straightforward.

# Extension to wrap-around images

A wrap-around image is defined as one where the memory access is 1D, but the coordinates of the image have a well-defined periodicity.

The best analogy for this is a tape around a cylinder. The tape has a well-defined beginning and end, and we divide the tape into equal length 'loops' around the cylinder (for our purposes we assume the tape stops at an integer number of 'loops').

If we were to slice the cylinder at the position just before the start of the tape (or right after the end) and then unroll it, we could obtain a rectangular 'image' with $m$ rows and $n$ columns, just like before:

```
A   B   C

D   E   F

G   H   I
```

However, we could also slice it slightly forward, which would force us to discard our starting point:

```
B   C   D

E   F   G

H   I   ?
```

Similarly, we could slice it slightly backward, which would force us to discard our ending point:

```
?   A   B

C   D   E

F   G   H
```

In both latter cases, we would have a __blank__ (denoted by ?) where the tape would not exist. However, for all points in-between, interpolation would be valid in _all the cases_.

This means that points to the right of $C$ would now be considered valid for interpolation; there are still valid points to retrieve corner pixels from, as you only need to 'rotate the cylinder' to see them:

```
A   B   C         B   C   D
          P             P
D   E   F    ->   E   F   G

G   H   I         H   I   ?
```

## Implementation of the wrap-around

The above scenario might beg the question: do we have to consider all rotations? The answer, thankfully, is no. We only need to consider effectively one.

The key idea here is to extend the base image by 1 column. We do the following:

```
A   B   C        A   B   C   D

D   E   F   ->   D   E   F   G

G   H   I        G   H   I   ?
```

Ignore the unknown ending pixel for now. We will deal with it later.

We now have an $m \times n+1$ image, where the repeated column is the 1st column with a row offset. ___This is sufficient to perform all valid bilinear interpolations.___ Note that the range of $x$ is now $x \in [0, n)$, where the upper limit $n$ is now _non-inclusive_. This is an important detail we will come back to.

The key here is to simply transform the original requested pixel coordinates into _unwrapped pixel coordinates_. This is simple:

$$
x = \alpha n + u
$$

where $\alpha$ is an integer factor of the side length (which is now $n$). Extracting $u$ from this is simply the remainder of a floating-point division, as usually done by `modf` or a similar function.

> [!NOTE]
> In C++, this can be more efficiently performed using `remquof` or similar, and this is how I do it in the code.

The quotient of the division is the number of 'rows' we have equivalent 'wrapped-around', so we increment $y$ by this amount to simply obtain:

$$
v = y + \alpha
$$

The result of this is that we now have the unwrapped coordinates $x \rightarrow u$, $y \rightarrow v$ with our desired unwrapping of the $x$-direction value such that $u \in [0, n)$.

We can now determine validity of an interpolation request using these unwrapped coordinates $u$ and $v$. The following conditions are now required:

1. $u \in [0, n)$
2. $v \in [0, m]$

Again, note that $u$ does not include the ending pixel edge, since this would be a repeat of the first column.

## Taking care of the blank pixel

The above 2 conditions are not sufficient, as they would include pixels in the bottom right of the extended image i.e. around the square `FG?I`. However, since this is a square and it is at the bottom right, it is easy to ignore this area by adding 1 more condition:

3. $\text{NOT}\, \left(u \in (n-1, n) \,\text{AND}\, v \in (m-1, m]\right)$

The 3 conditions should be checked in order, since the points in this special bottom right square are 'in-range' of either dimension individually, but not as a pair.

## Bilinear interpolation is now the same as before

Assuming the 3 conditions are taken care of, we now can proceed with the standard 4 corner pixel extraction like before, setting out-of-bounds corners to 0. This will easily take care of the top and bottom 'edges' of the image (the only real edges are the top and bottom of the tape). The unwrapped coordinates will take care of the rest. The reader is invited to validate this (or believe the unit tests that have been set up).

Note that a beautiful side-effect of the unwrapping is that (assuming it is done correctly), negative requested pixel coordinates are now also valid, and can extend indefinitely (as long as the 3 conditions hold).
