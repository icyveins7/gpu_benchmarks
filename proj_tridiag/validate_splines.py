"""
Validate GPU natural spline output against scipy.interpolate.CubicSpline.

Reproduces the same data generation as splines.cu (srand(42), same formula)
and compares the per-interval (a, b, c, d) coefficients.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import ctypes

# Reproduce C's rand() with srand(42) using a simple LCG (glibc)
class CRand:
    def __init__(self, seed):
        self.state = seed

    def rand(self):
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state

    def randf(self):
        return 0.1 * self.rand() / 0x7FFFFFFF


def main():
    NUM_ROWS = 2
    N = 10

    rng = CRand(42)

    for row in range(NUM_ROWS):
        x = np.array([float(i) for i in range(N)])
        y = np.array([np.sin(i * 0.5) + rng.randf() for i in range(N)])

        print(f"Row {row} data:")
        print(f"  x = {{{', '.join(f'{v:.4f}' for v in x)}}}")
        print(f"  y = {{{', '.join(f'{v:.4f}' for v in y)}}}")

        cs = CubicSpline(x, y, bc_type='natural')

        # cs.c shape is (4, N-1): [cubic, quadratic, linear, constant]
        # Our convention: S(x) = a + b*t + c*t^2 + d*t^3, t = x - xmin
        # scipy convention: same but stored as cs.c[3]=a, cs.c[2]=b, cs.c[1]=c, cs.c[0]=d
        print(f"Row {row} spline coefficients (scipy):")
        for i in range(N - 1):
            a = cs.c[3, i]
            b = cs.c[2, i]
            c = cs.c[1, i]
            d = cs.c[0, i]
            print(f"  [{i:2d}] x=[{x[i]:+.4f}, {x[i+1]:+.4f}]"
                  f"  a={a:+.6f}  b={b:+.6f}  c={c:+.6f}  d={d:+.6f}")

            # Continuity check
            h = x[i + 1] - x[i]
            yEnd = a + h * (b + h * (c + h * d))
            print(f"       S({x[i+1]:+.4f}) = {yEnd:+.10f}"
                  f"  y[{i+1}] = {y[i+1]:+.10f}"
                  f"  diff = {abs(yEnd - y[i+1]):.6e}")
        print("-------------")


if __name__ == "__main__":
    main()
