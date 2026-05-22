import numpy as np
from scipy.interpolate import CubicSpline

# Set your data here
x = np.array([0.0, 1.0, 2.0, 3.0]) # , 4.0])
y = np.array([0.00334699, 0.51242196, 0.91053456, 1.03974365]) # , 0.3])

print("--- Natural spline ---")
cs = CubicSpline(x, y, bc_type='natural')

for i in range(len(x) - 1):
    a, b, c, d = cs.c[3, i], cs.c[2, i], cs.c[1, i], cs.c[0, i]
    print(f"[{i}] x=[{x[i]:.4f}, {x[i+1]:.4f}]  a={a:+.10f}  b={b:+.10f}  c={c:+.10f}  d={d:+.10f}")

print("\n--- Clamped spline ---")
slopeLeft = 0.5
slopeRight = -0.3
cs_clamped = CubicSpline(x, y, bc_type=((1, slopeLeft), (1, slopeRight)))

for i in range(len(x) - 1):
    a, b, c, d = cs_clamped.c[3, i], cs_clamped.c[2, i], cs_clamped.c[1, i], cs_clamped.c[0, i]
    print(f"[{i}] x=[{x[i]:.4f}, {x[i+1]:.4f}]  a={a:+.10f}  b={b:+.10f}  c={c:+.10f}  d={d:+.10f}")
