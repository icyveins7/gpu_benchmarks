import numpy as np
import scipy as sp

def disk(radius: float):
    xg, yg = np.meshgrid(np.arange(-int(radius), int(radius)+1), np.arange(-int(radius), int(radius)+1))
    return (xg**2 + yg**2 <= radius**2).astype(np.uint8)


x = np.array([[-2,    1,    2,    0,   -2],
              [ 0,    1,   -3,    4,   -4],
              [-3,    2,   -5,    4,   -2],
              [ 1,   -5,    1,   -3,    1],
              [-4,    3,    2,    4,   -3]])

radius = 1.1
kern = disk(radius) * 1
print(kern)

res = sp.signal.convolve2d(x, kern)
length = kern.shape[0] // 2
print(res[length:-length, length:-length])
