import numpy as np

from scipy.interpolate import RegularGridInterpolator

factor = 3
step = 0.8/factor

x = np.arange(4)
y = np.arange(4)
z = np.arange(x.size * y.size).reshape((x.size, y.size)) + 1
print(z)
zpad = np.pad(z, ((1,1), (1,1)))
xpad = np.arange(x.size + 2) - 1
ypad = np.arange(y.size + 2) - 1

interp = RegularGridInterpolator((xpad, ypad), zpad)

outShape = np.array([5, 5])
xp = np.arange(factor * outShape[1]) * step - step
yp = np.arange(factor * outShape[0]) * step - step
xg, yg = np.meshgrid(xp, yp)
zp = interp(np.vstack((xg.ravel(), yg.ravel())).T).reshape((yp.size,xp.size)).T

print(zp)
print(zp[:3,:3])
print(zp[:3,:3].sum())

for i in range(outShape[0]):
    s = f""
    for j in range(outShape[1]):
        s += f"{zp[i*factor:(i+1)*factor, j*factor:(j+1)*factor].sum():8.3f} "
    print(s)

