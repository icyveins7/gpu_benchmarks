import numpy as np

from scipy.interpolate import RegularGridInterpolator

factor = 5
padding = factor // 2
step = 0.8/factor

inwidth = 4
inheight = 4
outwidth = 5
outheight = 5

x = np.arange(inwidth)
y = np.arange(inheight)
z = np.arange(x.size * y.size).reshape((x.size, y.size)) + 1
print(z)
zpad = np.pad(z, ((padding,padding), (padding,padding)))
xpad = np.arange(x.size + 2*padding) - padding
ypad = np.arange(y.size + 2*padding) - padding
print(xpad)
print(ypad)

interp = RegularGridInterpolator((xpad, ypad), zpad)

outShape = np.array([outheight, outwidth])
xp = np.arange(factor * outShape[1]) * step - step * padding
yp = np.arange(factor * outShape[0]) * step - step * padding
print(xp)
print(yp)
xg, yg = np.meshgrid(xp, yp, indexing="ij")
# zp = interp(np.vstack((xg.ravel(), yg.ravel())).T).reshape((yp.size,xp.size)).T
zp = interp((xg, yg))


zfinal = np.zeros((outShape[0], outShape[1]), np.float64)
for i in range(outShape[0]):
    s = f""
    for j in range(outShape[1]):
        agg = zp[i*factor:(i+1)*factor, j*factor:(j+1)*factor].sum() / (factor**2)
        # agg = zp[i*factor:(i+1)*factor, j*factor:(j+1)*factor].sum()
        s += f"{agg:8.3f} "
        zfinal[i, j] = agg
    print(s)

# print(f"37,0: {zfinal[37, 0]}")

