import numpy as np

from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation

factor = 3
padding = factor // 2
step = 0.8/factor
angle = np.pi/2 # corresponds to -90 deg in the oversample_remap.cu option

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

interp = RegularGridInterpolator((xpad, ypad), zpad, bounds_error=False, fill_value=None) # pyright:ignore

outShape = np.array([outheight, outwidth])
xp = np.arange(factor * outShape[1]) * step - step * padding
yp = np.arange(factor * outShape[0]) * step - step * padding
print(xp)
print(yp)
# xpg_0, ypg_0 = np.meshgrid(xp, yp, indexing="ij")
xpg_0, ypg_0 = np.meshgrid(xp, yp)

centre = np.array([
    inwidth // 2 - (0.5 * (inwidth%2==0)),
    inheight // 2 - (0.5 * (inheight%2==0))
])

if angle != 0:
    rotmat = Rotation.from_rotvec(angle * np.array([0, 0, 1])).as_matrix()[:2,:2]
    print(rotmat)
    xypg = np.vstack((xpg_0.ravel(), ypg_0.ravel())) - centre.reshape((-1,1)) # from centre
    xypg = rotmat @ xypg
    xpg = xypg[0].reshape(xpg_0.shape) + centre[0] # add back
    ypg = xypg[1].reshape(ypg_0.shape) + centre[1]
else:
    xpg = xpg_0
    ypg = ypg_0


zp = interp((xpg, ypg)).T


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

