import cupy as cp

size = 8192

x = cp.random.randn(size,size) + cp.random.randn(size,size)*1j
y = cp.random.randn(size,size) + cp.random.randn(size,size)*1j

x = x.astype(cp.complex64)
y = y.astype(cp.complex64)

z = x * y
