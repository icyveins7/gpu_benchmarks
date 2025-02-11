import cupy as cp

loops = 5

data = cp.zeros(1000000000, dtype=cp.float32)
for i in range(loops):
    data = cp.random.rand(data.size, dtype=cp.float32)

h_data = data.get()
print(h_data[:10])
