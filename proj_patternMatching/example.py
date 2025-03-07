import numpy as np
import cupy as cp
import time

# Some parameters
numPatterns = 4096
size = 64
numData = 1024


# Generate some patterns
patterns = np.random.randn(numPatterns, size).astype(np.float32)

# Generate some data
data = np.random.randn(numData, size).astype(np.float32)

# Define matching function
def matchSqErr(patterns: np.ndarray | cp.ndarray, data: np.ndarray | cp.ndarray):
    xp = cp.get_array_module(patterns)
    patternMinIdxs = xp.zeros(data.shape[0], dtype=np.uint32)
    patternMinCosts = xp.zeros(data.shape[0], dtype=np.float32)
    for i, dataInstance in enumerate(data):
        costs = xp.sum((dataInstance - patterns)**2, axis=1)
        mi = xp.argmin(costs)
        patternMinIdxs[i] = mi
        patternMinCosts[i] = costs[mi]

    return patternMinIdxs, patternMinCosts

# Call it
start = time.perf_counter()
patternMinIdxs, patternMinCosts = matchSqErr(patterns, data)
end = time.perf_counter()
print("Time: %s seconds" % (end - start))


# Try on GPU
d_patterns = cp.asarray(patterns)
d_data = cp.asarray(data)

# Repeat a few times for timing purposes
for i in range(3):
    d_patternMinIdxs, d_patternMinCosts = matchSqErr(d_patterns, d_data)
    h_patternMinIdxs = d_patternMinIdxs.get()
    h_patternMinCosts = d_patternMinCosts.get()



