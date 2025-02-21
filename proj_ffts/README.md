# benchmark_cufft

This repository is a quick and dirty study to measure GPU acceleration with respect to DFTs, specifically for real-valued inputs (like image data).

# Building and running

```
cmake -B build
cd build
make

./build/bin/{your_executable_here}
```

# Available executables

## 1d_r2c_c2r_example
This is just copied directly from the NVIDIA CUDA Library examples repository, with some changes. It tests both the forward and backward transforms, with an additional scaling kernel to ensure equality at the end. Note that the transforms here are only 1D.

## thrust_example
This is designed to be more C++-friendly, using `thrust` containers, and a memory-managed cuFFT wrapper (see `cuFFT_wrapper.cuh`). It only studies the forward 1D transform. However it also includes a simple IPP 1D Fourier Transform as a comparison.

## r2c_2d
This is designed using `thrust` containers (and a similar memory-managed cuFFT wrapper). It also only studies the forward 1D transform, but for 2D (batches are still allowed). It also includes an equivalent IPP 2D Fourier transform as a comparison.

# Sample Results

## Intel Xeon Gold 6338T, NVIDIA A10

### Length 10000, 10000 rows (1D, 32f->32fc)

IPP 1D: 137ms.

cuFFT (fft kernel + postprocess kernel): ~3.3ms.

![screenshot of cufft calls from nsys profile](screenshots/cufft_10000x10000_1d_r2c_A10.png)

cuFFT (HtoD memcpy): 37.8ms (~10GB/s, nowhere near the maximum pinned memory transfer speed).

cuFFT (DtoH memcpy): 40ms (~10GB/s, nowhere near the maximum pinned memory transfer speed).

The memcpys tend to be very slow on the first invocation, so for good timing, the same data was copied twice (in both directions), and the lower (second) duration was taken.

Hopefully it is obvious by now that the FFTs *cannot* be used alone, as the memcpy overhead just for copying the input/output is still too large.

NumPy/SciPy: 263ms.
```python
import numpy as np
import scipy as sp

x = np.random.randn(10000, 10000).astype(np.float32) # Specifically cast to float32

%%timeit
y = sp.fft.rfft(x) # This does it for each row, scipy maintains wordsizes so it's complex64
```

### Dimensions 10000x10000, batch size 2 (2D, 32f->32fc)
cuFFT (fft kernel + postprocess kernel + 2 regular_fft_factor kernels): ~14.3ms.

IPP 2D: 1.09s.

### Dimensions 10000x10000, batch size 16 (2D, 32f->32fc)
cuFFT (fft kernel + postprocess kernel + 2 regular_fft_factor kernels): ~113ms.

No additional speedup compared to batch size 2; GPU occupancy is already 100% here.

IPP 2D: 8.34s.


