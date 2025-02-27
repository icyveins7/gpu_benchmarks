# proj_ffts

This subdirectory is concerned with FFTs. For now, it has only tested cuFFT, although I intend to look at cuFFTdx.

Some scenario-specific implementations are noteworthy, and are documented below.

# Noteworthy Algorithms

## Padded 2D FFTs

For 1D FFTs, there is no way to improve the computational complexity of a padded output (or at least not one without several caveats).

The 2D padded case, however, offers up to an asymptotic 100% increase in speed (50% reduction in complexity), depending on the ratio of the padding to the input.

Recall that the 2D FFT can be implemented as just a series of 1D FFTs:

1. Take the FFTs down each column.
2. Take the result of the previous step and FFT each row.

The reverse is also possible (rows then columns).
As it turns out, this is probably identical (or very close to) cuFFT's internal 2D transform implementation, at least as far as the timings suggest.

The above affords a simple way to optimise the padded FFT. We simply ignore the padding in one of the dimensions!

For example, if we wish to have an $M \times N$ input padded to an $A \times B$ transform:

1. Take the column transforms for only the $N$ input columns, each with height $A$. The rest of the $B-N$ columns would transform to all 0s anyway.
2. Take the row transforms over all $A$ rows, each of length $B$.

With greater padding, this would asymptotically make the column transform complexity negligible.
In practice, the effect can be even greater; CUDA favours row transforms over column ones, since row data is contiguous.
Thus, reducing the column-wise complexity can result in a greater than 50% reduction in time.


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


