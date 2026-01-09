import numpy as np
import scipy as sp

from sats import drawCircle

def manual_convolve_2d(x: np.ndarray, kernel: np.ndarray, keepValid: bool = False):
    # Assume kernel is square
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel must be square")

    kernelRadius = kernel.shape[0] // 2

    fftlen = np.array(x.shape) + kernelRadius
    print(f"FFT dims = {fftlen}")

    kernelpad = np.pad(kernel, ((0,fftlen[0]-kernel.shape[0]),(0,fftlen[1]-kernel.shape[1])))
    print(kernelpad.shape)
    kernelfft = sp.fft.fft2(kernelpad)

    xpad = np.pad(x, ((0,fftlen[0]-x.shape[0]),(0,fftlen[1]-x.shape[1])))
    print(xpad.shape)
    xpadfft = sp.fft.fft2(xpad)

    result = sp.fft.ifft2(xpadfft * kernelfft)

    if keepValid:
        return result[kernelRadius:, kernelRadius:]

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        length = int(sys.argv[1])
        kernLength = int(sys.argv[2])
    else:
        length = 5
        kernLength = 3

    x = np.arange(length*length).reshape((length, length))
    print(x)
    # kernel = np.ones((kernLength, kernLength))
    kernel, _ = drawCircle(kernLength//2)
    print(kernel)

    res = manual_convolve_2d(x, kernel, True)
    print(res.real)

