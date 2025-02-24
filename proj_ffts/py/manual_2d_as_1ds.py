import numpy as np

dims = (3, 2)
data = np.random.randn(*dims) + 1j*np.random.randn(*dims)

def manual1ds(data):
    first = np.fft.fft(data, axis=0) # apply to each col
    second = np.fft.fft(first, axis=1) # apply to each row
    return second

manualf = manual1ds(data)
f = np.fft.fft2(data)

np.testing.assert_allclose(manualf, f)

# Now check if it's padded
paddims = (5, 5)

def manualpad1ds(data, paddims, rowFirst=True):
    origdims = data.shape
    f = np.pad(data, ((0, paddims[0]-origdims[0]), (0, paddims[1]-origdims[1])), 'constant')
    # Only do the occupied rows at the start, row-wise ffts
    if rowFirst:
        print("Doing rows first")
        f[:origdims[0], :] = np.fft.fft(data, n=paddims[1], axis=1)
        f = np.fft.fft(f, axis=0)
    else:
        print("Doing cols first")
        f[:, :origdims[1]] = np.fft.fft(data, n=paddims[0], axis=0)
        f = np.fft.fft(f, axis=1)

    return f

manualfp = manualpad1ds(data, paddims, rowFirst=True)
manualfp2 = manualpad1ds(data, paddims, rowFirst=False)
np.testing.assert_allclose(manualfp, manualfp2)
fp = np.fft.fft2(data, s=paddims)

np.testing.assert_allclose(manualfp, fp)


