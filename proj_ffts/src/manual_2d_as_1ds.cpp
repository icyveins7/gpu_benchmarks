#include "cuFFT_wrapper.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/sequence.h>
#include <nvtx3/nvToolsExt.h>

int main()
{
    std::array<int, 2> fftSize = { 2, 3 };
    int batch = 2;
    thrust::host_vector<thrust::complex<float>> x(fftSize[0] * fftSize[1] * 2);
    thrust::sequence(x.begin(), x.end());
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < fftSize[0]; i++)
        {
            for (int j = 0; j < fftSize[1]; j++)
            {
                std::cout << x[b * fftSize[0] * fftSize[1] + i * fftSize[1] + j] << " ";
            }
            std::cout << std::endl;
        }
         std::cout << "=======================" << std::endl;
    }
    thrust::device_vector<thrust::complex<float>> d_x = x;

    // First run it via the default 2D transforms
    printf("Default 2D FFT (ground truth)\n");
    cuFFTWrapper_2D<CUFFT_C2C> def_fft(fftSize, batch);
    thrust::device_vector<thrust::complex<float>> d_y(d_x.size());
    def_fft.exec(d_x, d_y);
    thrust::host_vector<thrust::complex<float>> h_y = d_y;
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < fftSize[0]; i++)
        {
            for (int j = 0; j < fftSize[1]; j++)
            {
                std::cout << h_y[b * fftSize[0] * fftSize[1] + i * fftSize[1] + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "=======================" << std::endl;
    }

    // Now run it via the manual batched 1D transforms
    printf("Row-wise transforms only\n");
    cuFFTWrapper_2D_as_1Ds<CUFFT_C2C> man_fft(fftSize, batch);
    thrust::device_vector<thrust::complex<float>> d_z(d_x.size());
    man_fft.execRows(d_x, d_z);
    thrust::host_vector<thrust::complex<float>> h_z = d_z;
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < fftSize[0]; i++)
        {
            for (int j = 0; j < fftSize[1]; j++)
            {
                std::cout << h_z[b * fftSize[0] * fftSize[1] + i * fftSize[1] + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "=======================" << std::endl;
    }

    printf("Col-wise transforms only\n");
    man_fft.execCols(d_x, d_z);
    h_z = d_z;
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < fftSize[0]; i++)
        {
            for (int j = 0; j < fftSize[1]; j++)
            {
                std::cout << h_z[b * fftSize[0] * fftSize[1] + i * fftSize[1] + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "=======================" << std::endl;
    }

    // Assume col-wise already done, do row-wise on interrim output, in place
    printf("Col+Row-wise == 2D\n");
    man_fft.execRows(d_z, d_z);
    h_z = d_z;
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < fftSize[0]; i++)
        {
            for (int j = 0; j < fftSize[1]; j++)
            {
                std::cout << h_z[b * fftSize[0] * fftSize[1] + i * fftSize[1] + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "=======================" << std::endl;
    }

    // Now let's run a big one
    std::array<int, 2> bigfftSize = {8192, 8192};
    thrust::host_vector<thrust::complex<float>> h_big(bigfftSize[0] * bigfftSize[1] * batch);
    thrust::sequence(x.begin(), x.end());
    thrust::device_vector<thrust::complex<float>> d_big = h_big;
    thrust::device_vector<thrust::complex<float>> d_bigout(d_big.size());

    // Run default 2D
    cuFFTWrapper_2D<CUFFT_C2C> def_bigfft(bigfftSize, batch);
    nvtxRangePush("Default 2D FFT");
    def_bigfft.exec(d_big, d_bigout);
    nvtxRangePop();

    // Now run it via the manual batched 1D transforms
    cuFFTWrapper_2D_as_1Ds<CUFFT_C2C> man_bigfft(bigfftSize, batch);
    nvtxRangePush("Manual 2D FFT");
    man_bigfft.execCols(d_big, d_bigout);
    man_bigfft.execRows(d_bigout, d_bigout);
    nvtxRangePop();




    return 0;
}
