#include "cuFFT_wrapper.cuh"
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/sequence.h>
#include <nvtx3/nvToolsExt.h>

int main()
{
  int batch = 2;
  std::array<int, 2> inputSize = {2, 2};
  std::array<int, 2> paddedSize = {4, 3};
  thrust::device_vector<thrust::complex<float>> d_input(inputSize[0] * inputSize[1] * batch);
  thrust::sequence(d_input.begin(), d_input.end());
  // pad the input
  thrust::device_vector<thrust::complex<float>> d_padded(paddedSize[0] * paddedSize[1] * batch);
  // copy to top left corner for each image
  for (int b = 0; b < batch; b++)
  {
    cudaMemcpy2D(
      thrust::raw_pointer_cast(&d_padded[b * paddedSize[0] * paddedSize[1]]),
      paddedSize[1] * sizeof(thrust::complex<float>),
      thrust::raw_pointer_cast(&d_input[b * inputSize[0] * inputSize[1]]),
      inputSize[1] * sizeof(thrust::complex<float>),
      inputSize[1] * sizeof(thrust::complex<float>),
      inputSize[0],
      cudaMemcpyDeviceToDevice
    );
  }
  // check that the padding is correct
  thrust::host_vector<thrust::complex<float>> h_padded = d_padded;
  for (int b = 0; b < batch; b++)
  {
    for (int i = 0; i < paddedSize[0]; i++)
    {
      for (int j = 0; j < paddedSize[1]; j++)
      {
        std::cout << h_padded[b * paddedSize[0] * paddedSize[1] + i * paddedSize[1] + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "=======================" << std::endl;
  }

  // now execute the default fft
  printf("Default 2D FFT (ground truth)\n");
  cuFFTWrapper_2D<CUFFT_C2C> def_fft(paddedSize, batch);
  thrust::device_vector<thrust::complex<float>> d_y(d_padded.size());
  def_fft.exec(d_padded, d_y);
  thrust::host_vector<thrust::complex<float>> h_y = d_y;
  for (int b = 0; b < batch; b++)
  {
    for (int i = 0; i < paddedSize[0]; i++)
    {
      for (int j = 0; j < paddedSize[1]; j++)
      {
        std::cout << h_y[b * paddedSize[0] * paddedSize[1] + i * paddedSize[1] + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "=======================" << std::endl;
  }

  // and finally execute it via our manual class
  printf("Manual 2D FFT\n");
  cuFFTWrapper_2DPad_as_1Ds<CUFFT_C2C> man_fft(paddedSize, inputSize, batch);
  thrust::device_vector<thrust::complex<float>> d_z(d_padded.size());
  man_fft.exec(d_padded, d_z);
  thrust::host_vector<thrust::complex<float>> h_z = d_z;
  for (int b = 0; b < batch; b++)
  {
    for (int i = 0; i < paddedSize[0]; i++)
    {
      for (int j = 0; j < paddedSize[1]; j++)
      {
        std::cout << h_z[b * paddedSize[0] * paddedSize[1] + i * paddedSize[1] + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "=======================" << std::endl;
  }

  // time a large one
  std::array<int, 2> bigInputSize = {512, 512};
  std::array<int, 2> bigPaddedSize = {8192, 8192};
  thrust::device_vector<thrust::complex<float>> d_biginput(bigInputSize[0] * bigInputSize[1] * batch);
  thrust::sequence(d_biginput.begin(), d_biginput.end());
  thrust::device_vector<thrust::complex<float>> d_bigpadded(bigPaddedSize[0] * bigPaddedSize[1] * batch);
  for (int b = 0; b < batch; b++)
  {
    cudaMemcpy2D(
      thrust::raw_pointer_cast(&d_bigpadded[b * bigPaddedSize[0] * bigPaddedSize[1]]),
      bigPaddedSize[1] * sizeof(thrust::complex<float>),
      thrust::raw_pointer_cast(&d_biginput[b * bigInputSize[0] * bigInputSize[1]]),
      bigInputSize[1] * sizeof(thrust::complex<float>),
      bigInputSize[1] * sizeof(thrust::complex<float>),
      bigInputSize[0],
      cudaMemcpyDeviceToDevice
    );
  }

  // run the default one
  cuFFTWrapper_2D<CUFFT_C2C> big_def_fft(bigPaddedSize, batch);
  thrust::device_vector<thrust::complex<float>> d_bigy(d_bigpadded.size());
  nvtxRangePush("Default 2D FFT");
  big_def_fft.exec(d_bigpadded, d_bigy);
  nvtxRangePop();

  // run the manual one that replicates the default
  cuFFTWrapper_2D_as_1Ds<CUFFT_C2C> big_man_fft_nopad(bigPaddedSize, batch);
  thrust::device_vector<thrust::complex<float>> d_bigz_nopad(d_bigpadded.size());
  nvtxRangePush("Manual 2D FFT");
  big_man_fft_nopad.execCols(d_bigpadded, d_bigz_nopad);
  big_man_fft_nopad.execRows(d_bigz_nopad, d_bigz_nopad);
  nvtxRangePop();

  // run our manual padded one
  cuFFTWrapper_2DPad_as_1Ds<CUFFT_C2C> big_man_fft(bigPaddedSize, bigInputSize, batch);
  thrust::device_vector<thrust::complex<float>> d_bigz(d_bigpadded.size());
  nvtxRangePush("Custom Padded 2D FFT");
  big_man_fft.exec(d_bigpadded, d_bigz);
  nvtxRangePop();

  return 0;
}
