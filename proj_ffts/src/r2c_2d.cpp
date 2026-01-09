#include <cufft.h>
#include "cufft_utils.h"

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/async/copy.h>

#include "cuFFT_wrapper.cuh"


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "ipp_ext.h"
#pragma GCC diagnostic pop

#include "timer.h"

using dim_t = std::array<int, 2>;

int main(int argc, char *argv[]){

  // Define default sizes
  dim_t fft_size = {2, 4}; // height * width
  int batch_size = 2;

  // Retrieve from command line args
  // Expect (fft_size) (batch_size)
  if (argc >= 2)
    std::sscanf(argv[1], "%d,%d", &fft_size[0], &fft_size[1]);

  if (argc >= 3)
    std::sscanf(argv[2], "%d", &batch_size);

  if (argc >= 4)
  {
    std::printf("Arguments are (fft_size) (batch_size)\n");
    return -1;
  }

  printf("Beginning CUDA test (%d x %d, batch %d).\n",
      fft_size[0], fft_size[1], batch_size
      );

  size_t element_count = fft_size[0] * fft_size[1] * batch_size;
  thrust::host_vector<float> x(element_count);

  // Fill easily from 0 to N-1
  thrust::sequence(x.begin(), x.end());

  // // For debugging, write values exactly (for 2x4, batch 2)
  // x[0] = 0;
  // x[1] = 1;
  // x[2] = 3;
  // x[3] = 2;
  // x[4] = 4;
  // x[5] = 9;
  // x[6] = 7;
  // x[7] = 1;
  //
  // x[8] = 0;
  // x[9] = 1;
  // x[10] = 3;
  // x[11] = 2;
  // x[12] = 4;
  // x[13] = 9;
  // x[14] = 7;
  // x[15] = 1;

  // Check
  if (x.size () <= 100)
  {
    for (size_t i = 0; i < x.size(); ++i){
      if (i % (fft_size[0]*fft_size[1]) == 0)
        printf("---------\n");

      printf("%5zd: %f\n", i, x[i]);
    }
  }
  printf("===============\n");

  // Copy to device
  thrust::device_vector<float> d_x = x;
  // Make output
  size_t outputPerRow = fft_size[1]/2 + 1;
  size_t perImage = fft_size[0] * outputPerRow;
  thrust::device_vector<std::complex<float>> d_y(perImage * batch_size);

  // // Create and run cuFFT 2D
  // cufftHandle planr2c;
  // // CUFFT_CALL(cufftCreate(&planr2c));
  // CUFFT_CALL(cufftPlanMany(&planr2c, 
  //       fft_size.size(), fft_size.data(),
  //       nullptr, 1, 0,
  //       nullptr, 1, 0,
  //       CUFFT_R2C, batch_size));
  //
  //
  // // Execute
  // CUFFT_CALL(cufftExecR2C(planr2c,
  //     thrust::reinterpret_pointer_cast<cufftReal*>(d_x.data()),
  //     thrust::reinterpret_pointer_cast<cufftComplex*>(d_y.data())));
  // // Repeat for timing purposes
  // CUFFT_CALL(cufftExecR2C(planr2c,
  //     thrust::reinterpret_pointer_cast<cufftReal*>(d_x.data()),
  //     thrust::reinterpret_pointer_cast<cufftComplex*>(d_y.data())));
  //
  // // Free
  // CUFFT_CALL(cufftDestroy(planr2c));
  //

  // Memory managed wrapper version
  cuFFTWrapper_2D<CUFFT_R2C> fft(fft_size, batch_size);
  fft.exec(d_x, d_y);
  fft.exec(d_x, d_y);

  cuFFTWrapper_2D<CUFFT_C2R> fftc2r(fft_size, batch_size);
  fftc2r.exec(d_y, d_x);
  fftc2r.exec(d_y, d_x);

  // Read output and print
  thrust::host_vector<std::complex<float>> h_y = d_y;

  if (x.size() <= 100)
  {
    for (int b = 0; b < batch_size; ++b)
    {
      for (int i = 0; i < fft_size[0]; ++i)
      {
        for (int j = 0; j < outputPerRow; ++j) {
          printf("%12.6f %12.6fi, ", 
              h_y[b*perImage + i*outputPerRow + j].real(),
              h_y[b*perImage + i*outputPerRow + j].imag());
        }
        printf("\n");

      }
      printf("------------------------------\n");
    }
  }


  // Attempt IPP 2D checks
  printf("Beginning IPP tests..\n");
  // ippi::image<Ipp32f, ippi::channels::C1> img(fft_size[1], fft_size[0]);
  ippi::image<Ipp32f, ippi::channels::C1> i_y(fft_size[1], fft_size[0]); // allocate for dft output

  // make dft object (note it's width, height in IPP, so reversed the index order)
  ippi::DFT_RToPack<Ipp32f> dft({fft_size[1], fft_size[0]});

  // loop over every image
  {
    HighResolutionTimer timer;
    for (int k = 0; k < batch_size; ++k)
    {
      dft.fwd<ippi::channels::C1>(
        &x[k * fft_size[0] * fft_size[1]],
        sizeof(Ipp32f) * fft_size[1], // thrust vector is just contiguous memory
        i_y.data(),
        static_cast<int>(i_y.stepBytes()) // image object has its own stepBytes
      );


      // check output
      if (x.size() <= 100)
      {
        // make full complex output
        ippi::image<Ipp32fc, ippi::channels::C1> i_y_unpack(fft_size[1], fft_size[0]);

        // unpack it for better viewing
        ippiPackToCplxExtend_32f32fc_C1R(
            i_y.data(),
            i_y.size(),
            static_cast<int>(i_y.stepBytes()),
            i_y_unpack.data(),
            static_cast<int>(i_y_unpack.stepBytes())
            );

        // printing
        for (int i = 0; i < fft_size[0]; ++i)
        {
          for (int j = 0; j < fft_size[1]; ++j) {
            printf("%12.6f %12.6fi, ", i_y_unpack.at(i, j).re, i_y_unpack.at(i, j).im);
          }
          printf("\n");

        }
        printf("------------------------------\n");
      }
    }

  }

  return 0;
}
