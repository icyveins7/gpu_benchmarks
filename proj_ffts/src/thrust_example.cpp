#include <complex>
#include <cstdio>
#include <iostream>


#include <cufft.h>
#include "cufft_utils.h"

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/async/copy.h>

#include "cuFFT_wrapper.cuh"

#include "ipp.h"
#include "ipp/ipptypes.h"
#include "ipp_ext.h"

#include "timer.h"

int main(int argc, char *argv[]){

  // Define default sizes
  int fft_size = 8;
  int batch_size = 2;

  // Retrieve from command line args
  // Expect (fft_size) (batch_size)
  if (argc >= 2)
    std::sscanf(argv[1], "%d", &fft_size);

  if (argc >= 3)
    std::sscanf(argv[2], "%d", &batch_size);

  if (argc >= 4)
  {
    std::printf("Arguments are (fft_size) (batch_size)\n");
    return -1;
  }

  printf("Beginning CUDA test.\n");
  // Use thrust vectors!
  size_t element_count = fft_size * batch_size;
  thrust::host_vector<float> x(element_count);

  // Fill easily from 0 to N-1
  thrust::sequence(x.begin(), x.end());

  // Check
  if (x.size () <= 100)
  {
    for (size_t i = 0; i < x.size(); ++i){
      if (i % fft_size == 0)
        printf("---------\n");

      printf("%5zd: %f\n", i, x[i]);
    }
  }

  // (Async) Copy to device
  thrust::device_vector<float> d_x(x.size());
  // This seems to explicitly make a new stream
  thrust::device_event e = thrust::async::copy(
      x.begin(), x.end(), d_x.begin());

  // Note that this is the same as this, so just use this from now on..
  // This uses the default stream
  d_x = x;

  // // Create the simple forward plan
  // cufftHandle planr2c;
  // CUFFT_CALL(cufftCreate(&planr2c));
  // CUFFT_CALL(cufftPlan1d(&planr2c, fft_size, CUFFT_R2C, batch_size));
  //
  // // Perform FFT on default stream
  // size_t outputPerRow = fft_size/2 + 1;
  // size_t outputSize = outputPerRow * batch_size;
  // thrust::device_vector<std::complex<float>> d_y(outputSize);
  // CUFFT_CALL(cufftExecR2C(planr2c,
  //       reinterpret_cast<cufftReal*>(thrust::raw_pointer_cast(d_x.data())), // input
  //       reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(d_y.data()))) // output
  //     );
  // // We run a 2nd time so that the two resultant kernels are stacked next to each other
  // CUFFT_CALL(cufftExecR2C(planr2c,
  //       reinterpret_cast<cufftReal*>(thrust::raw_pointer_cast(d_x.data())), // input
  //       reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(d_y.data()))) // output
  //     );
  // CUFFT_CALL(cufftDestroy(planr2c));

  // Memory managed container
  thrust::device_vector<std::complex<float>> d_y;
  size_t outputPerRow;
  cuFFTWrapper_1D<CUFFT_R2C> cfft(fft_size, batch_size);
  d_y.resize(cfft.outputSize());
  outputPerRow = cfft.outputPerRow();
  printf("Output per row = %zd\n", outputPerRow);
  cfft.exec(d_x, d_y);
  cfft.exec(d_x, d_y); // exec 2nd time for timing purposes

  // Get output back on host
  thrust::host_vector<std::complex<float>> h_y = d_y;
  // get it again to check timing
  h_y = d_y;

  // Check
  if (h_y.size() <= 100)
  {
    for (size_t i = 0; i < h_y.size(); ++i){
      if (i % outputPerRow == 0)
        printf("---------\n");

      printf("%5zd: %f, %f\n", i, h_y[i].real(), h_y[i].imag());
    }
  }


  // Now test IPP implementation, a simple one
  printf("Beginning IPP tests.\n");
  ipps::vector<Ipp32f> ix(x.size());
  ipps::Copy(x.data(), ix.data(), ix.size());


  // // Prepare IPPI DFT for R2C, for a single row for now
  // // width, height
  IppiSize roiSize = {fft_size, 1};
  // int sizeSpec, sizeInit, sizeBuffer;
  // ippiDFTGetSize_R_32f( roiSize, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuffer );
  // /// allocate memory for required buffers
  // IppiDFTSpec_R_32f* pMemSpec = (IppiDFTSpec_R_32f*) ippMalloc ( sizeSpec );
  //
  // Ipp8u* pMemInit;
  // if ( sizeInit > 0 )
  // {
  //   pMemInit = (Ipp8u*)ippMalloc ( sizeInit );
  // }
  //
  // Ipp8u* pMemBuffer;
  // if ( sizeBuffer > 0 )
  // {
  //   pMemBuffer = (Ipp8u*)ippMalloc ( sizeBuffer );
  // }
  //
  // /// initialize DFT specification structure
  // ippiDFTInit_R_32f( roiSize, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, pMemSpec, pMemInit );
  //
  // /// free initialization buffer
  // if ( sizeInit > 0 )
  // {
  //   ippFree( pMemInit );
  // }
  //

  // use new wrapped ippi DFT object
  ippi::DFT_RToPack<Ipp32f> dft(roiSize);

  // Invoke DFTs (for every row)
  ipps::vector<Ipp32fc> iy(h_y.size(), {0.0f, 0.0f});
  ipps::vector<Ipp32fc> iy_unpack(fft_size * batch_size, {0.0f, 0.0f}); // full complex FFT output

  {
    HighResolutionTimer timer;
    for (size_t i = 0; i < batch_size; ++i)
    {
      // new wrapped call
      dft.fwd<ippi::channels::C1>(
        &ix.at(i*fft_size),
        fft_size*sizeof(Ipp32f), // note this is number of bytes!
        (Ipp32f*)(&iy.at(i*(fft_size/2+1))),
        (fft_size/2 + 1)*sizeof(Ipp32f) // note this is number of bytes!
      );

      // old raw call (number of step bytes is wrong, but doesn't matter because 1d!)
      // IppStatus sts = ippiDFTFwd_RToPack_32f_C1R(
      //     &ix.at(i*fft_size), fft_size, 
      //     (Ipp32f*)(&iy.at(i*(fft_size/2 +1))), fft_size/2 + 1, 
      //     pMemSpec, pMemBuffer);
    }
  }

  // Check
  // NOTE: IPP stores in RCPack2D Format
  // For 1-D,
  // what this means is it drops the first and last imaginary values of the output (which are always 0)
  // Hence the storage is actually 1 complex element less than CUDA (or other FFTW equivalents, like numpy/scipy)
  // 0: Re_0
  // 1: Re_1
  // 2: Im_1
  // ...
  //  : Re_(N/2)
  //  In short, REAL | CPLX ...... CPLX | REAL

  if (h_y.size() <= 100)
  {
    for (size_t i = 0; i < iy.size(); ++i)
    {
      if (i % outputPerRow == 0)
        printf("---------\n");
      printf("%3zd: %f, %f\n", i, iy.at(i).re, iy.at(i).im);
    }
  }

  // Unpack data for better views
  for (size_t i = 0; i < batch_size; ++i){
    ippiPackToCplxExtend_32f32fc_C1R(
        (Ipp32f*)&iy.at(i * outputPerRow),
        roiSize,
        outputPerRow * sizeof(Ipp32fc),
        &iy_unpack.at(i * fft_size),
        fft_size * sizeof(Ipp32fc)
        );
  }


  if (h_y.size() <= 100)
  {
    for (size_t i = 0; i < iy_unpack.size(); ++i)
    {
      if (i % fft_size == 0)
        printf("---------\n");
      printf("%3zd: %f, %f\n", i, iy_unpack.at(i).re, iy_unpack.at(i).im);
    }
  }

  // // IPP Cleanup
  // ippFree(pMemBuffer);
  // ippFree(pMemSpec);


  printf("End.\n");
  return 0;
}
