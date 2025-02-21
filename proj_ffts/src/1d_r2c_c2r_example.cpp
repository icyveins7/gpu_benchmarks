/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 *
 *
 * ====================================
 * Edited from the sample source code.
 */

#include <complex>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cufft.h>
#include "cufft_utils.h"


int main(int argc, char *argv[]) {
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

    // Print to check
    std::printf("fft_size = %d\nbatch_size = %d\n", fft_size, batch_size);

    cufftHandle planr2c, planc2r;
    cudaStream_t stream = NULL;

    int element_count = batch_size * fft_size;

    using scalar_type = float;
    using input_type = scalar_type;
    using output_type = std::complex<scalar_type>;

    std::vector<input_type> input(element_count, 0);
    std::printf("Allocated %zd for input \n", input.size());
    std::vector<output_type> output((fft_size / 2 + 1) * batch_size);
    std::printf("Allocated %zd for output \n", output.size());

    for (auto i = 0; i < element_count; i++) {
        input[i] = static_cast<input_type>(i);
    }

    std::printf("Input array:\n");
    // Don't print too much
    for (int i = 0; i < std::min(static_cast<int>(input.size()), 32); ++i) {
        // Show separation of every 'individual' array
        if (i % fft_size == 0)
          std::printf("---------\n");

        std::printf("%f\n", input.at(i));
    }
    std::printf("=====\n");

    input_type *d_input = nullptr;
    cufftComplex *d_output = nullptr;

    CUFFT_CALL(cufftCreate(&planr2c));
    CUFFT_CALL(cufftCreate(&planc2r));
    CUFFT_CALL(cufftPlan1d(&planr2c, fft_size, CUFFT_R2C, batch_size));
    CUFFT_CALL(cufftPlan1d(&planc2r, fft_size, CUFFT_C2R, batch_size));

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(planr2c, stream));
    CUFFT_CALL(cufftSetStream(planc2r, stream));

    // Create device arrays
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(input_type) * input.size()));
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(output_type) * output.size()));
    CUDA_RT_CALL(cudaMemcpyAsync(d_input, input.data(), sizeof(input_type) * input.size(),
                                 cudaMemcpyHostToDevice, stream));

    // out-of-place Forward transform
    CUFFT_CALL(cufftExecR2C(planr2c, d_input, d_output));

    CUDA_RT_CALL(cudaMemcpyAsync(output.data(), d_output, sizeof(output_type) * output.size(),
                                 cudaMemcpyDeviceToHost, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    std::printf("Output array after Forward FFT:\n");
    // Don't print too much
    for (int i = 0; i < std::min(static_cast<int>(output.size()), fft_size); ++i) {
        if (i % fft_size == 0)
          std::printf("---------\n");
        std::printf("%f + %fj\n", output.at(i).real(), output.at(i).imag());
    }
    std::printf("=====\n");


    // out-of-place Inverse transform
    CUFFT_CALL(cufftExecC2R(planc2r, d_output, d_input));

    // Normalize the data (AFTER, to prevent possible scaling issues)
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    real_scaling_kernel<<<1, 128, 0, stream>>>(
        d_input, static_cast<int>(input.size()),
        1.0f/static_cast<float>(fft_size));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    CUDA_RT_CALL(cudaMemcpy(output.data(), d_input, sizeof(input_type) * input.size(),
                                 cudaMemcpyDeviceToHost));

    std::printf("Output array after Forward FFT, Normalization, and Inverse FFT:\n");
    for (auto i = 0; i < std::min(static_cast<int>(input.size()/2), 32); i++) {
        if (i % (fft_size / 2) == 0)
          std::printf("---------\n");
        std::printf("%f\n", output[i].real());
        std::printf("%f\n", output[i].imag());
    }
    std::printf("=====\n");

    /* free resources */
    CUDA_RT_CALL(cudaFree(d_input));
    CUDA_RT_CALL(cudaFree(d_output));

    CUFFT_CALL(cufftDestroy(planr2c));
    CUFFT_CALL(cufftDestroy(planc2r));

    CUDA_RT_CALL(cudaStreamDestroy(stream));

    CUDA_RT_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}
