#include <iostream>

#include "block_and_grid_sizing.cuh"
#include "containers/image.cuh"
#include "containers/streams.cuh"
#include "extra_type_traits.cuh"
#include "pinnedalloc.cuh"

#include <nvtx3/nvToolsExt.h>

template <typename Tdata, typename Tidx = int16_t>
__global__ void
keep_index_matching_value_kernel(const containers::Image<const Tdata, Tidx> in,
                                 const Tdata val, cuda_vec2_t<Tidx> *outlist,
                                 unsigned int *outcount) {
  for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < in.height;
       i += blockDim.y * gridDim.y) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < in.width;
         j += blockDim.x * gridDim.x) {
      if (in.at(i, j) == val) {
        outlist[atomicAdd(outcount, 1)] = cuda_vec2_t<Tidx>{(Tidx)j, (Tidx)i};
      }
    }
  }
}

template <typename Tdata, typename Tidx = int16_t>
__global__ void keep_index_matching_value_tile_kernel(
    const containers::Image<const Tdata, Tidx> in, const Tdata val,
    const Tidx startRow, const Tidx startCol, const Tidx tileRows,
    const Tidx tileCols, cuda_vec2_t<Tidx> *outlist, unsigned int *outcount) {
  for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < tileRows;
       i += blockDim.y * gridDim.y) {
    int y = i + startRow;
    if (!in.rowIsValid(y)) {
      continue;
    }
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < tileCols;
         j += blockDim.x * gridDim.x) {
      int x = j + startCol;
      if (!in.colIsValid(x)) {
        continue;
      }

      if (in.at(y, x) == val) {
        outlist[atomicAdd(outcount, 1)] = cuda_vec2_t<Tidx>{(Tidx)y, (Tidx)x};
      }
    }
  }
}

template <typename Tdata, typename Tidx = int16_t>
__global__ void copy_compacted_list_via_mapped_host_pointer_kernel(
    const cuda_vec2_t<Tidx> *outlist, const unsigned int *outcount,
    cuda_vec2_t<Tidx> *h_outlist) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (int)*outcount;
       i += blockDim.x * gridDim.x) {
    h_outlist[i] = outlist[i];
  }
}

int main() {
  printf("Stream compaction via custom kernel\n");

  using Tidx = int16_t;
  using Tdata = int8_t;

  containers::DeviceImageStorage<Tdata, Tidx> d_in(15000, 15000);
  thrust::pinned_host_vector<Tdata> h_in(d_in.width * d_in.height);
  for (int i = 0; i < d_in.width * d_in.height; ++i) {
    h_in[i] = std::rand() % 2;
  }
  d_in.vec = h_in;

  thrust::pinned_host_vector<unsigned> h_count(1);
  thrust::device_vector<unsigned> d_count(1);
  thrust::pinned_host_vector<cuda_vec2_t<Tidx>> h_out(h_in.size());
  thrust::device_vector<cuda_vec2_t<Tidx>> d_out(h_in.size());

  for (int i = 0; i < 3; ++i) {
    nvtxRangePush("default");
    constexpr dim3 tpb(32, 4);
    dim3 blks(justEnoughBlocks(tpb.x, (unsigned)d_in.width),
              justEnoughBlocks(tpb.y, (unsigned)d_in.height));

    // H2D
    d_in.vec = h_in;
    // Kernel
    keep_index_matching_value_kernel<Tdata, Tidx><<<blks, tpb>>>(
        d_in.cimage(), (Tdata)1, d_out.data().get(), d_count.data().get());

    // D2H
    h_count = d_count;
    h_out = d_out;

    nvtxRangePop();

    // Reset
    d_count[0] = 0;
  }

  // ========== Use multiple streams ===========
  constexpr int numStreams = 4;
  containers::CudaStream streams[numStreams];

  // make separate count containers
  thrust::pinned_host_vector<unsigned> h_counts(numStreams);
  thrust::device_vector<unsigned> d_counts(numStreams);
  // for now split via rows, since it's simple
  int numRowsPerStream = d_in.height / numStreams + 1;

  for (int i = 0; i < 3; ++i) {
    nvtxRangePush("multistream");
    // we can keep the original input data containers, we just need to make
    // special image sizes and index appropriately
    for (int s = 0; s < numStreams; ++s) {
      auto &stream = streams[s];
      int startRow = s * numRowsPerStream;
      int startCol = 0;
      int tileRows = numRowsPerStream;
      int tileCols = d_in.width;

      // H2D
      int numRowsUsed = std::min(tileRows, d_in.height - startRow);
      cudaMemcpyAsync(d_in.vec.data().get() + (startRow * d_in.width),
                      h_in.data().get() + (startRow * d_in.width),
                      numRowsUsed * tileCols * sizeof(Tdata),
                      cudaMemcpyHostToDevice, stream());

      // Kernel
      const dim3 tpb(32, 4);
      dim3 blks(justEnoughBlocks(tpb.x, (unsigned)tileCols),
                justEnoughBlocks(tpb.y, (unsigned)tileRows));
      keep_index_matching_value_tile_kernel<Tdata, Tidx>
          <<<blks, tpb, 0, stream()>>>(
              d_in.cimage(), (Tdata)1, startRow, startCol, numRowsUsed,
              tileCols, d_out.data().get() + startRow * d_in.width,
              d_counts.data().get() + s);

      // D2H
      cudaMemcpyAsync(h_out.data().get() + startRow * d_in.width,
                      d_out.data().get() + startRow * d_in.width,
                      numRowsUsed * tileCols * sizeof(cuda_vec2_t<Tidx>),
                      cudaMemcpyDeviceToHost, stream());
      cudaMemcpyAsync(h_counts.data().get() + s, d_counts.data().get() + s,
                      sizeof(unsigned), cudaMemcpyDeviceToHost, stream());
    }

    // Synchronize everything
    for (int s = 0; s < numStreams; ++s) {
      streams[s].sync();
    }

    nvtxRangePop();

    // Reset
    thrust::fill(d_counts.begin(), d_counts.end(), 0);
  }

  // ==== Mapping host pinned pointer to kernel-usable pointer =======
  cuda_vec2_t<Tidx> *d_mapped_h_out;
  cudaError_t err =
      cudaHostGetDevicePointer(&d_mapped_h_out, h_out.data().get(), 0);
  if (err != cudaSuccess) {
    printf("Failed to get device pointer for pinned host vector: %s\n",
           cudaGetErrorName(err));
    return 1;
  } else {
    printf("Got device pointer for pinned host vector\n");
  }

  for (int i = 0; i < 3; ++i) {
    nvtxRangePush("multistream_with_mapped_host");
    // we can keep the original input data containers, we just need to make
    // special image sizes and index appropriately
    for (int s = 0; s < numStreams; ++s) {
      auto &stream = streams[s];
      int startRow = s * numRowsPerStream;
      int startCol = 0;
      int tileRows = numRowsPerStream;
      int tileCols = d_in.width;

      // H2D
      int numRowsUsed = std::min(tileRows, d_in.height - startRow);
      cudaMemcpyAsync(d_in.vec.data().get() + (startRow * d_in.width),
                      h_in.data().get() + (startRow * d_in.width),
                      numRowsUsed * tileCols * sizeof(Tdata),
                      cudaMemcpyHostToDevice, stream());

      // Kernel
      {
        const dim3 tpb(32, 4);
        dim3 blks(justEnoughBlocks(tpb.x, (unsigned)tileCols),
                  justEnoughBlocks(tpb.y, (unsigned)tileRows));
        keep_index_matching_value_tile_kernel<Tdata, Tidx>
            <<<blks, tpb, 0, stream()>>>(
                d_in.cimage(), (Tdata)1, startRow, startCol, numRowsUsed,
                tileCols, d_out.data().get() + startRow * d_in.width,
                d_counts.data().get() + s);
      }

      // D2H via mapped host pointer kernel copy
      {
        const unsigned tpb = 128;
        unsigned blks = justEnoughBlocks(tpb, (unsigned)numRowsUsed * tileCols);
        copy_compacted_list_via_mapped_host_pointer_kernel<Tdata, Tidx>
            <<<blks, tpb, 0, stream()>>>(
                d_out.data().get() + startRow * d_in.width,
                d_counts.data().get() + s, d_mapped_h_out);
      }
    }

    // Technically, can just copy the counts down altogether
    // just use first stream
    cudaMemcpyAsync(h_counts.data().get(), d_counts.data().get(),
                    numStreams * sizeof(unsigned), cudaMemcpyDeviceToHost,
                    streams[0]());

    // Synchronize everything
    for (int s = 0; s < numStreams; ++s) {
      streams[s].sync();
    }

    nvtxRangePop();

    // Reset
    thrust::fill(d_counts.begin(), d_counts.end(), 0);
  }

  return 0;
}
