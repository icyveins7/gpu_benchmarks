#pragma once

/*
 * See with scan_short_words.cu.
 * This does appear to be slightly faster 1.067ms vs 1.224ms, for 16384 x 16384.
 * One block per row, 256 threads per block.
 *
 * Serves as a good guide for future kernels that can assure full utilization
 * while assigning one block per row.
 */

#include "containers/image.cuh"
#include "cub/cub.cuh"

#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda_pipeline.h>

#include <cuda/std/version>
// For older CUDA versions before 12.9(?)
#if CUDA_VERSION < 12090
#include <thrust/functional.h>
template <typename T> using MaxOp = thrust::maximum<T>;
#else
template <typename T> using MaxOp = ::cuda::maximum<T>;
#endif

template <typename T, typename Tpacked, int NUM_THREADS,
          typename ScanOp = MaxOp<T>>
__global__ void packedwords_rowwise_inclusive_scan_kernel(
    const containers::Image<const T> input, containers::Image<T> output,
    ScanOp op = MaxOp<T>{}) {
  constexpr int PACKSIZE = sizeof(Tpacked) / sizeof(T);

  // Each block handles one row
  if (blockIdx.x >= input.height)
    return;

  using BlockScan = cub::BlockScan<T, NUM_THREADS>;
  __shared__ typename BlockScan::TempStorage tempStorage;
  __shared__ Tpacked s_data[NUM_THREADS]; // +1 in case of incomplete word
  T* s_data_small = (T*)s_data;

  int offset = 0;
  T aggregate = 0; // initial value aggregate
  while (offset < input.width) {
    // Define number of actual values (unpacked)
    int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - offset);

    Tpacked* inputPackedOffset = (Tpacked*)&input.at(blockIdx.x, offset);
    Tpacked* outputPackedOffset = (Tpacked*)&output.at(blockIdx.x, offset);

    // Load
    if (numThisIter % PACKSIZE == 0) {
      // Load full words, one packed word per thread
      for (int t = threadIdx.x; t < numThisIter / PACKSIZE; t += blockDim.x) {
        s_data[t] = inputPackedOffset[t];
      }
    } else {
      // Load normally as unpacked values
      for (int t = threadIdx.x; t < numThisIter; t += blockDim.x) {
        s_data_small[t] = input.at(blockIdx.x, offset + t);
      }
    }
    // Handle first thread inclusion (from last iteration)
    if (offset != 0 && threadIdx.x == 0) {
      s_data_small[0] = op(aggregate, s_data_small[0]);
    }
    __syncthreads();

    // Perform inclusive scan repeatedly as unpacked values
    // NOTE: the aggregate on the final scan is not correct if it the iteration
    // does not fully utilize the entire shared memory length, but this isn't
    // important anyway since the aggregate would no longer be needed after.
    int numRepeatScans =
        numThisIter / NUM_THREADS + (numThisIter % NUM_THREADS == 0 ? 0 : 1);
    for (int i = 0; i < numRepeatScans; i++) {
      BlockScan(tempStorage)
          .InclusiveScan(s_data_small[i * NUM_THREADS + threadIdx.x],
                         s_data_small[i * NUM_THREADS + threadIdx.x], op,
                         aggregate);
      // fix next scan's first value using aggregate
      if (i < numRepeatScans - 1 && threadIdx.x == 0) {
        s_data_small[(i + 1) * NUM_THREADS] =
            op(aggregate, s_data_small[(i + 1) * NUM_THREADS]);
      }
      __syncthreads(); // for reuse of tempStorage
    }

    // Write out
    if (numThisIter % PACKSIZE == 0) {
      // Store full words, one packed word per thread
      for (int t = threadIdx.x; t < numThisIter / PACKSIZE; t += blockDim.x) {
        outputPackedOffset[t] = s_data[t];
      }
    } else {
      // Store normally as unpacked values
      for (int t = threadIdx.x; t < numThisIter; t += blockDim.x) {
        output.at(blockIdx.x, offset + t) = s_data_small[t];
      }
    }

    // sync before next iteration, also needed to reuse BlockScan
    offset += numThisIter;
    __syncthreads();
  }
}

// Disables `pipeline_shared_state` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

// NOTE: actually very slow, and drops to 66% occupancy
// due to 33->50+ reg per thread
template <typename T, typename Tpacked, int NUM_THREADS,
          typename ScanOp = MaxOp<T>>
__global__ void packedwords_rowwise_inclusive_scan_pipelined_kernel(
    const containers::Image<const T> input, containers::Image<T> output,
    ScanOp op = MaxOp<T>{}) {
  constexpr int PACKSIZE = sizeof(Tpacked) / sizeof(T);

  // Each block handles one row
  if (blockIdx.x >= input.height)
    return;

  auto group = cooperative_groups::this_thread_block();

  using BlockScan = cub::BlockScan<T, NUM_THREADS>;
  __shared__ typename BlockScan::TempStorage tempStorage;
  __shared__ Tpacked
      s[NUM_THREADS * 2]; // 2 separate workspaces for async copies

  int prod_offset = 0, cons_offset = 0;
  T aggregate = 0; // initial value aggregate

  // Create pipeline
  constexpr auto scope = cuda::thread_scope_block;
  constexpr auto stages_count = 2;
  __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
  auto pipeline = cuda::make_pipeline(group, &shared_state);

  // Prime pipeline (read first batch). Scope braces are for readability.
  {
    pipeline.producer_acquire();
    int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - prod_offset);

    if (numThisIter % PACKSIZE == 0) {
      cuda::memcpy_async(group, s, (Tpacked*)&input.at(blockIdx.x, prod_offset),
                         sizeof(Tpacked) * (numThisIter / PACKSIZE), pipeline);
    } else {
      cuda::memcpy_async(group, (T*)s, &input.at(blockIdx.x, prod_offset),
                         sizeof(T) * numThisIter, pipeline);
    }

    prod_offset += numThisIter;
    pipeline.producer_commit();
  }

  // Pipeline copy/compute
  int pipeline_prod_iter = 1;
  while (prod_offset < input.width) {
    {
      pipeline.producer_acquire();
      int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - prod_offset);

      if (numThisIter % PACKSIZE == 0) {
        cuda::memcpy_async(group, s + (pipeline_prod_iter % 2) * NUM_THREADS,
                           (Tpacked*)&input.at(blockIdx.x, prod_offset),
                           sizeof(Tpacked) * (numThisIter / PACKSIZE),
                           pipeline);
      } else {
        cuda::memcpy_async(group,
                           (T*)(s + (pipeline_prod_iter % 2) * NUM_THREADS),
                           &input.at(blockIdx.x, prod_offset),
                           sizeof(T) * numThisIter, pipeline);
      }

      prod_offset += numThisIter;
      pipeline.producer_commit();
    }
    // ----------------------------------------------------
    // end producer N, start consumer N - 1
    // ----------------------------------------------------
    {
      pipeline.consumer_wait();
      // NOTE: uses cons_offset, separate increment
      int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - cons_offset);

      // Grab the previous buffer that has now filled
      Tpacked* s_data_consumer =
          s + ((pipeline_prod_iter - 1) % 2) * NUM_THREADS;
      T* s_data_small_consumer = (T*)s_data_consumer;

      // Handle first thread inclusion (from last iteration)
      if (cons_offset != 0 && threadIdx.x == 0) {
        s_data_small_consumer[0] = op(aggregate, s_data_small_consumer[0]);
      }
      __syncthreads();

      // Perform inclusive scan repeatedly as unpacked values
      // NOTE: the aggregate on the final scan is not correct if it the
      // iteration does not fully utilize the entire shared memory length, but
      // this isn't important anyway since the aggregate would no longer be
      // needed after.
      int numRepeatScans =
          numThisIter / NUM_THREADS + (numThisIter % NUM_THREADS == 0 ? 0 : 1);
      for (int i = 0; i < numRepeatScans; i++) {
        BlockScan(tempStorage)
            .InclusiveScan(s_data_small_consumer[i * NUM_THREADS + threadIdx.x],
                           s_data_small_consumer[i * NUM_THREADS + threadIdx.x],
                           op, aggregate);
        // fix next scan's first value using aggregate
        if (i < numRepeatScans - 1 && threadIdx.x == 0) {
          s_data_small_consumer[(i + 1) * NUM_THREADS] =
              op(aggregate, s_data_small_consumer[(i + 1) * NUM_THREADS]);
        }
        __syncthreads(); // for reuse of tempStorage
      }

      // Write out
      Tpacked* outputPackedOffset =
          (Tpacked*)&output.at(blockIdx.x, cons_offset);
      if (numThisIter % PACKSIZE == 0) {
        // Store full words, one packed word per thread
        for (int t = threadIdx.x; t < numThisIter / PACKSIZE; t += blockDim.x) {
          outputPackedOffset[t] = s_data_consumer[t];
        }
      } else {
        // Store normally as unpacked values
        for (int t = threadIdx.x; t < numThisIter; t += blockDim.x) {
          output.at(blockIdx.x, cons_offset + t) = s_data_small_consumer[t];
        }
      }

      cons_offset += numThisIter;
      pipeline.consumer_release();
    }
    // ----------------------------------------------------
    // end consumer N - 1
    // ----------------------------------------------------

    // increment before next producer/consumer loop
    pipeline_prod_iter++;
  }

  // Drain pipeline (code is copied; ideally would refactor)
  {
    pipeline.consumer_wait();
    // NOTE: uses cons_offset, separate increment
    int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - cons_offset);

    // Grab the previous buffer that has now filled
    Tpacked* s_data_consumer = s + ((pipeline_prod_iter - 1) % 2) * NUM_THREADS;
    T* s_data_small_consumer = (T*)s_data_consumer;

    // Handle first thread inclusion (from last iteration)
    if (cons_offset != 0 && threadIdx.x == 0) {
      s_data_small_consumer[0] = op(aggregate, s_data_small_consumer[0]);
    }
    __syncthreads();

    // Perform inclusive scan repeatedly as unpacked values
    // NOTE: the aggregate on the final scan is not correct if it the
    // iteration does not fully utilize the entire shared memory length, but
    // this isn't important anyway since the aggregate would no longer be
    // needed after.
    int numRepeatScans =
        numThisIter / NUM_THREADS + (numThisIter % NUM_THREADS == 0 ? 0 : 1);
    for (int i = 0; i < numRepeatScans; i++) {
      BlockScan(tempStorage)
          .InclusiveScan(s_data_small_consumer[i * NUM_THREADS + threadIdx.x],
                         s_data_small_consumer[i * NUM_THREADS + threadIdx.x],
                         op, aggregate);
      // fix next scan's first value using aggregate
      if (i < numRepeatScans - 1 && threadIdx.x == 0) {
        s_data_small_consumer[(i + 1) * NUM_THREADS] =
            op(aggregate, s_data_small_consumer[(i + 1) * NUM_THREADS]);
      }
      __syncthreads(); // for reuse of tempStorage
    }

    // Write out
    Tpacked* outputPackedOffset = (Tpacked*)&output.at(blockIdx.x, cons_offset);
    if (numThisIter % PACKSIZE == 0) {
      // Store full words, one packed word per thread
      for (int t = threadIdx.x; t < numThisIter / PACKSIZE; t += blockDim.x) {
        outputPackedOffset[t] = s_data_consumer[t];
      }
    } else {
      // Store normally as unpacked values
      for (int t = threadIdx.x; t < numThisIter; t += blockDim.x) {
        output.at(blockIdx.x, cons_offset + t) = s_data_small_consumer[t];
      }
    }
    pipeline.consumer_release();
  }
}

#pragma nv_diag_suppress static_var_with_dynamic_init

// NOTE: by itself, barriers have no difference to the pipeline;
// here I attempted a few more optimizations to try to reduce register use
// but just look at the final cprimitive version for the best
template <typename T, typename Tpacked, int NUM_THREADS,
          typename ScanOp = MaxOp<T>>
__global__ void packedwords_rowwise_inclusive_scan_barrier_kernel(
    const containers::Image<const T> input, containers::Image<T> output,
    ScanOp op = MaxOp<T>{}) {
  constexpr int PACKSIZE = sizeof(Tpacked) / sizeof(T);

  // Each block handles one row
  if (blockIdx.x >= input.height)
    return;

  auto group = cooperative_groups::this_thread_block();

  using BlockScan = cub::BlockScan<T, NUM_THREADS>;
  __shared__ typename BlockScan::TempStorage tempStorage;
  __shared__ Tpacked
      s[NUM_THREADS * 2]; // 2 separate workspaces for async copies
  __shared__ cuda::barrier<cuda::thread_scope_block> bars[2];

  if (threadIdx.x < 2)
    init(&bars[threadIdx.x], blockDim.x);
  __syncthreads();

  int prod_offset = 0, cons_offset = 0;
  T aggregate = 0; // initial value aggregate
  int buf = 0;     // consumer buffer index; producer uses buf ^ 1

  // Prime pipeline (read first batch into buf 0)
  {
    int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - prod_offset);

    if (numThisIter % PACKSIZE == 0) {
      cuda::memcpy_async(group, s, (Tpacked*)&input.at(blockIdx.x, prod_offset),
                         sizeof(Tpacked) * (numThisIter / PACKSIZE), bars[0]);
    } else {
      cuda::memcpy_async(group, (T*)s, &input.at(blockIdx.x, prod_offset),
                         sizeof(T) * numThisIter, bars[0]);
    }

    prod_offset += numThisIter;
  }

  // Pipeline copy/compute
  while (prod_offset < input.width) {
    // Producer: load next batch into buf ^ 1
    {
      int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - prod_offset);

      if (numThisIter % PACKSIZE == 0) {
        cuda::memcpy_async(group, s + (buf ^ 1) * NUM_THREADS,
                           (Tpacked*)&input.at(blockIdx.x, prod_offset),
                           sizeof(Tpacked) * (numThisIter / PACKSIZE),
                           bars[buf ^ 1]);
      } else {
        cuda::memcpy_async(group, (T*)(s + (buf ^ 1) * NUM_THREADS),
                           &input.at(blockIdx.x, prod_offset),
                           sizeof(T) * numThisIter, bars[buf ^ 1]);
      }

      prod_offset += numThisIter;
    }
    // ----------------------------------------------------
    // end producer N, start consumer N - 1
    // ----------------------------------------------------
    {
      // NOTE: uses cons_offset, separate increment
      int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - cons_offset);

      Tpacked* s_data_consumer = s + buf * NUM_THREADS;
      T* s_data_small_consumer = (T*)s_data_consumer;

      bars[buf].arrive_and_wait(); // waits for async copy + syncs all threads

      // Handle first thread inclusion (from last iteration)
      if (cons_offset != 0 && threadIdx.x == 0) {
        s_data_small_consumer[0] = op(aggregate, s_data_small_consumer[0]);
      }
      __syncthreads();

      // Perform inclusive scan repeatedly as unpacked values
      int numRepeatScans =
          numThisIter / NUM_THREADS + (numThisIter % NUM_THREADS == 0 ? 0 : 1);
      for (int i = 0; i < numRepeatScans; i++) {
        BlockScan(tempStorage)
            .InclusiveScan(s_data_small_consumer[i * NUM_THREADS + threadIdx.x],
                           s_data_small_consumer[i * NUM_THREADS + threadIdx.x],
                           op, aggregate);
        // fix next scan's first value using aggregate
        if (i < numRepeatScans - 1 && threadIdx.x == 0) {
          s_data_small_consumer[(i + 1) * NUM_THREADS] =
              op(aggregate, s_data_small_consumer[(i + 1) * NUM_THREADS]);
        }
        __syncthreads(); // for reuse of tempStorage
      }

      // Write out
      Tpacked* outputPackedOffset =
          (Tpacked*)&output.at(blockIdx.x, cons_offset);
      if (numThisIter % PACKSIZE == 0) {
        // Store full words, one packed word per thread
        for (int t = threadIdx.x; t < numThisIter / PACKSIZE; t += blockDim.x) {
          outputPackedOffset[t] = s_data_consumer[t];
        }
      } else {
        // Store normally as unpacked values
        for (int t = threadIdx.x; t < numThisIter; t += blockDim.x) {
          output.at(blockIdx.x, cons_offset + t) = s_data_small_consumer[t];
        }
      }

      cons_offset += numThisIter;
      buf ^= 1;
    }
    // ----------------------------------------------------
    // end consumer N - 1
    // ----------------------------------------------------
  }

  // Drain pipeline (code is copied; ideally would refactor)
  {
    // NOTE: uses cons_offset, separate increment
    int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - cons_offset);

    Tpacked* s_data_consumer = s + buf * NUM_THREADS;
    T* s_data_small_consumer = (T*)s_data_consumer;

    bars[buf].arrive_and_wait(); // waits for async copy + syncs all threads

    // Handle first thread inclusion (from last iteration)
    if (cons_offset != 0 && threadIdx.x == 0) {
      s_data_small_consumer[0] = op(aggregate, s_data_small_consumer[0]);
    }
    __syncthreads();

    // Perform inclusive scan repeatedly as unpacked values
    int numRepeatScans =
        numThisIter / NUM_THREADS + (numThisIter % NUM_THREADS == 0 ? 0 : 1);
    for (int i = 0; i < numRepeatScans; i++) {
      BlockScan(tempStorage)
          .InclusiveScan(s_data_small_consumer[i * NUM_THREADS + threadIdx.x],
                         s_data_small_consumer[i * NUM_THREADS + threadIdx.x],
                         op, aggregate);
      // fix next scan's first value using aggregate
      if (i < numRepeatScans - 1 && threadIdx.x == 0) {
        s_data_small_consumer[(i + 1) * NUM_THREADS] =
            op(aggregate, s_data_small_consumer[(i + 1) * NUM_THREADS]);
      }
      __syncthreads(); // for reuse of tempStorage
    }

    // Write out
    Tpacked* outputPackedOffset = (Tpacked*)&output.at(blockIdx.x, cons_offset);
    if (numThisIter % PACKSIZE == 0) {
      // Store full words, one packed word per thread
      for (int t = threadIdx.x; t < numThisIter / PACKSIZE; t += blockDim.x) {
        outputPackedOffset[t] = s_data_consumer[t];
      }
    } else {
      // Store normally as unpacked values
      for (int t = threadIdx.x; t < numThisIter; t += blockDim.x) {
        output.at(blockIdx.x, cons_offset + t) = s_data_small_consumer[t];
      }
    }
  }
}

// NOTE: this is the best performing version as the C pipeline primitives
// do not/barely consume extra registers. Coupled with reducing a lot of
// 'extra' variables the register count was pushed down to 40, achieving 100%
// occupancy. However, this ended up at like 1-2% slower than the original
// version at the top. Hence the pipelines/async memcpys are not useful in this
// scenario.
template <typename T, typename Tpacked, int NUM_THREADS,
          typename ScanOp = MaxOp<T>>
__global__ void packedwords_rowwise_inclusive_scan_cpipelineprimitive_kernel(
    const containers::Image<const T> input, containers::Image<T> output,
    ScanOp op = MaxOp<T>{}) {
  constexpr int PACKSIZE = sizeof(Tpacked) / sizeof(T);

  // Each block handles one row
  if (blockIdx.x >= input.height)
    return;

  using BlockScan = cub::BlockScan<T, NUM_THREADS>;
  __shared__ typename BlockScan::TempStorage tempStorage;
  __shared__ Tpacked
      s[NUM_THREADS * 2]; // 2 separate workspaces for async copies

  int cons_offset = 0;
  T aggregate = 0; // initial value aggregate
  int buf = 0;     // consumer buffer index; producer uses buf ^ 1

  // Helper lambda to issue one batch of async copies into a destination buffer.
  // Falls back to a synchronous copy for the unpacked tail (sizeof(T) < 4,
  // which __pipeline_memcpy_async does not support), then commits a group
  // either way to keep __pipeline_wait_prior counts consistent.
  auto issue_and_commit = [&](Tpacked* dst, int offset, int numElems) {
    if (numElems % PACKSIZE == 0) {
      Tpacked* src = (Tpacked*)&input.at(blockIdx.x, offset);
      for (int t = threadIdx.x; t < numElems / PACKSIZE; t += blockDim.x)
        __pipeline_memcpy_async(dst + t, src + t, sizeof(Tpacked));
    } else {
      // Synchronous fallback: __pipeline_memcpy_async requires >= 4 bytes
      const T* src = &input.at(blockIdx.x, offset);
      T* dst_small = (T*)dst;
      for (int t = threadIdx.x; t < numElems; t += blockDim.x)
        dst_small[t] = src[t];
    }
    __pipeline_commit();
  };

  // Prime pipeline (read first batch into buf 0)
  {
    int numThisIter = min(NUM_THREADS * PACKSIZE, input.width);
    issue_and_commit(s, 0, numThisIter);
  }

  // Pipeline copy/compute.
  // prod_offset is always cons_offset + NUM_THREADS * PACKSIZE, so it is
  // computed inline rather than tracked as a separate register.
  while (cons_offset + NUM_THREADS * PACKSIZE < input.width) {
    // Producer: load next batch into buf ^ 1
    {
      int prod_at = cons_offset + NUM_THREADS * PACKSIZE;
      int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - prod_at);
      issue_and_commit(s + (buf ^ 1) * NUM_THREADS, prod_at, numThisIter);
    }
    // ----------------------------------------------------
    // end producer N, start consumer N - 1
    // ----------------------------------------------------
    {
      // NOTE: uses cons_offset, separate increment
      int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - cons_offset);

      Tpacked* s_data_consumer = s + buf * NUM_THREADS;
      T* s_data_small_consumer = (T*)s_data_consumer;

      __pipeline_wait_prior(1); // wait for all but the most-recent group
      __syncthreads();

      // Handle first thread inclusion (from last iteration)
      if (cons_offset != 0 && threadIdx.x == 0) {
        s_data_small_consumer[0] = op(aggregate, s_data_small_consumer[0]);
      }
      __syncthreads();

      // Perform inclusive scan repeatedly as unpacked values
      int numRepeatScans =
          numThisIter / NUM_THREADS + (numThisIter % NUM_THREADS == 0 ? 0 : 1);
      for (int i = 0; i < numRepeatScans; i++) {
        BlockScan(tempStorage)
            .InclusiveScan(s_data_small_consumer[i * NUM_THREADS + threadIdx.x],
                           s_data_small_consumer[i * NUM_THREADS + threadIdx.x],
                           op, aggregate);
        // fix next scan's first value using aggregate
        if (i < numRepeatScans - 1 && threadIdx.x == 0) {
          s_data_small_consumer[(i + 1) * NUM_THREADS] =
              op(aggregate, s_data_small_consumer[(i + 1) * NUM_THREADS]);
        }
        __syncthreads(); // for reuse of tempStorage
      }

      // Write out
      Tpacked* outputPackedOffset =
          (Tpacked*)&output.at(blockIdx.x, cons_offset);
      if (numThisIter % PACKSIZE == 0) {
        // Store full words, one packed word per thread
        for (int t = threadIdx.x; t < numThisIter / PACKSIZE; t += blockDim.x) {
          outputPackedOffset[t] = s_data_consumer[t];
        }
      } else {
        // Store normally as unpacked values
        for (int t = threadIdx.x; t < numThisIter; t += blockDim.x) {
          output.at(blockIdx.x, cons_offset + t) = s_data_small_consumer[t];
        }
      }

      cons_offset += numThisIter;
      buf ^= 1;
    }
    // ----------------------------------------------------
    // end consumer N - 1
    // ----------------------------------------------------
  }

  // Drain pipeline (code is copied; ideally would refactor)
  {
    // NOTE: uses cons_offset, separate increment
    int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - cons_offset);

    Tpacked* s_data_consumer = s + buf * NUM_THREADS;
    T* s_data_small_consumer = (T*)s_data_consumer;

    __pipeline_wait_prior(0); // wait for all remaining groups
    __syncthreads();

    // Handle first thread inclusion (from last iteration)
    if (cons_offset != 0 && threadIdx.x == 0) {
      s_data_small_consumer[0] = op(aggregate, s_data_small_consumer[0]);
    }
    __syncthreads();

    // Perform inclusive scan repeatedly as unpacked values
    int numRepeatScans =
        numThisIter / NUM_THREADS + (numThisIter % NUM_THREADS == 0 ? 0 : 1);
    for (int i = 0; i < numRepeatScans; i++) {
      BlockScan(tempStorage)
          .InclusiveScan(s_data_small_consumer[i * NUM_THREADS + threadIdx.x],
                         s_data_small_consumer[i * NUM_THREADS + threadIdx.x],
                         op, aggregate);
      // fix next scan's first value using aggregate
      if (i < numRepeatScans - 1 && threadIdx.x == 0) {
        s_data_small_consumer[(i + 1) * NUM_THREADS] =
            op(aggregate, s_data_small_consumer[(i + 1) * NUM_THREADS]);
      }
      __syncthreads(); // for reuse of tempStorage
    }

    // Write out
    Tpacked* outputPackedOffset = (Tpacked*)&output.at(blockIdx.x, cons_offset);
    if (numThisIter % PACKSIZE == 0) {
      // Store full words, one packed word per thread
      for (int t = threadIdx.x; t < numThisIter / PACKSIZE; t += blockDim.x) {
        outputPackedOffset[t] = s_data_consumer[t];
      }
    } else {
      // Store normally as unpacked values
      for (int t = threadIdx.x; t < numThisIter; t += blockDim.x) {
        output.at(blockIdx.x, cons_offset + t) = s_data_small_consumer[t];
      }
    }
  }
}
