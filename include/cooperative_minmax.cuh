#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/**
 * @brief Minimises values across a thread group.
 *
 * @tparam T Type of values
 * @param g Thread group; usually a block or a warp
 * @param temp Temporary workspace, usually shared memory, same size as the
 *             thread group; used for holding values
 * @param tempIdx Temporary workspace, usually shared memory, same size as the
 *                thread group; used for holding indices
 * @param val Value for each thread in the group
 * @return Minimum value (only in thread 0), minimum index is in tempIdx[0]
 */
template <typename T>
__device__ T reduce_min(cg::thread_group g, T *temp, unsigned int *tempIdx,
                        T val) {
  int lane = g.thread_rank();

  for (int i = g.size() / 2; i > 0; i /= 2) {
    temp[lane] = val;
    tempIdx[lane] = lane;
    g.sync(); // wait for all threads to store
    if (lane < i) {
      if (val > temp[lane + i]) {
        val = temp[lane + i];
        tempIdx[lane] = lane + i;
      }
    }
    g.sync(); // wait for all threads to load
  }
  return val; // note: only thread 0 returns minimum
  // note: minIdx is in tempIdx[0];
}
