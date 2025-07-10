#pragma once

#include "atomic_extensions.cuh"
#include <cuda/std/limits>

/*
 * Design scope: perform multiple iterations over a set of data on device, where
 * the statistics are re-calculated at each iteration and outliers are removed
 * until a satisfactory criterion, usually dependent on the statistics
 * themselves.
 */

/**
 * @brief Removes outliers by overwriting with indicated invalid values.
 * Outliers are defined by exceeding a multiple of the standard deviation.
 *
 * This kernel keeps track of the changing valid size of each section by using
 * 3 separate arrays (prev, curr, next). At the start of a given iteration,
 * it is expected that currSize == nextSize, since the kernel will only modify
 * 'nextSize' by decrementing the valid count, rather than re-counting from
 * scratch. The kernel will not compute for the section if 'currSize' ==
 * 'prevSize'. Essentially, this requires additional steps after this kernel, to
 * set, in order,
 *   1) prevSize = currSize
 *   2) currSize = nextSize
 * for each section.
 *
 * The sum and sum of squares are not used in for determining the stopping
 * criterion, so there is only 1 copy for each section, which is modified
 * in-place.
 *
 * @tparam Tdata Data type
 * @tparam Tsum Data type of sum and sum of squares
 * @tparam Tcalc Data type to use in internal calculations
 * @param d_data Input data, dim maxSectionSize x numSections; this is modified
 * in-place to mark invalid values (outliers)
 * @param maxSectionSize Maximum number of elements in each section
 * @param currSectionSizes Number of valid elements in each section in current
 * iteration, dim 1 x numSections
 * @param prevSectionSizes Number of valid elements in each section in previous
 * iteration, dim 1 x numSections; this is used to determine whether to proceed
 * @param nextSectionSizes Number of valid elements in each section in next
 * iteration, dim 1 x numSections; this is filled in this kernel, but it is
 * expected to be initialized to be EQUAL to currSectionSizes
 * @param numSections Total number of sections
 * @param d_sumSections Sum of elements in each section
 * @param d_sumSqSections Sum of squares of elements in each section
 * @param stdFactor Factor to multiply standard deviation by, for outlier
 * marking
 * @param invalidValue Value to replace outliers with, and to ignore in input
 */
template <typename Tdata, typename Tsum, typename Tcalc, typename Tsize = int>
__global__ void remove_sigma_multiple_outlier_removal_sectioned_kernel(
    Tdata *d_data, const Tsize maxSectionSize, const Tsize *currSectionSizes,
    const Tsize *prevSectionSizes, Tsize *nextSectionSizes,
    const Tsize numSections, Tsum *d_sumSections, Tsum *d_sumSqSections,
    const Tcalc stdFactor,
    const Tdata invalidValue = cuda::std::numeric_limits<Tdata>::max()) {
  static_assert(cuda::std::is_floating_point_v<Tcalc>,
                "Tcalc must be float or double");
  // Get section index
  Tsize sectionIdx = blockIdx.y;
  if (sectionIdx >= numSections) {
    return;
  }
  // Read associated sum and sum squared values
  Tcalc sectionSum = d_sumSections[sectionIdx];
  Tcalc sectionSumSq = d_sumSqSections[sectionIdx];

  // Read the latest and previous valid section size
  Tsize currentSectionSize = currSectionSizes[sectionIdx];
  Tsize prevSectionSize = prevSectionSizes[sectionIdx];
  // Stop here if the section size did not change
  if (currentSectionSize == prevSectionSize) {
    return;
  }

  // Then define the associated mean and variance values
  Tcalc actualReciprocalSectionSize = 1.0 / currSectionSizes[sectionIdx];
  Tcalc mean = sectionSum * actualReciprocalSectionSize;
  Tcalc variance = sectionSumSq * actualReciprocalSectionSize - mean * mean;
  Tcalc std = sqrt(variance);

  // Get index inside section
  Tsize idx = blockIdx.x * blockDim.x + threadIdx.x;
  Tdata data;
  if (idx < maxSectionSize)
    data = d_data[sectionIdx * maxSectionSize + idx];
  else
    data = invalidValue;

  // Cast to floating point template type
  Tcalc cdata = (Tcalc)data;
  // Calculate the abs differences in floating point
  Tcalc diff = abs(cdata - mean);
  // Determine if it's a new outlier (should NOT be an invalid value)
  bool isOutlier = (diff > stdFactor * std) && (data != invalidValue);
  // Define adjustments for every thread in the warp
  Tsum sumAdj = isOutlier ? -static_cast<Tsum>(data) : 0;
  Tsum sumSqAdj =
      isOutlier ? -static_cast<Tsum>(data) * static_cast<Tsum>(data) : 0;

  // Warp-aggregatic atomic adjustment of the sum and sum squared values
  atomicAggIncSum(&d_sumSections[sectionIdx], sumAdj);
  atomicAggIncSum(&d_sumSqSections[sectionIdx], sumSqAdj);

  // Overwrite if it is an outlier
  if (isOutlier) {
    d_data[sectionIdx * maxSectionSize + idx] = invalidValue;

    // Reduce the count of valid values
    atomicAggDec(&nextSectionSizes[sectionIdx]);
    // This should be ready for the next iteration, since the total sums now
    // reflect the newly invalidated elements, and so does the counter
  }
}
