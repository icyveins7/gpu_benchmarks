#pragma once

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
 * @tparam Tdata
 * @tparam Tsum
 * @tparam Tcalc
 * @param d_data
 * @param maxSectionSize
 * @param sectionSizes
 * @param numSections
 * @param d_sumSections
 * @param d_sumSqSections
 * @param stdFactor
 * @param invalidValue
 */
template <typename Tdata, typename Tsum, typename Tcalc, typename Tsize = int>
__global__ void remove_sigma_multiple_outlier_removal_sectioned_kernel(
    Tdata *d_data, Tsize maxSectionSize, Tsize *sectionSizes,
    Tsize *prevSectionSizes, Tsize numSections, Tsum *d_sumSections,
    Tsum *d_sumSqSections, Tcalc stdFactor,
    Tdata invalidValue = cuda::std::numeric_limits<Tdata>::max()) {
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

  // Then define the associated mean and variance values
  Tcalc actualReciprocalSectionSize = 1.0 / sectionSizes[sectionIdx];
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
    atomicAggDec(&sectionSizes[sectionIdx]);
    // This should be ready for the next iteration, since the total sums now
    // reflect the newly invalidated elements, and so does the counter
  }
}
