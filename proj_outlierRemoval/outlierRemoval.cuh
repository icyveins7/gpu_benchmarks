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
    Tdata *d_data, Tsize maxSectionSize, Tsize *sectionSizes, Tsize numSections,
    Tsum *d_sumSections, Tsum *d_sumSqSections, Tcalc stdFactor,
    Tdata invalidValue = cuda::std::numeric_limits<Tdata>::max()) {
  static_assert(cuda::std::is_floating_point_v<Tcalc>,
                "Tcalc must be float or double");
  // Get global index
  Tsize idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numSections * maxSectionSize) {
    return;
  }

  // Get section index
  Tsize sectionIdx = idx / maxSectionSize;

  // Read associated sum and sum squared values
  Tcalc sectionSum = d_sumSections[sectionIdx];
  Tcalc sectionSumSq = d_sumSqSections[sectionIdx];

  // Then the associated mean and variance
  Tcalc actualReciprocalSectionSize = 1.0 / sectionSizes[sectionIdx];
  Tcalc mean =
      sectionSum * actualReciprocalSectionSize; // TODO: ignore invalid values
  Tcalc variance = sectionSumSq * actualReciprocalSectionSize - mean * mean;
  Tcalc std = sqrt(variance);

  // Read data and compare
  Tdata data = d_data[idx];
  if (data != invalidValue) {
    // Cast to floating point template type
    Tcalc cdata = (Tcalc)data;
    // Calculate the abs differences
    Tcalc diff = abs(cdata - mean);
    // Rewrite if it exceeds
    if (diff > stdFactor * std) {
      d_data[idx] = invalidValue;
    }
  }
}
