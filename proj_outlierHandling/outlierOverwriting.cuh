#pragma once

#include <cuda/std/limits>

/*
 * Design scope: this depends on the other header's implementations, where
 * outliers are removed from sections of data. Here the remaining valid elements
 * in each sections should be used in determining thresholds for a larger
 * population of data i.e. the sections of data each represent a subset of a
 * larger set of data:
 *   1) Subsample original data Tsizeo a section (done elsewhere)
 *   2) Calculate statistics and remove outliers from the section (done in
 * outlierRemoval)
 *   3) Use section's statistics to determine thresholds for original data
 *   4) Overwrite outliers (beyond thresholds) in original data
 */

/**
 * @brief Performs a min/max-hold by overwriting elements in the data that
 * extend beyond median +/- minmaxStdFactor*std, for each section.
 *
 * Both the median value and the std are assumed to be estimated elsewhere, and
 * these statistics are used as representative values when overwriting values in
 * the larger, original data. See medians.cuh for more details.
 *
 * Original use case:
 *   1) Original data is subsampled, per section
 *   2) Subsampled sections have outliers removed iteratively (invalid elements
 * overwritten, see outlierRemoval.cuh)
 *   3) Remaining subsampled sections have medians estimated (see medians.cuh)
 *   4) This kernel is called, using the previous kernels' output to determine
 * thresholds, and overwrite those that exceed it
 *
 * @tparam Tdata Data type for input/medians
 * @tparam Tsum Data type for the sum and sum of squares
 * @tparam Tcalc Data type used for internal calculations
 * @tparam Tsize Data type for sizes like counters
 * @param d_data Larger, original input data (still has sections)
 * @param dataLengthPerSection Length of each section in d_data
 * @param numSections Number of sections
 * @param sectionMedians Medians of each section to use for thresholds
 * @param d_sumSections Sum per section, used in mean/std calculations
 * @param d_sumSqSections Sum of squares per section, similar to sum
 * @param validSectionSizes Counter of valid elements used in the sum/sumSq
 * @param minmaxStdFactor Factor to multiply standard deviation by, for
 * thresholds
 */
template <typename Tdata, typename Tsum, typename Tcalc, typename Tsize>
__global__ void minmax_hold_from_section_medians_thresholds_kernel(
    Tdata *d_data, Tsize dataLengthPerSection, Tsize numSections,
    const Tdata *sectionMedians, Tsum *d_sumSections, Tsum *d_sumSqSections,
    Tsize *validSectionSizes, const Tcalc minmaxStdFactor) {
  static_assert(cuda::std::is_floating_point_v<Tcalc>,
                "Tcalc must be float or double");
  // Same as before, assume we operate on each section via blockIdx.y
  Tsize sectionIdx = blockIdx.y;
  if (sectionIdx >= numSections) {
    return;
  }

  // Calculate section mean/std
  Tsum sectionSum = d_sumSections[sectionIdx];
  Tsum sectionSumSq = d_sumSqSections[sectionIdx];

  Tcalc actualReciprocalSectionSize = 1.0 / validSectionSizes[sectionIdx];
  Tcalc mean = sectionSum * actualReciprocalSectionSize;
  Tcalc variance = sectionSumSq * actualReciprocalSectionSize - mean * mean;
  Tcalc std = sqrt(variance);

  // Calculate the upper and lower thresholds
  Tdata sectionMedian = sectionMedians[sectionIdx];
  Tcalc upperThreshold = (Tcalc)sectionMedian + minmaxStdFactor * std;
  Tcalc lowerThreshold = (Tcalc)sectionMedian - minmaxStdFactor * std;

  // Now read and replace the data
  Tsize idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dataLengthPerSection) {
    Tdata data = d_data[sectionIdx * dataLengthPerSection + idx];
    if (data > upperThreshold) {
      d_data[sectionIdx * dataLengthPerSection + idx] = (Tdata)upperThreshold;
    } else if (data < lowerThreshold) {
      d_data[sectionIdx * dataLengthPerSection + idx] = (Tdata)lowerThreshold;
    }
  }
}
