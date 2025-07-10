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

template <typename Tdata, typename Tsum, typename Tcalc, typename Tsize>
__global__ void minmax_hold_from_section_medians_thresholds_kernel(
    Tdata *d_data, Tsize dataLengthPerSection, Tsize numSections,
    const Tdata *sectionMedians, Tsum *d_sumSections, Tsum *d_sumSqSections,
    Tsize *validSectionSizes, const Tcalc minmaxStdFactor) {
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
  Tdata data = d_data[sectionIdx * dataLengthPerSection + idx];
  if (data > upperThreshold) {
    d_data[sectionIdx * dataLengthPerSection + idx] = (Tdata)upperThreshold;
  } else if (data < lowerThreshold) {
    d_data[sectionIdx * dataLengthPerSection + idx] = (Tdata)lowerThreshold;
  }
}
