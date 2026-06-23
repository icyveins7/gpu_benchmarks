/*
This is to test detections of what I call 1-2-1 segments.

Based on a flag array, a segment is valid if in between two '1's there is at
least a single '2'.

Every segment should be returned. We also include periodic boundaries, so
segments can wrap-around.

We test using rows from an image to include keyed scans.

The idea is to drag the left boundary column index along throughout the segment,
along with the largest flag seen since then. So there are 2 'operations':

1) Reset: when hitting a new boundary (flag == 1), set the scan value to be
{column index, 1}. This corresponds to the flag itself.

2) Extend: for any other case, extend by keeping the largest flag since then;
this ensures that as long as there is a single '2' flag, we will drag it to the
right boundary.

NOTE: there are several ways to do this, and a common trip-up is that you may
write something that is *not associative*: that is, f(f(a,b),c) != f(a,f(b,c)).
GPU parallel scans evaluate the operator in a tree, so non-associative operators
produce wrong results.

The specific failure: if the reset condition is `b.flag == 1`, the 'max' in the
extend branch can overwrite a flag of 1 with a 2, silently erasing the reset
signal. Example with a={-1,0}, b={col=0,flag=1}, c={col=-1,flag=2}:

  Left grouping:  f(f(a,b), c)
    f(a, b): b.flag==1, reset  -> {0, 1}
    f({0,1}, c): extend        -> {0, max(1,2)} = {0, 2}   <- idx=0 (correct)

  Right grouping: f(a, f(b,c))
    f(b, c): c.flag==2, extend -> {0, max(1,2)} = {0, 2}   <- flag is now 2!
    f(a, {0,2}): 2!=1, extend  -> {-1, max(0,2)} = {-1, 2} <- idx=-1 (wrong)

The fix: store the reset signal in `idx` (>=0 means reset, -1 means not).
The extend branch copies idx from `a` unchanged -- `max` never touches it.
So even after combining b and c into {0,2}, idx=0>=0 still signals a reset:

  Right grouping: f(a, f(b,c))
    f(b, c): c.idx==-1, extend -> {0, max(1,2)} = {0, 2}   <- idx=0 preserved!
    f(a, {0,2}): idx>=0, reset -> {0, 2}                    <- idx=0 (correct)
*/

#include "containers/cubwrappers.cuh"
#include "containers/image.cuh"
#include "pinnedalloc.cuh"

#include <cstdint>
#include <cstdio>
#include <vector>

#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/zip_iterator.h"

template <typename Tflag> struct FlagAndIndex {
  int idx;
  Tflag flag;

  // does this in-house functor work? then i dont need a separate struct?
  __host__ __device__ FlagAndIndex<Tflag>
  operator()(thrust::tuple<Tflag, int> t) {
    Tflag f = thrust::get<0>(t);
    int col = thrust::get<1>(t);
    return {f == 1 ? col : -1, f};
  }

  bool operator==(const FlagAndIndex<Tflag> &o) const {
    return idx == o.idx && flag == o.flag;
  }
};

template <typename T, typename Op>
bool testAssociativity(Op op, const std::vector<T> &vals) {
  for (const auto &a : vals)
    for (const auto &b : vals)
      for (const auto &c : vals) {
        T lhs = op(op(a, b), c);
        T rhs = op(a, op(b, c));
        if (!(lhs == rhs)) {
          printf("Not associative: f(f({%d,%d},{%d,%d}),{%d,%d})={%d,%d} "
                 "!= f({%d,%d},f({%d,%d},{%d,%d}))={%d,%d}\n",
                 a.idx, (int)a.flag, b.idx, (int)b.flag, c.idx, (int)c.flag,
                 lhs.idx, (int)lhs.flag, a.idx, (int)a.flag, b.idx, (int)b.flag,
                 c.idx, (int)c.flag, rhs.idx, (int)rhs.flag);
          return false;
        }
      }
  return true;
}

template <typename Tflag> struct ScanOp {
  __host__ __device__ FlagAndIndex<Tflag>
  operator()(const FlagAndIndex<Tflag> &a, const FlagAndIndex<Tflag> &b) {
    // reset if b carries a boundary index
    if (b.idx >= 0) {
      return b;
    }
    // keep the larger flag, but keep the latest index
    else {
      return {a.idx, (Tflag)max(b.flag, a.flag)};
    }
  }
};

template <typename Tflag>
void printSegments(const Tflag *in_flags, const FlagAndIndex<Tflag> *scan_out,
                   const int length) {
  for (int i = 1; i < length; ++i) {
    if (in_flags[i] == 1) {
      if (scan_out[i - 1].flag == 2 && scan_out[i - 1].idx >= 0) {
        printf("[%d : %d]\n", scan_out[i - 1].idx, i);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  printf("Scans 1-2-1\n");

  using Tflag = int8_t;
  // non-reset elements have idx=-1; reset elements have idx>=0
  std::vector<FlagAndIndex<Tflag>> testVals = {
      {-1, 0}, {-1, 2}, {0, 1}, {1, 1}, {2, 1}, {0, 2}, {1, 2}, {2, 2}};
  bool assoc = testAssociativity(ScanOp<Tflag>{}, testVals);
  printf("ScanOp associative: %s\n", assoc ? "yes" : "no");

  int rows = 3;
  int cols = 8;
  // clang-format off
  // Sections marked * have a wraparound.
  thrust::pinned_host_vector<Tflag> h_flags = {
    1, 2, 1, 2, 0, 1, 0, 1, // [0:2], [2:5]
    0, 1, 0, 2, 1, 2, 0, 0, // [1:4], [4:1]*
    0, 2, 1, 2, 1, 0, 0, 0  // [2:4], [4:2]*
  };
  // clang-format on
  if ((int)h_flags.size() != rows * cols) {
    throw std::runtime_error("Mismatched flags dimensions, " +
                             std::to_string(h_flags.size()) +
                             " != " + std::to_string(rows * cols));
  }

  containers::DeviceImageStorage<Tflag> d_flags;
  d_flags.vec = h_flags;

  containers::DeviceImageStorage<FlagAndIndex<Tflag>> d_out(rows, cols);

  // Create iterators
  auto values_iter = thrust::make_transform_iterator(
      thrust::make_zip_iterator(d_flags.vec.data().get(),
                                cubw::helpers::makeColIndexIterator(cols)),
      FlagAndIndex<Tflag>{});
  auto keys_iter = cubw::helpers::makeRowIndexIterator(cols);
  auto out_iter = d_out.vec.data().get();

  // Create scanner
  cubw::DeviceScan::InclusiveScanByKey<decltype(keys_iter),
                                       decltype(values_iter),
                                       decltype(out_iter), ScanOp<Tflag>>
      scanner(rows * cols);
  scanner.exec(keys_iter, values_iter, out_iter, ScanOp<Tflag>{}, rows * cols);

  containers::PinnedHostImageStorage<FlagAndIndex<Tflag>> h_out(rows, cols);
  d_out.toHost(h_out);

  cudaDeviceSynchronize();
  if (rows == 3 && cols == 8) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%2d ", h_flags[i * cols + j]);
      }
      printf("\n");
    }
    printf("=======================\n");

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%2hhd ", h_out.vec[i * cols + j].flag);
      }
      printf("\n");
    }
    printf("-----------------------\n");

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%2d ", h_out.vec[i * cols + j].idx);
      }
      printf("\n");
    }
    printf("-----------------------\n");

    for (int i = 0; i < rows; ++i) {
      printf("Row %d: \n", i);
      printSegments<Tflag>(h_flags.data().get() + i * cols,
                           h_out.vec.data().get() + i * cols, cols);
      printf("-----------------------\n");
    }
  }

  return 0;
}
