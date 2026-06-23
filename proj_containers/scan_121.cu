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
#include <random>
#include <vector>

#include "cxxopts.hpp"

#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/transform_output_iterator.h"
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

struct Segment {
  int row;
  int startcol;
  int endcol;
};

template <typename Tflag, bool AllowWraparound = false> struct ToSegment {
  containers::Image<FlagAndIndex<Tflag>> scan_out;

  ToSegment() : scan_out(nullptr, 0, 0) {}
  ToSegment(containers::Image<FlagAndIndex<Tflag>> out) : scan_out(out) {}

  // NOTE: this entire function depends on the SelectOp below, with the
  // AllowWraparound template being identical.
  __host__ __device__ Segment
  operator()(const thrust::tuple<int, int> &t) const {
    int row = thrust::get<0>(t);
    int col = thrust::get<1>(t);
    // It is safe to just do this wraparound calculation all the time,
    // since the SelectOp would not produce an output for non wraparound
    // segments if disabled.
    int prevCol = col == 0 ? scan_out.width - 1 : col - 1;
    int startcol = scan_out.at(row, prevCol).idx;
    if constexpr (AllowWraparound) {
      // If idx < 0, there was no valid starting '1' to the left in the linear
      // scan. The segment wraps around: the start is the last '1' in the row.
      if (startcol < 0) {
        startcol = scan_out.at(row, scan_out.width - 1).idx;
        // usually 'nicer' if we present it as strictly increasing from left to
        // right boundaries
        startcol = startcol > col ? startcol - scan_out.width : startcol;
      }
    }
    return {row, startcol, col};
  }
};

template <typename Tflag, bool AllowWraparound = false> struct SelectOp {
  containers::Image<Tflag> original;
  containers::Image<FlagAndIndex<Tflag>> scan_out;

  SelectOp() : original(nullptr, 0, 0), scan_out(nullptr, 0, 0) {}
  SelectOp(containers::Image<Tflag> orig,
           containers::Image<FlagAndIndex<Tflag>> out)
      : original(orig), scan_out(out) {}

  __host__ __device__ bool
  operator()(const thrust::tuple<int, int> &idx) const {
    auto row = thrust::get<0>(idx);
    auto col = thrust::get<1>(idx);

    int prevCol = col - 1;
    Tflag originalFlag = original.at(row, col);
    // Only look at the segment edges
    if (originalFlag != 1)
      return false;

    if constexpr (AllowWraparound) {
      prevCol = prevCol < 0 ? prevCol + original.width : prevCol;
      // Read the prevCol flag, which is by definition always in bounds
      Tflag prevColFlag = scan_out.at(row, prevCol).flag;
      // This may either be 2 already
      // e.g. 0 2 1 ......
      // in which case it is already a 'valid open' segment
      if (prevColFlag == 2) {
        return true;
      }
      // or it may be an 'unknown open' segment at the front
      // e.g. 0 0 1 ..... 1 2 0
      // which could be 'validated' at the back
      else if (scan_out.at(row, prevCol).idx < 0) {
        // Then we need to check the last index
        return scan_out.at(row, scan_out.width - 1).flag == 2;
      }
    }
    // Logic for inner-only segments
    else {
      if (col >= 0 && prevCol >= 0 && scan_out.at(row, prevCol).flag == 2 &&
          scan_out.at(row, prevCol).idx >= 0) {
        return true;
      } else {
        return false;
      }
    }
  }
};

int main(int argc, char *argv[]) {
  printf("Scans 1-2-1\n");

  using Tflag = int8_t;
  // non-reset elements have idx=-1; reset elements have idx>=0
  std::vector<FlagAndIndex<Tflag>> testVals = {
      {-1, 0}, {-1, 2}, {0, 1}, {1, 1}, {2, 1}, {0, 2}, {1, 2}, {2, 2}};
  bool assoc = testAssociativity(ScanOp<Tflag>{}, testVals);
  printf("ScanOp associative: %s\n", assoc ? "yes" : "no");

  // clang-format off
  cxxopts::Options options("scan_121", "1-2-1 segment scan");
  options.add_options()
    ("r,rows", "Number of rows",    cxxopts::value<int>()->default_value("4"))
    ("c,cols", "Number of columns", cxxopts::value<int>()->default_value("8"))
    ("h,help", "Print usage")
  ;
  // clang-format on
  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    printf("%s\n", options.help().c_str());
    return 0;
  }

  int rows = result["rows"].as<int>();
  int cols = result["cols"].as<int>();
  bool randomize = result.count("rows") || result.count("cols");

  thrust::pinned_host_vector<Tflag> h_flags(rows * cols);
  if (randomize) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 2);
    for (auto &f : h_flags)
      f = (Tflag)dist(rng);
  } else {
    // clang-format off
    // Sections marked * have a wraparound.
    h_flags = {
      1, 2, 1, 2, 0, 1, 0, 1, // [0:2], [2:5]
      0, 1, 0, 2, 1, 2, 0, 0, // [1:4], [4:1]*
      0, 2, 1, 2, 1, 0, 0, 0, // [2:4], [4:2]*
      0, 2, 1, 2, 1, 0, 0, 1  // [2:4], [4:7], [7:2]*
    };
    // clang-format on
  }

  containers::DeviceImageStorage<Tflag> d_flags(cols, rows);
  d_flags.vec = h_flags;

  containers::DeviceImageStorage<FlagAndIndex<Tflag>> d_out(cols, rows);

  // Create scan iterators
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

  containers::PinnedHostImageStorage<FlagAndIndex<Tflag>> h_out(cols, rows);
  d_out.toHost(h_out);

  cudaDeviceSynchronize();
  if (rows * cols <= 64) {
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

  // Create selector iterators
  thrust::device_vector<Segment> d_select_output(rows * cols);
  thrust::device_vector<unsigned int> d_select_count(1);
  auto select_in_iter =
      thrust::make_zip_iterator(cubw::helpers::makeRowIndexIterator(cols),
                                cubw::helpers::makeColIndexIterator(cols));
  auto select_out_iter = thrust::make_transform_output_iterator(
      d_select_output.data().get(), ToSegment<Tflag, true>{d_out.image()});
  auto selectOp = SelectOp<Tflag, true>{d_flags.image(), d_out.image()};

  cubw::DeviceSelect::If<decltype(select_in_iter), decltype(select_out_iter),
                         decltype(d_select_count.data().get()),
                         SelectOp<Tflag, true>>
      selector(rows * cols);

  selector.exec(select_in_iter, select_out_iter, d_select_count.data().get(),
                rows * cols, selectOp);

  cudaDeviceSynchronize();
  thrust::pinned_host_vector<unsigned int> h_select_count = d_select_count;
  thrust::host_vector<Segment> h_select_output = d_select_output;
  printf("Select count: %u\n", h_select_count[0]);

  if (rows * cols <= 64) {
    for (int i = 0; i < (int)h_select_count[0]; ++i) {
      auto &s = h_select_output[i];
      printf("row=%d [%d:%d]\n", s.row, s.startcol, s.endcol);
    }
  }

  return 0;
}
