// nvcc -std=c++17 -o macro_pollution macro_pollution.cu -keep

#define _T

#include <cub/device/device_radix_sort.cuh>

int main() { return 0; }

// clang-format off
/*

Examples of the errors this causes:


C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin/../include\cub/warp/specializations/warp_scan_shfl.cuh(408): error: identifier "input" is undefined
    __declspec(__device__) __forceinline  InclusiveScanStep( input, ScanOpT scan_op, int first_lane, int offset)    
                                                             ^

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin/../include\cub/warp/specializations/warp_scan_shfl.cuh(408): error: explicit type is missing ("int" assumed)
    __declspec(__device__) __forceinline  InclusiveScanStep( input, ScanOpT scan_op, int first_lane, int offset)    


Inspecting the preprocessor output:


template <typename , typename ScanOpT>
__declspec(__device__) __forceinline 
InclusiveScanStep( input, ScanOpT scan_op, int first_lane, int offset, ::cuda::std::true_type )
{
  return InclusiveScanStep(input, scan_op, first_lane, offset);
}

where the first template is missing the _T in the original file.
*/
