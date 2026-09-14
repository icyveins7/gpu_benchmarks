# proj_slipups

This project isn't really something to build, but a collection of errors that I can reproduce that may be useful to remember for the future;
this way I can remember what happeend and what caused it, and possibly also ideas of how to fix similar issues later on.

## no_wmma_when_saturated

This benchmark demonstrates that kernels in separate CUDA streams are not guaranteed to run concurrently, even when one primarily uses CUDA cores and the other uses tensor cores. Both kernels still require SM resources for their thread blocks to become resident. When the burn kernel fills every SM to its occupancy limit, the WMMA kernel mostly waits until the burn kernel reaches its final wave and SM resources begin to become available.

The following results were measured with Nsight Systems 2025.2.1 on an NVIDIA A10 with 72 SMs. Each run used `burnIters=30000000`, `wmmaIters=5000000`, 12 resident burn blocks per SM, and equal burn and WMMA grid sizes of `864 * N`, where `N` is the burn grid multiplier. Times are relative to the start of the burn kernel in the concurrent case. The overlap is `burn end - WMMA start`.

| N | Grid size | WMMA start | Burn end | Overlap | Burn/N | Overlap fraction | Expected 1/N |
|--:|----------:|-----------:|---------:|--------:|-------:|-----------------:|-------------:|
| 1 | 864 | 198.048 ms | 684.544 ms | 486.496 ms | 684.544 ms | 71.07% | 100.00% |
| 2 | 1,728 | 731.882 ms | 1,294.141 ms | 562.259 ms | 647.070 ms | 43.45% | 50.00% |
| 3 | 2,592 | 1,334.430 ms | 1,912.040 ms | 577.610 ms | 637.347 ms | 30.21% | 33.33% |
| 4 | 3,456 | 1,936.586 ms | 2,515.272 ms | 578.685 ms | 628.818 ms | 23.01% | 25.00% |
| 5 | 4,320 | 2,536.269 ms | 3,119.364 ms | 583.096 ms | 623.873 ms | 18.69% | 20.00% |
| 6 | 5,184 | 3,152.411 ms | 3,731.813 ms | 579.402 ms | 621.969 ms | 15.53% | 16.67% |
| 7 | 6,048 | 3,760.619 ms | 4,343.546 ms | 582.927 ms | 620.507 ms | 13.42% | 14.29% |
| 8 | 6,912 | 4,415.829 ms | 4,978.548 ms | 562.719 ms | 622.318 ms | 11.30% | 12.50% |
| 9 | 7,776 | 5,010.416 ms | 5,593.800 ms | 583.384 ms | 621.533 ms | 10.43% | 11.11% |
| 10 | 8,640 | 5,619.395 ms | 6,202.622 ms | 583.227 ms | 620.262 ms | 9.40% | 10.00% |

The concurrent burn duration scales linearly with the multiplier (`64.6 ms + 613.3 ms * N`, R² = 0.999975), while the overlap for `N=3..10` remains nearly constant at `578.9 +/- 6.9 ms`. The measured overlap fraction is approximately `0.927/N`. This supports the hypothesis that the WMMA kernel begins running during the final burn wave as individual SMs start freeing enough resources to admit WMMA blocks. The shorter `N=1` and `N=2` runs deviate more because launch ramp-up and scheduling effects are a larger fraction of their execution time.
