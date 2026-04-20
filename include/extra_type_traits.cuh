#pragma once

/**
 * @brief Primarily used to template int2/float2/etc from a templated int/float
 * type.
 *
 * @example
```
 // Here x is a float2
 template <typename T = float>
 __global__ void kernel(cuda_vec2_t<T> x) { ... }
```
 */
template <typename T> struct cuda_vec2;
template <> struct cuda_vec2<float> {
  using type = float2;
};
template <> struct cuda_vec2<double> {
  using type = double2;
};
template <> struct cuda_vec2<int> {
  using type = int2;
};
template <> struct cuda_vec2<unsigned int> {
  using type = uint2;
};

template <typename T> using cuda_vec2_t = typename cuda_vec2<T>::type;
