#include <thrust/host_vector.h>

#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/system/cuda/memory_resource.h>

template <typename T>
using pinned_allocator = thrust::mr::stateless_resource_allocator<
    T, thrust::system::cuda::universal_host_pinned_memory_resource>;

// Don't use this one! Here as a reminder that it's wrong!
// template <typename T>
// using wrong_pinned_allocator = thrust::mr::stateless_resource_allocator<
//     T, thrust::system::cpp::universal_host_pinned_memory_resource>;

// Define some quick template for a pinned host vector
namespace thrust {
template <typename T>
using pinned_host_vector = thrust::host_vector<T, pinned_allocator<T>>;
}
