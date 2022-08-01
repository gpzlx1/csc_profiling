#pragma once

#include <torch/script.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cub/cub.cuh>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#define WARP_SIZE 32
#define MIN(x, y) ((x < y) ? x : y)
#define MAX(x, y) ((x > y) ? x : y)

// wrapper
template<typename IdType>
inline void cub_exclusiveSum(
    IdType* arrays,
    const IdType array_length
){
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        arrays,
        arrays,
        array_length
    );

    c10::Allocator* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
    c10::DataPtr _temp_data = cuda_allocator->allocate(temp_storage_bytes);
    d_temp_storage = _temp_data.get();

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        arrays,
        arrays,
        array_length
    );
}

__device__ __forceinline__ int getLaneId() {
    int laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}