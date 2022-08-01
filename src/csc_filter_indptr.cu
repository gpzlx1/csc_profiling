#include <vector>
#include "cuda_utils.cuh"


template<typename IdType, int BLOCK_WARPS, int TILE_SIZE> 
__global__ void csc_filter_indptr_kernel(
    const IdType* __restrict__ index,
    const IdType* __restrict__ indptr,
    const IdType* __restrict__ indices,
    int num_items,
    const IdType* __restrict__ new_indptr,
    IdType* __restrict__ new_indices
){
    assert(blockDim.x == WARP_SIZE);

    int laneid = getLaneId();
    IdType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
    IdType last_row = MIN(static_cast<IdType>(blockDim.x + 1) * TILE_SIZE, num_items);

    while (out_row < last_row)
    {
        IdType row = index[out_row];
        IdType in_row_start = indptr[row];
        IdType out_row_start = new_indptr[out_row];
        IdType deg = new_indptr[out_row + 1] - out_row_start;

        for(int id = laneid; id < deg; id += WARP_SIZE){
            new_indices[out_row_start + id] = indices[in_row_start + id];
        }

        out_row += BLOCK_WARPS;
    }
}

// [todo(ping)] Can we merge compute new_indptr and compute new_indices into one kernel?   
template<typename IdType>
std::tuple<torch::Tensor, torch::Tensor> csc_filter_indptr(
    torch::Tensor index,
    torch::Tensor indptr,
    torch::Tensor indices
){
    auto cuda_allocator = c10::cuda::CUDACachingAllocator::get();
    int num_items = index.numel();

    // allocate space for new_indptr 
    torch::Tensor new_indptr = torch::zeros({num_items + 1,}, indptr.options());
    thrust::device_ptr<IdType> item_prefix(static_cast<IdType*>(new_indptr.data_ptr<IdType>()));

    // compute new_indptr
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(thrust::device, it(0), it(num_items),
                    [in = index.data_ptr<IdType>(),
                     in_indptr = indptr.data_ptr<IdType>(),
                     out = thrust::raw_pointer_cast(item_prefix)] __device__(int i) mutable {
                        IdType begin = in_indptr[in[i]];
                        IdType end = in_indptr[in[i] + 1];
                        out[i] = end - begin;
                     }
    );

    // prefix_sum
    cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix), num_items + 1);

    // allocate space for new_indices
    int nnz = item_prefix[num_items];
    torch::Tensor new_indices = torch::zeros({nnz,}, indices.options());

    // compute new_indices
    constexpr int BLOCK_WARP = 128 / WARP_SIZE;
    constexpr int TILE_SIZE = BLOCK_WARP * 16;
    const dim3 block(WARP_SIZE, BLOCK_WARP);
    const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
    csc_filter_indptr_kernel<IdType, BLOCK_WARP, TILE_SIZE><<<grid, block>>>(
        index.data_ptr<IdType>(),
        indptr.data_ptr<IdType>(),
        indices.data_ptr<IdType>(),
        num_items,
        new_indptr.data_ptr<IdType>(),
        new_indices.data_ptr<IdType>()
    );

    return std::make_tuple(new_indptr, new_indices);
}


static auto registry =
    torch::RegisterOperators(
        "csc_profiling::csc_filter_indptr_int32(Tensor index, Tensor indptr, Tensor indices)"
            "-> (Tensor, Tensor)", &csc_filter_indptr<int32_t>);