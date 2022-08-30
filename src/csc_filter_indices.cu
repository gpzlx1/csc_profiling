#include <vector>
#include "cuda_utils.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

template<typename IdType, int BLOCK_WARPS, int TILE_SIZE>
__global__ void csc_filter_indices_kernel(
    const IdType* __restrict__ indptr,
    const bool* __restrict__ bool_mask,
    int num_items,
    IdType* new_deg
){
    assert(blockDim.x == WARP_SIZE);

    auto block = cg::this_thread_block();
    auto group = cg::tiled_partition<WARP_SIZE>(block);
    int laneid = group.thread_rank();

    IdType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
    IdType last_row = MIN(static_cast<IdType>(blockIdx.x + 1) * TILE_SIZE, num_items);

    while(out_row < last_row){
        int true_count = 0;

        IdType row = out_row;
        IdType in_row_start = indptr[row];
        IdType deg = indptr[row + 1] - in_row_start;

        for(int idx = laneid + in_row_start; idx < deg + in_row_start; idx += WARP_SIZE){
            true_count += int(bool_mask[idx]);
        }
        
        // reduce::sum
        true_count = cg::reduce(group, true_count, cg::plus<int>());
        
        if(laneid == 0){
            new_deg[out_row] = true_count;
        }

        out_row += BLOCK_WARPS;
    }
}


// ping : we remove edges or nodes according to Tensor mask
template<typename IdType>
std::tuple<torch::Tensor, torch::Tensor> csc_filter_indices(
    torch::Tensor mask,
    torch::Tensor indptr,
    torch::Tensor indices
){
    int num_items = indptr.numel() - 1;
    torch::Tensor _bool_mask = mask.to(torch::kBool);

    // compute indptr
    //// compute deg
    torch::Tensor new_indptr =  torch::zeros_like(indptr);
    constexpr int BLOCK_WARP = 128 / WARP_SIZE;
    constexpr int TILE_SIZE = BLOCK_WARP * 16;
    const dim3 block(WARP_SIZE, BLOCK_WARP);
    const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
    csc_filter_indices_kernel<IdType, BLOCK_WARP, TILE_SIZE><<<grid, block>>>(
        indptr.data_ptr<IdType>(),
        _bool_mask.data_ptr<bool>(),
        num_items,
        new_indptr.data_ptr<IdType>()
    );

    //// prefix_sum
    thrust::device_ptr<IdType> item_prefix(static_cast<IdType*>(new_indptr.data_ptr<IdType>()));
    cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix), num_items + 1);

    // compute indices
    torch::Tensor _index = torch::nonzero(_bool_mask).reshape({-1,});
    torch::Tensor new_indices = indices.index({_index});

    return std::make_tuple(new_indptr, new_indices);
}


static auto registry =
    torch::RegisterOperators(
        "csc_profiling::csc_filter_indices_int32(Tensor mask, Tensor indptr, Tensor indices)"
            "-> (Tensor, Tensor)", &csc_filter_indices<int32_t>);
