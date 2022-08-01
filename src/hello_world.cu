#include <torch/script.h>
#include <cuda.h>


__global__ void hello_world_kernel()
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("hello world from %d\n", tid);
}


void hello_world_from_gpu(
    int64_t thread_num
){
    hello_world_kernel<<<1, thread_num>>>();
    return;
}

static auto registry =
    torch::RegisterOperators(
        "csc_profiling::hello_world_from_gpu(int thread_num) -> ()", &hello_world_from_gpu);