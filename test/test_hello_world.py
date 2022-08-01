import torch
torch.ops.load_library("./build/libcsc_profiling.so")


torch.ops.csc_profiling.hello_world_from_gpu(2)