import torch
torch.ops.load_library("./build/libcsc_profiling.so")

deg = torch.randint(0, 100, (1000,)).int().cuda()
indptr = torch.zeros((deg.numel() + 1)).int().cuda()
indptr[1:] = torch.cumsum(deg, dim=0).int().cuda()
indices = torch.arange(0, indptr[-1]).int().cuda()

index = torch.randint(0, 1000, (100,)).int().cuda()
for i in torch.ops.csc_profiling.csc_filter_indptr_int32(index, indptr, indices):
    print(i.shape, i)