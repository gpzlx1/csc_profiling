import torch
torch.ops.load_library("./build/libcsc_profiling.so")

deg = torch.randint(0, 100, (100,)).int().cuda()
indptr = torch.zeros((deg.numel() + 1)).int().cuda()
indptr[1:] = torch.cumsum(deg, dim=0).int().cuda()
indices = torch.arange(0, indptr[-1]).int().cuda()


mask = torch.randint(0, 2, (indices.numel(),)).bool().cuda()
for i in torch.ops.csc_profiling.csc_filter_indices_int32(mask, indptr, indices):
    print(i.shape, i)