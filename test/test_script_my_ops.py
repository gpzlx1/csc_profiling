import torch
torch.ops.load_library("./build/libcsc_profiling.so")

self = torch.randn(100, 100)
offset = torch.ones(100, 100)
print(torch.ops.my_ops.my_add)
print(torch.ops.my_ops.my_add(self, offset))

def test_func(self, offset):
    d = {'1':self, '2':offset}
    return torch.ops.my_ops.my_add(d['1'], d['2'])

func_script = torch.jit.script(test_func)
print(func_script.code)
print(func_script.graph)