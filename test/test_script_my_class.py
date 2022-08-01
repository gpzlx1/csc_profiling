import torch
torch.ops.load_library("./build/libcsc_profiling.so")

self = torch.randn(100, 100)
offset = torch.ones(100, 100)

myobject = torch.classes.my_classes.MyObject(self, offset)

print(myobject.get())
print(myobject.compute())
print(myobject.get())
print(myobject.ret2())

def test_func(self, offset):
    object = torch.classes.my_classes.MyObject(self, offset)
    object.compute()
    return object.get()

func_script = torch.jit.script(test_func)
print(func_script.code)
print(func_script.graph)

