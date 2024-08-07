import torch

torch.ops.load_library("build/torchMatrixMultiply.so")

torch.manual_seed(42)
a = torch.rand(1000, 1000, device="cuda")
print(a)
b = torch.rand(1000, 1000, device="cuda")
print(b)

print("row kernel:")
print(torch.ops.myextension.mymatrixmultiply(a, b, "row"))
print("col kernel:")
print(torch.ops.myextension.mymatrixmultiply(a, b, "col"))
print("default kernel:")
print(torch.ops.myextension.mymatrixmultiply(a, b))
