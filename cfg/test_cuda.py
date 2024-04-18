import torch

if_cuda = torch.cuda.is_available()
print("if_cuda=",if_cuda)
gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)
import torch
from torch import nn
#  查看gpu信息
cudaMsg = torch.cuda.is_available()
gpuCount = torch.cuda.device_count()
print("1.是否存在GPU:{}".format(cudaMsg), "如果存在有：{}个".format(gpuCount))
