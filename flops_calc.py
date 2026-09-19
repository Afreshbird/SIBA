import torch
from thop import profile
from models.SIBA import SIBA

# The version of Thop that we used: 0.1.1.post2209072238

model = SIBA().cuda()
model.eval()

input = torch.randn(1, 1, 128, 128).cuda()
flops, params = profile(model, inputs=(input, input))
print('='*50)
print('FLOPs = {} G;   Params = {} M'.format(str(flops/1000**3), str(params/1000**2)))
print('='*50)

