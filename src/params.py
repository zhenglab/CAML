from tabnanny import verbose
from thop import profile
from config import Config
import os
from model.networks import *

config_path = os.path.join('./checkpoints', 'config.yml')
config = Config(config_path)

Inpainting = Inpainting(config)
Estimating = Estimating(config)

input = torch.randn(1, 3, 256, 256)
mask = torch.randn(1, 1, 256, 256)
input1 = torch.randn(1, 64, 256, 256)
input2 = torch.randn(1, 128, 128, 128)
input3 = torch.randn(1, 256, 64, 64)
flops0, params0 = profile(Inpainting, inputs=(input, mask, input1, input2, input3), verbose=False)
flops1, params1 = profile(Estimating, inputs=(input, input1), verbose=False)
flops = flops0+flops1
params = params0+params1
print("%.2fM" % (flops/1e6), "%.5fM" % (params/1e6))