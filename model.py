from sr3_modules.unet import UNet
import torch.nn as nn
import math


# UNet from https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
class SR3UNet(nn.Module):
    def __init__(self, grid_res=32):
        super().__init__()
        self.grid_res = grid_res
        # attention at all resolutions from grid_res/2 down to 8
        attn_res = tuple(grid_res // (2 ** i) for i in range(1, int(math.log2(grid_res)) - 1))
        self.unet = UNet(in_channel=4, out_channel=2, inner_channel=32, image_size=grid_res, attn_res=attn_res)
        self.upsample_input = nn.Upsample(size=(grid_res, grid_res), mode='bicubic', align_corners=False)

    def upsample_lr(self, lr_lowres):
        return self.upsample_input(lr_lowres)

    def forward(self, x_concat, noise_level):
        return self.unet(x_concat, noise_level)