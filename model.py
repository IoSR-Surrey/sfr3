from sr3_modules.unet import UNet
import torch.nn as nn
import argparse
import torch
import math


# UNet from https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
class SR3UNet(nn.Module):
    def __init__(self, grid_res=32, min_bottleneck=8):
        super().__init__()
        self.grid_res = grid_res

        # number of stages so bottleneck spatial size >= min_bottleneck
        n_stages = int(math.log2(grid_res / min_bottleneck)) + 1  # +1 to account for no final downsample
        channel_mults = tuple(2 ** i for i in range(n_stages))
        inner_channel = 64
        # attention only at resolutions <= 16
        attn_res = tuple(grid_res // (2 ** i) for i in range(1, n_stages) if grid_res // (2 ** i) <= 16)
        print(f"n_stages: {n_stages}")
        print(f"channel_mults: {channel_mults}")
        print(f"inner_channel: {inner_channel}")
        print(f"attn_res: {attn_res}")
        self.unet = UNet(in_channel=4, out_channel=2, inner_channel=inner_channel, channel_mults=channel_mults,
            image_size=grid_res, attn_res=attn_res, dropout=0.2)
        self.upsample_input = nn.Upsample(size=(grid_res, grid_res), mode='bicubic', align_corners=False)

    def upsample_lr(self, lr_lowres):
        return self.upsample_input(lr_lowres)

    def forward(self, x_concat, noise_level, freq):
        return self.unet(x_concat, noise_level, freq)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gridsize", type=int, default=32)
    args = parser.parse_args()

    def count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def suffix(n):
        if n >= 1_000_000_000:
            return f"{n:,} ({n/1_000_000_000:.2f}B)"
        if n >= 1_000_000:
            return f"{n:,} ({n/1_000_000:.2f}M)"
        if n >= 1_000:
            return f"{n:,} ({n/1_000:.2f}K)"
        return f"{n:,}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SR3UNet(grid_res=args.gridsize).to(device)
    unet = model.unet

    print("\nModel parameters")
    print("-" * 60)
    for name in ["cond_mlp", "downs", "mid", "ups", "final_conv"]:
        sub = getattr(unet, name, None)
        if sub is None:
            print(f"unet.{name:16s}: <missing>")
            continue
        tot, trn = count_params(sub)
        print(f"unet.{name:16s}: total {suffix(tot):>20s}   trainable {suffix(trn):>12s}")
    print("\nEncoder")
    for l, layer in enumerate(unet.downs):
        tot, trn = count_params(layer)
        print(f"  downs[{l:02d}] {type(layer).__name__:25s}: {suffix(tot):>20s}")
    print("\nDecoder")
    for l, layer in enumerate(unet.ups):
        tot, trn = count_params(layer)
        print(f"  ups[{l:02d}]   {type(layer).__name__:25s}: {suffix(tot):>20s}")
    print("-" * 60)
    model_total, model_trainable = count_params(model)
    print(f"\n{'full model':20s}: total {suffix(model_total):>20s}   trainable {suffix(model_trainable):>12s}\n")