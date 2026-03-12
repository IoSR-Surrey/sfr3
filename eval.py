from sr3_modules.diffusion import GaussianDiffusion
from data import SoundFieldDataset
import matplotlib.pyplot as plt
from model import SR3UNet
import torch.nn.functional
import numpy as np
import argparse
import random
import torch
import json
import os


def compute_nmse(target, prediction):
    """Returns the raw NMSE ratio (0 to 1+)."""
    return ((torch.abs(prediction - target) ** 2).sum() / (torch.norm(target, p=2) ** 2)).item()

def compute_ncc(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Returns Normalized Cross-Correlation (NCC) between estimated and ground truth."""
    prediction = prediction.ravel()
    target = target.ravel()
    return torch.abs(prediction @ torch.conj(target)) / (torch.norm(prediction, p=2) * torch.norm(target, p=2))

def get_metrics(gt, sr):
    nmse_raw = compute_nmse(gt, sr)
    nmse_db = 10 * np.log10(nmse_raw + 1e-15)
    ncc = compute_ncc(gt, sr)
    return {"NMSE_raw": nmse_raw, "NMSE_dB": nmse_db, "NCC": ncc}

def complex_to_magnitude(x):
    if x.dim() == 4 and x.size(1) == 2:
        real = x[:, 0:1, :, :]
        imag = x[:, 1:2, :, :]
        return torch.sqrt(real ** 2 + imag ** 2)
    return None


def evaluation(metadata_path='dataset/test/metadata.json', checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SoundFieldDataset(path=metadata_path)
    with open(metadata_path) as f:
        meta = json.load(f)
    fs = meta['fs']
    n_fft = meta['n_fft']
    room_idx = random.randint(0, dataset.num_rooms - 1)
    bin_idx = random.randint(0, dataset.bins_per_room - 1)
    sample_index = room_idx * dataset.bins_per_room + bin_idx

    # start from 2nd bin (index + 1) because 0Hz is removed
    freq_hz = (bin_idx + 1) * fs / n_fft

    print(f"[Eval] Random room: {room_idx}, Data Index: {bin_idx}")
    print(f"       Physical Freq: {freq_hz:.2f} Hz (Bin {bin_idx + 1})")

    model = SR3UNet(grid_res=dataset.hr_res).to(device)
    diffusion = GaussianDiffusion(denoise_fn=model, image_size=dataset.hr_res, channels=2).to(device)
    diffusion.set_new_noise_schedule(
        {'schedule': 'cosine', 'n_timestep': 1000, 'linear_start': 1e-4, 'linear_end': 2e-2}, device)

    ckpt_path = os.path.join(checkpoint_dir, "latest_model.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        diffusion.load_state_dict(ckpt['diffusion_state_dict'])
    diffusion.eval()

    gt_hr, lr_low, freq = dataset[sample_index]

    with torch.no_grad():
        lr_cond = model.upsample_lr(lr_low.unsqueeze(0).to(device))
        freq_tensor = torch.tensor([freq], dtype=torch.float32, device=device)
        sr_output = diffusion.super_resolution({'SR': lr_cond, 'freq': freq_tensor})
        sr_output = sr_output.unsqueeze(0)

    gt_mag = complex_to_magnitude(gt_hr.unsqueeze(0).to(device))
    sr_mag = complex_to_magnitude(sr_output)
    lr_mag = complex_to_magnitude(lr_low.unsqueeze(0).to(device))

    target_size = (sr_mag.shape[-2], sr_mag.shape[-1])
    lr_complex = lr_low.unsqueeze(0).to(device)
    bicubic_complex = torch.nn.functional.interpolate(lr_complex, target_size, mode='bicubic', align_corners=False)
    bicubic_out = complex_to_magnitude(bicubic_complex)

    m = get_metrics(gt_mag, sr_mag)
    m_bic = get_metrics(gt_mag, bicubic_out)
    print(f"NMSE: {m['NMSE_dB']:.2f} dB | NCC: {m['NCC']:.2f}")
    print(f"Bicubic NMSE: {m_bic['NMSE_dB']:.2f} dB | Bicubic NCC: {m_bic['NCC']:.2f}")

    fig, axes = plt.subplots(1, 4, figsize=(25, 5))

    axes[0].imshow(lr_mag[0, 0].cpu().numpy(), origin='lower')
    axes[0].set_title("Input (LR)")
    axes[1].imshow(bicubic_out[0, 0].cpu().numpy(), origin='lower')
    axes[1].set_title("Bicubic Interpolation")
    axes[2].imshow(sr_mag[0, 0].cpu().numpy(), origin='lower')
    axes[2].set_title("Generated (Diffusion)")
    axes[3].imshow(gt_mag[0, 0].cpu().numpy(), origin='lower')
    axes[3].set_title("Ground Truth")
    eval_dir = os.path.join(checkpoint_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    plt.suptitle(f"Room {room_idx} - Freq {freq_hz:.1f}Hz")
    plt.savefig(os.path.join(eval_dir, "gtgen_comparison.pdf"))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="dataset_4to32_500_(2000rooms)/test/metadata.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_4to32_500_(2000rooms)")
    parser.add_argument("--seed", type=int, default=67)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    evaluation(args.metadata, args.checkpoint_dir)