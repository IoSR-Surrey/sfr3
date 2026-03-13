from sr3_modules.diffusion import GaussianDiffusion
from data import SoundFieldDataset
import matplotlib.pyplot as plt
from model import SR3UNet
import torch.nn.functional
from tqdm import tqdm
import numpy as np
import argparse
import random
import torch
import json
import os


def compute_nmse(target, prediction):
    """Returns the raw NMSE ratio (0 to 1+)."""
    return ((torch.abs(prediction - target) ** 2).sum() / (torch.norm(target, p=2) ** 2)).item()

def compute_ncc(prediction: torch.Tensor, target: torch.Tensor):
    """Returns Normalized Cross-Correlation (NCC) between estimated and ground truth."""
    prediction = prediction.ravel()
    target = target.ravel()
    return (torch.abs(prediction @ torch.conj(target)) / (torch.norm(prediction, p=2) * torch.norm(target, p=2))).item()

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

    print(f"Random room: {room_idx}, Data Index: {bin_idx}")
    print(f"Physical Freq: {freq_hz:.2f} Hz (Bin {bin_idx + 1})")

    model = SR3UNet(grid_res=dataset.hr_res).to(device)
    diffusion = GaussianDiffusion(denoise_fn=model, image_size=dataset.hr_res, channels=2).to(device)
    diffusion.set_new_noise_schedule(
        {'schedule': 'cosine', 'n_timestep': 1000, 'linear_start': 1e-4, 'linear_end': 2e-2}, device)

    ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        diffusion.load_state_dict(ckpt['diffusion_state_dict'])
    model.eval()
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

def frequency_analysis(metadata_path, checkpoint_dir, num_rooms=None, bin_step=None, seed=42):
    """
    Run frequency analysis and compare diffusion SR to baselines

    Args:
        metadata_path (str): path to the dataset metadata JSON.
        num_rooms (int or None): If None, use all rooms. If int, randomly select `num_rooms`
                                 distinct rooms to use for the analysis (no seed).
        bin_step (int or None): If None, process ALL bins. If int>0, process every `bin_step`-th bin.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SoundFieldDataset(path=metadata_path)
    model = SR3UNet(grid_res=dataset.hr_res).to(device)
    diffusion = GaussianDiffusion(denoise_fn=model, image_size=dataset.hr_res, channels=2).to(device)
    diffusion.set_new_noise_schedule(
        {'schedule': 'cosine', 'n_timestep': 1000, 'linear_start': 1e-4, 'linear_end': 2e-2}, device)

    ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        if 'diffusion_state_dict' in ckpt:
            diffusion.load_state_dict(ckpt['diffusion_state_dict'])
    model.eval()
    diffusion.eval()

    with open(metadata_path, "r") as f:
        meta_json = json.load(f)

    fs = meta_json.get("fs")
    n_fft = meta_json.get("n_fft")

    k_all = np.arange(1, dataset.bins_per_room + 1)
    freqs_all = k_all * float(fs) / float(n_fft)

    # select bins
    bins_per_room = dataset.bins_per_room
    if bin_step is None:
        selected_bin_indices = np.arange(0, bins_per_room)
        print(f"processing all {bins_per_room} bins.")
    else:
        bin_step = int(bin_step)
        selected_bin_indices = np.arange(0, bins_per_room, bin_step)
        print(f"processing bins with step {bin_step}: {len(selected_bin_indices)} bins selected.")

    freqs = freqs_all[selected_bin_indices]

    # select rooms
    total_rooms = dataset.num_rooms
    if num_rooms is None:
        selected_rooms = list(range(total_rooms))
        print(f"using all {total_rooms} rooms for frequency analysis.")
    else:
        n = min(int(num_rooms), total_rooms)
        selected_rooms = sorted(random.Random(seed).sample(range(total_rooms), n))
        print(f"using {n} randomly selected rooms (seed={seed}).")

    print(f"{len(selected_bin_indices)} bins across {len(selected_rooms)} rooms")

    avg_nmse_sr, avg_nmse_bic = [], []
    avg_ncc_sr, avg_ncc_bic = [], []

    for bin_idx in tqdm(selected_bin_indices, desc="Processing Selected Bins"):
        b_nmse_sr, b_nmse_bic = [], []
        b_ncc_sr, b_ncc_bic = [], []

        freq_hz = freqs_all[bin_idx]

        for room_idx in selected_rooms:
            idx = room_idx * dataset.bins_per_room + int(bin_idx)
            gt_hr, lr_low, freq = dataset[idx]

            with torch.no_grad():
                lr_cond = model.upsample_lr(lr_low.unsqueeze(0).to(device))
                freq_tensor = torch.tensor([freq], dtype=torch.float32, device=device)
                sr_out = diffusion.super_resolution({'SR': lr_cond, 'freq': freq_tensor})
                sr_out = sr_out.unsqueeze(0)

            gt_mag = complex_to_magnitude(gt_hr.unsqueeze(0).to(device))
            sr_mag = complex_to_magnitude(sr_out.to(device))

            target_size = (gt_mag.shape[-2], gt_mag.shape[-1])
            lr_complex = lr_low.unsqueeze(0).to(device)
            bicubic_complex = torch.nn.functional.interpolate(lr_complex, target_size, mode='bicubic',
                                                              align_corners=False)
            bicubic_out = complex_to_magnitude(bicubic_complex)

            metrics_sr = get_metrics(gt_mag, sr_mag)
            metrics_bic = get_metrics(gt_mag, bicubic_out)

            b_nmse_sr.append(metrics_sr['NMSE_dB'])
            b_ncc_sr.append(metrics_sr['NCC'])

            b_nmse_bic.append(metrics_bic['NMSE_dB'])
            b_ncc_bic.append(metrics_bic['NCC'])

        avg_nmse_sr.append(float(np.mean(b_nmse_sr)))
        avg_nmse_bic.append(float(np.mean(b_nmse_bic)))
        avg_ncc_sr.append(float(np.mean(b_ncc_sr)))
        avg_ncc_bic.append(float(np.mean(b_ncc_bic)))

        print(
            f"\nFreq {freq_hz:.1f}Hz Done. "
            f"Model NMSE: {avg_nmse_sr[-1]:.2f} dB | NCC: {avg_ncc_sr[-1]:.4f} | "
            f"Bicubic NMSE: {avg_nmse_bic[-1]:.2f} dB | Bicubic NCC: {avg_ncc_bic[-1]:.4f}"
        )

    x_axis = freqs
    fig, ax = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # NMSE [dB]
    ax[0].plot(x_axis, avg_nmse_sr, label='SFR3')
    ax[0].plot(x_axis, avg_nmse_bic, label='Bicubic', linestyle=':')
    ax[0].set_ylabel("NMSE [dB]")
    ax[0].set_title("Model testing")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    # NCC
    ax[1].plot(x_axis, avg_ncc_sr, label='SFR3')
    ax[1].plot(x_axis, avg_ncc_bic, label='Bicubic', linestyle=':')
    ax[1].set_ylabel("NCC")
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylim(bottom=0, top=1.05)
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    eval_dir = os.path.join(checkpoint_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "freq_analysis_comparison.pdf"))
    plt.show()

    # Save JSON Results
    results = {
        'meta': {
            'checkpoint_dir': checkpoint_dir,
            'seed': seed,
            'num_rooms': len(selected_rooms),
            'room_indices': [int(r) for r in selected_rooms],
            'bin_step': bin_step,
        },
        'freqs': [float(x) for x in x_axis],
        'nmse_dB': {
            'diff_m': [float(x) for x in avg_nmse_sr],
            'bic_m': [float(x) for x in avg_nmse_bic]
        },
        'ncc': {
            'diff_m': [float(x) for x in avg_ncc_sr],
            'bic_m': [float(x) for x in avg_ncc_bic]
        }
    }
    json_path = os.path.join(eval_dir, "freq_analysis.json")
    with open(json_path, "w") as jf:
        json.dump(results, jf, indent=2)
    print(f"comparison results written to: {json_path}")

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

    #evaluation(args.metadata, args.checkpoint_dir)
    frequency_analysis(args.metadata, args.checkpoint_dir, num_rooms=1, bin_step=30)