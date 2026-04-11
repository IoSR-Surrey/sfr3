from train import SR3DiffusionModule
from data import SoundFieldDataset
import matplotlib.pyplot as plt
from kernel import UenoKernel
import torch.nn.functional
from tqdm import tqdm
import numpy as np
import argparse
import random
import torch
import json
import sys
import csv
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "complex_unet"))
from complex_unet.eval_complexunet import load_complexunet, complexunet_infer


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

def plot_training(checkpoints, show_lr=True):

    csv_path = os.path.join(checkpoints, 'training_history.csv')
    epochs, train_losses, val_losses, lrs = [], [], [], []
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            lrs.append(float(row['lr']))

    nrows = 2 if show_lr else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 5 * nrows), sharex=True, squeeze=False)

    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue', alpha=0.7, linewidth=2)
    ax1.plot(epochs, val_losses, label='Validation Loss', color='orange', alpha=0.7, linewidth=2)
    ax1.set_title('Train and Val losses', fontsize=15)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=12)

    if show_lr:
        ax2 = axes[1, 0]
        ax2.plot(epochs, lrs, label='Learning Rate', color='red', linewidth=2, drawstyle='steps-post')
        ax2.set_title('Learning Rate over epochs', fontsize=15)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_yscale('log')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=12)

    axes[-1, 0].set_xlabel('Epoch', fontsize=12)
    plt.tight_layout()
    plt.show()


def evaluation(metadata_path='dataset/test/metadata.json', checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SoundFieldDataset(path=metadata_path)
    cvnn_dataset = SoundFieldDataset(path=metadata_path, cvnn=True)
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

    ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
    lit = SR3DiffusionModule.load_from_checkpoint(ckpt_path, map_location=device)
    model = lit.diffusion.denoise_fn.to(device)
    diffusion = lit.diffusion.to(device)
    diffusion.eval()

    cvnn_model = load_complexunet(cvnn_dataset, device)

    gt_hr, lr_low, freq = dataset[sample_index]

    kernel = UenoKernel(lr_res=dataset.lr_res, hr_res=dataset.hr_res)
    lr_batch = lr_low.unsqueeze(0).to(kernel.device)
    kernel_recon = kernel.reconstruct(lr_batch, freq_hz=freq_hz).squeeze(0).cpu()
    kernel_mag = complex_to_magnitude(kernel_recon.unsqueeze(0).to(device))

    with torch.no_grad():
        lr_cond = model.upsample_lr(lr_low.unsqueeze(0).to(device))
        freq_tensor = torch.tensor([freq], dtype=torch.float32, device=device)
        sr_output = diffusion.super_resolution({'SR': lr_cond, 'freq': freq_tensor})
        sr_output = sr_output.unsqueeze(0)

    gt_mag = complex_to_magnitude(gt_hr.unsqueeze(0).to(device))
    sr_mag = complex_to_magnitude(sr_output)
    lr_mag = complex_to_magnitude(lr_low.unsqueeze(0).to(device))
    cvnn_mag = complexunet_infer(cvnn_model, cvnn_dataset, room_idx, bin_idx, device)

    gt_hr_raw, _, _ = cvnn_dataset[sample_index]
    slice_max = torch.max(torch.abs(gt_hr_raw)).to(device)
    cvnn_mag_norm = cvnn_mag / slice_max

    target_size = (sr_mag.shape[-2], sr_mag.shape[-1])
    lr_complex = lr_low.unsqueeze(0).to(device)
    bicubic_complex = torch.nn.functional.interpolate(lr_complex, target_size, mode='bicubic', align_corners=False)
    bicubic_out = complex_to_magnitude(bicubic_complex)

    m = get_metrics(gt_mag, sr_mag)
    m_bic = get_metrics(gt_mag, bicubic_out)
    m_ker = get_metrics(gt_mag, kernel_mag)
    m_cvnn = get_metrics(gt_mag, cvnn_mag_norm)
    print(f"NMSE: {m['NMSE_dB']:.2f} dB | NCC: {m['NCC']:.2f}")
    print(f"Bicubic NMSE: {m_bic['NMSE_dB']:.2f} dB | Bicubic NCC: {m_bic['NCC']:.2f}")
    print(f"Kernel NMSE: {m_ker['NMSE_dB']:.2f} dB | Kernel NCC: {m_ker['NCC']:.2f}")
    print(f"CVNN NMSE: {m_cvnn['NMSE_dB']:.2f} dB | NCC: {m_cvnn['NCC']:.2f}")

    fig, axes = plt.subplots(1, 6, figsize=(30, 5))

    axes[0].imshow(lr_mag[0, 0].cpu().numpy(), origin='lower')
    axes[0].set_title("Input (LR)")
    axes[0].yaxis.get_major_locator().set_params(integer=True)
    axes[1].imshow(bicubic_out[0, 0].cpu().numpy(), origin='lower')
    axes[1].set_title("Bicubic Interpolation")
    axes[2].imshow(kernel_mag[0, 0].cpu().numpy(), origin='lower')
    axes[2].set_title("Ueno Kernel")
    axes[3].imshow(cvnn_mag[0, 0].cpu().numpy(), origin='lower')
    axes[3].set_title("CVNN")
    axes[4].imshow(sr_mag[0, 0].cpu().numpy(), origin='lower')
    axes[4].set_title("Generated (Diffusion)")
    axes[5].imshow(gt_mag[0, 0].cpu().numpy(), origin='lower')
    axes[5].set_title("Ground Truth")
    eval_dir = os.path.join(checkpoint_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    plt.suptitle(f"Room {room_idx} - Freq {freq_hz:.1f}Hz")
    plt.savefig(os.path.join(eval_dir, "gtgen_comparison.pdf"))
    plt.show()


def frequency_analysis(metadata_path, checkpoint_dir, num_rooms=None, bin_step=None, seed=42, run_inference=False):
    """
    Run frequency analysis and compare diffusion SR to baselines
    """

    eval_dir = os.path.join(checkpoint_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    json_path = os.path.join(eval_dir, "freq_analysis.json")

    if not os.path.exists(json_path):
        run_inference = True

    if run_inference:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = SoundFieldDataset(path=metadata_path)
        cvnn_dataset = SoundFieldDataset(path=metadata_path, cvnn=True)
        ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
        lit = SR3DiffusionModule.load_from_checkpoint(ckpt_path, map_location=device)
        model = lit.diffusion.denoise_fn.to(device)
        diffusion = lit.diffusion.to(device)
        diffusion.eval()

        cvnn_model = load_complexunet(cvnn_dataset , device)

        kernel = UenoKernel(lr_res=dataset.lr_res, hr_res=dataset.hr_res)

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
        avg_nmse_ker, avg_ncc_ker = [], []
        avg_nmse_cvnn, avg_ncc_cvnn = [], []

        for bin_idx in tqdm(selected_bin_indices, desc="Processing bins"):
            b_nmse_sr, b_nmse_bic = [], []
            b_ncc_sr, b_ncc_bic = [], []
            b_nmse_ker, b_ncc_ker = [], []
            b_nmse_cvnn, b_ncc_cvnn = [], []

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

                lr_batch = lr_low.unsqueeze(0).to(kernel.device)
                kernel_recon = kernel.reconstruct(lr_batch, freq_hz=float(freq_hz))
                kernel_mag = complex_to_magnitude(kernel_recon.to(device))

                cvnn_mag = complexunet_infer(cvnn_model, cvnn_dataset , room_idx, int(bin_idx), device)

                gt_hr_raw, _, _ = cvnn_dataset[room_idx * dataset.bins_per_room + int(bin_idx)]
                slice_max = torch.max(torch.abs(gt_hr_raw)).to(device)
                cvnn_mag_norm = cvnn_mag / slice_max

                metrics_sr = get_metrics(gt_mag, sr_mag)
                metrics_bic = get_metrics(gt_mag, bicubic_out)
                metrics_ker = get_metrics(gt_mag, kernel_mag)
                metrics_cvnn = get_metrics(gt_mag, cvnn_mag_norm)

                b_nmse_sr.append(metrics_sr['NMSE_dB'])
                b_ncc_sr.append(metrics_sr['NCC'])

                b_nmse_bic.append(metrics_bic['NMSE_dB'])
                b_ncc_bic.append(metrics_bic['NCC'])

                b_nmse_ker.append(metrics_ker['NMSE_dB'])
                b_ncc_ker.append(metrics_ker['NCC'])

                b_nmse_cvnn.append(metrics_cvnn['NMSE_dB'])
                b_ncc_cvnn.append(metrics_cvnn['NCC'])

            avg_nmse_sr.append(float(np.mean(b_nmse_sr)))
            avg_nmse_bic.append(float(np.mean(b_nmse_bic)))
            avg_ncc_sr.append(float(np.mean(b_ncc_sr)))
            avg_ncc_bic.append(float(np.mean(b_ncc_bic)))
            avg_nmse_ker.append(float(np.mean(b_nmse_ker)))
            avg_ncc_ker.append(float(np.mean(b_ncc_ker)))
            avg_nmse_cvnn.append(float(np.mean(b_nmse_cvnn)))
            avg_ncc_cvnn.append(float(np.mean(b_ncc_cvnn)))

            print(
                f"\nFreq {freq_hz:.1f}Hz Done. "
                f"Model NMSE: {avg_nmse_sr[-1]:.2f} dB | NCC: {avg_ncc_sr[-1]:.4f} | "
                f"Bicubic NMSE: {avg_nmse_bic[-1]:.2f} dB | Bicubic NCC: {avg_ncc_bic[-1]:.4f} | "
                f"Kernel NMSE: {avg_nmse_ker[-1]:.2f} dB | Kernel NCC: {avg_ncc_ker[-1]:.4f} | "
                f"CVNN NMSE: {avg_nmse_cvnn[-1]:.2f} dB | CVNN NCC: {avg_ncc_cvnn[-1]:.4f}"
            )

        results = {
            'meta': {
                'checkpoint_dir': checkpoint_dir,
                'seed': seed,
                'num_rooms': len(selected_rooms),
                'room_indices': [int(r) for r in selected_rooms],
                'bin_step': bin_step,
            },
            'freqs': [float(x) for x in freqs],  # changed from x_axis to freqs to match inference loop
            'nmse_dB': {
                'diff_m': [float(x) for x in avg_nmse_sr],
                'bic_m': [float(x) for x in avg_nmse_bic],
                'ker_m': [float(x) for x in avg_nmse_ker],
                'cvnn_m': [float(x) for x in avg_nmse_cvnn]
            },
            'ncc': {
                'diff_m': [float(x) for x in avg_ncc_sr],
                'bic_m': [float(x) for x in avg_ncc_bic],
                'ker_m': [float(x) for x in avg_ncc_ker],
                'cvnn_m': [float(x) for x in avg_ncc_cvnn]
            }
        }
        with open(json_path, "w") as jf:
            json.dump(results, jf, indent=2)
        print(f"Comparison results written to: {json_path}")

        x_axis = freqs

    else:
        # load from JSON instead of running inference
        print(f"Loading results from {json_path}")
        with open(json_path, "r") as jf:
            results = json.load(jf)

        x_axis = results['freqs']
        avg_nmse_sr = results['nmse_dB']['diff_m']
        avg_nmse_bic = results['nmse_dB']['bic_m']
        avg_ncc_sr = results['ncc']['diff_m']
        avg_ncc_bic = results['ncc']['bic_m']
        avg_nmse_ker = results['nmse_dB']['ker_m']
        avg_ncc_ker = results['ncc']['ker_m']
        avg_nmse_cvnn = results['nmse_dB']['cvnn_m']
        avg_ncc_cvnn = results['ncc']['cvnn_m']

    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=True)

    # NMSE [dB]
    ax[0].plot(x_axis, avg_nmse_sr, label='SFR3')
    ax[0].plot(x_axis, avg_nmse_bic, label='Bicubic', linestyle=':')
    ax[0].plot(x_axis, avg_nmse_ker, label='Kernel', linestyle='--')
    ax[0].plot(x_axis, avg_nmse_cvnn, label='CVNN', linestyle='--')
    ax[0].set_ylabel("NMSE [dB]")
    ax[0].set_xlabel("Frequency [Hz]")
    ax[0].set_title("NMSE vs Freq")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    # NCC
    ax[1].plot(x_axis, avg_ncc_sr, label='SFR3')
    ax[1].plot(x_axis, avg_ncc_bic, label='Bicubic', linestyle=':')
    ax[1].plot(x_axis, avg_ncc_ker, label='Kernel', linestyle='--')
    ax[1].plot(x_axis, avg_ncc_cvnn, label='CVNN', linestyle='--')
    ax[1].set_ylabel("NCC")
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylim(bottom=0, top=1.05)
    ax[1].set_title("NCC vs Freq")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "freq_analysis_comparison.pdf"))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="dataset_4to32_500_10krooms/test/metadata.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_4to32_500_10krooms")
    parser.add_argument("--seed", type=int, default=6158)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    plot_training(args.checkpoint_dir, show_lr=True)
    evaluation(args.metadata, args.checkpoint_dir)
    frequency_analysis(args.metadata, args.checkpoint_dir, num_rooms=50, bin_step=1, run_inference=False)