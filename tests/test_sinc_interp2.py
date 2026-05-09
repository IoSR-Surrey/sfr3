import os
import sys
import json
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytest
from pathlib import Path

rootdir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(rootdir))

from scripts.sinc_interp import sinc_interp2
from eval import get_metrics, complex_to_magnitude
from data import SoundFieldDataset
from scripts.plot_sinc_interp2 import test_visualize_sinc_interpolation

configs = [("datasets/dataset_4to32_500/test/metadata.json", "checkpoints/checkpoints_4to32_500"),
           ("datasets/dataset_8to64_1000/test/metadata.json", "checkpoints/checkpoints_8to64_1000")]


def test_sinc_interp2_identity_on_grid() -> None:
    # If we query exactly at the original sample points, ideal sinc interpolation must reproduce
    # the samples: sinc(0)=1 and sinc(n)=0 for any nonzero integer n, so only the (i,j) term remains.
    rng = np.random.default_rng(0)

    X = np.arange(8.0)
    Y = np.arange(6.0)
    f = rng.standard_normal((len(Y), len(X)))

    XX, YY = np.meshgrid(X, Y, indexing="xy")
    f_hat = sinc_interp2(f, X, Y, XX, YY)

    np.testing.assert_allclose(f_hat, f, rtol=0.0, atol=1e-12)


def test_sinc_interp2_delta_matches_sinc_product() -> None:
    # Put a single "1" on the grid (a 2D Kronecker delta). The interpolated field should equal the
    # kernel centered at that sample: sinc((x-x0)/dx) * sinc((y-y0)/dy).
    X = np.arange(-10.0, 11.0)
    Y = np.arange(-10.0, 11.0)

    f = np.zeros((len(Y), len(X)))
    ix, iy = 7, 12
    f[iy, ix] = 1.0

    x = np.linspace(X[0], X[-1], 401)
    y = np.linspace(Y[0], Y[-1], 401)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    fq = sinc_interp2(f, X, Y, xx, yy)
    expected = np.sinc(xx - X[ix]) * np.sinc(yy - Y[iy])

    np.testing.assert_allclose(fq, expected, rtol=0.0, atol=1e-12)


def test_sinc_interp2_agrees_with_definition() -> None:
    # Cross-check against the literal definition of 2D sinc interpolation:
    #   f̂(x,y) = Σ_j Σ_i f[j,i] * sinc((x-X[i])/dx) * sinc((y-Y[j])/dy)
    # Implemented below as explicit loops for a tiny problem (slow, but straightforward).
    rng = np.random.default_rng(1)

    X = np.arange(6.0)
    Y = np.arange(5.0)
    f = rng.standard_normal((len(Y), len(X)))

    x = np.array([0.3, 2.7, 4.1])
    y = np.array([1.2, 3.4, 0.5])

    fq = sinc_interp2(f, X, Y, x, y)

    dx, dy = X[1] - X[0], Y[1] - Y[0]
    fq_ref = np.empty_like(x, dtype=np.result_type(f, x, y))
    for k in range(len(x)):
        s = 0.0
        for j in range(len(Y)):
            for i in range(len(X)):
                s += (
                    f[j, i]
                    * np.sinc((x[k] - X[i]) / dx)
                    * np.sinc((y[k] - Y[j]) / dy)
                )
        fq_ref[k] = s

    np.testing.assert_allclose(fq, fq_ref, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("metadata_path, checkpoint_dir", configs)
def test_frequency_analysis_with_sinc(metadata_path, checkpoint_dir):

    eval_dir = os.path.join(rootdir, "evaluation", checkpoint_dir.split("checkpoints_")[-1])
    json_path = os.path.join(eval_dir, "freq_analysis.json")

    with open(json_path, "r") as jf:
        results = json.load(jf)

    room_indices = results['meta']['room_indices']
    bin_step = results['meta'].get('bin_step') or 1
    freqs_all = results['freqs']

    dataset = SoundFieldDataset(path=rootdir / metadata_path)
    selected_bin_indices = np.arange(0, dataset.bins_per_room, bin_step)

    x_lr = np.linspace(0, 1, dataset.lr_res)
    y_lr = np.linspace(0, 1, dataset.lr_res)
    x_hr = np.linspace(0, 1, dataset.hr_res)
    y_hr = np.linspace(0, 1, dataset.hr_res)
    xx, yy = np.meshgrid(x_hr, y_hr, indexing="xy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_nmse_sinc, avg_ncc_sinc = [], []

    for bin_idx in tqdm(selected_bin_indices, desc="Evaluating Sinc Baseline"):
        b_nmse_sinc, b_ncc_sinc = [], []
        for room_idx in room_indices:
            idx = room_idx * dataset.bins_per_room + int(bin_idx)
            gt_hr, lr_low, _ = dataset[idx]

            lr_real, lr_imag = lr_low[0].cpu().numpy(), lr_low[1].cpu().numpy()
            sinc_real = sinc_interp2(lr_real, x_lr, y_lr, xx, yy)
            sinc_imag = sinc_interp2(lr_imag, x_lr, y_lr, xx, yy)

            sinc_complex = torch.tensor(np.stack([sinc_real, sinc_imag], axis=0),
                                        dtype=torch.float32).unsqueeze(0).to(device)
            sinc_mag = complex_to_magnitude(sinc_complex)
            gt_mag = complex_to_magnitude(gt_hr.unsqueeze(0).to(device))

            metrics_sinc = get_metrics(gt_mag, sinc_mag)
            b_nmse_sinc.append(metrics_sinc['NMSE_dB'])
            b_ncc_sinc.append(metrics_sinc['NCC'])

        avg_nmse_sinc.append(float(np.mean(b_nmse_sinc)))
        avg_ncc_sinc.append(float(np.mean(b_ncc_sinc)))

    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 11,
        'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'axes.linewidth': 0.8, 'figure.dpi': 150,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    colors = ['#0072B2', '#D55E00', '#009E73', '#C1292E', '#E69F00']
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    labels = ['SFR3', 'Bicubic', 'Kernel', 'CVNN', 'Sinc']

    def plot_lines(ax, datasets):
        for data, color, ls, label in zip(datasets, colors, styles, labels):
            ax.plot(freqs_all, data, label=label, color=color, linestyle=ls, linewidth=1.6)

    fig_nmse, ax = plt.subplots(1, 1, figsize=(4, 2.55))
    plot_lines(ax, [results['nmse_dB']['diff_m'], results['nmse_dB']['bic_m'], results['nmse_dB']['ker_m'],
                    results['nmse_dB']['cvnn_m'], avg_nmse_sinc])
    ax.set_ylabel("NMSE [dB]")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title(f"NMSE: {dataset.lr_res}x{dataset.lr_res} to " f"{dataset.hr_res}x{dataset.hr_res}")
    ax.set_ylim(-72, -5)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(framealpha=0.9, edgecolor='0.8', fontsize=7, loc='lower right')
    fig_nmse.tight_layout()
    plt.show(block=True)

    fig_ncc, ax = plt.subplots(1, 1, figsize=(4, 3))
    plot_lines(ax,
               [results['ncc']['diff_m'], results['ncc']['bic_m'], results['ncc']['ker_m'],
                results['ncc']['cvnn_m'], avg_ncc_sinc])
    ax.set_ylabel("NCC")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title(f"NCC: {dataset.lr_res}x{dataset.lr_res} to " f"{dataset.hr_res}x{dataset.hr_res}")
    ax.set_ylim(bottom=0.91, top=1.01)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(framealpha=0.9, edgecolor='0.8', fontsize=9, loc='lower left')
    fig_ncc.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":

    for meta, ckpt in configs:
        test_frequency_analysis_with_sinc(os.path.join(rootdir, meta), os.path.join(rootdir, ckpt))
        test_visualize_sinc_interpolation(meta)