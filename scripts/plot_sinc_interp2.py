import numpy as np
import matplotlib.pyplot as plt

from .sinc_interp import sinc_interp2

import torch
import random
import pytest
from pathlib import Path

rootdir = Path(__file__).resolve().parent.parent
from data import SoundFieldDataset

configs = [("datasets/dataset_4to32_500/test/metadata.json", "checkpoints/checkpoints_4to32_500"),
           ("datasets/dataset_8to64_1000/test/metadata.json", "checkpoints/checkpoints_8to64_1000")]

@pytest.mark.parametrize("metadata_path", [cfg[0] for cfg in configs])
def test_visualize_sinc_interpolation(metadata_path, seed=123):

    dataset = SoundFieldDataset(path=rootdir / metadata_path)

    random.seed(seed)
    room_idx = random.randint(0, dataset.num_rooms - 1)
    bin_idx = random.randint(0, dataset.bins_per_room - 1)
    idx = room_idx * dataset.bins_per_room + bin_idx
    gt_hr, lr_low, _ = dataset[idx]
    freq_hz = (bin_idx + 1) * dataset.fsampl / dataset.n_fft

    x_lr = np.linspace(0, 1, dataset.lr_res)
    y_lr = np.linspace(0, 1, dataset.lr_res)
    x_hr = np.linspace(0, 1, dataset.hr_res)
    y_hr = np.linspace(0, 1, dataset.hr_res)
    xx, yy = np.meshgrid(x_hr, y_hr, indexing="xy")

    s_real = sinc_interp2(lr_low[0].numpy(), x_lr, y_lr, xx, yy)
    s_imag = sinc_interp2(lr_low[1].numpy(), x_lr, y_lr, xx, yy)
    lr_mag = torch.sqrt(lr_low[0] ** 2 + lr_low[1] ** 2).numpy()
    gt_mag = torch.sqrt(gt_hr[0] ** 2 + gt_hr[1] ** 2).numpy()
    sinc_mag = np.sqrt(s_real ** 2 + s_imag ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    plots = [(lr_mag, f"Low Res ({dataset.lr_res}x{dataset.lr_res})"), (gt_mag, f"Ground Truth ({freq_hz:.1f} Hz)"),
             (sinc_mag, f"Sinc Interp ({dataset.hr_res}x{dataset.hr_res})")]

    for ax, (mag, title) in zip(axes, plots):
        im = ax.imshow(mag, origin='lower', extent=[0, 1, 0, 1], cmap='inferno')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    plt.tight_layout()
    plt.show(block=True)

def main() -> None:
    # Example script showing how to call `sinc_interp2` and visualize the result.
    
    # Coarse grid samples
    X = np.linspace(0.0, 1.0, 16, endpoint=False)
    Y = np.linspace(0.0, 1.0, 16, endpoint=False)
    XX, YY = np.meshgrid(X, Y, indexing="xy")
    f = np.cos(2 * np.pi * 3 * XX) * np.sin(2 * np.pi * 2 * YY)

    # Fine query grid
    x = np.linspace(0.0, 1.0, 64, endpoint=False)
    y = np.linspace(0.0, 1.0, 64, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    fq = sinc_interp2(f, X, Y, xx, yy)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].imshow(f, origin="lower", aspect="auto")
    ax[0].set_title("Coarse samples")
    ax[1].imshow(fq, origin="lower", aspect="auto")
    ax[1].set_title("Sinc interpolated")
    plt.show()


if __name__ == "__main__":
    main()

