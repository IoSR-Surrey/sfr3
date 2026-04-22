from data import SoundFieldDataset
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch


class UenoKernel:
    """
    Kernel ridge interpolation using the 3D spherical kernel j0(k r) = sin(k r) / (k r),
    evaluated between points lying on a 2D plane (x,y).
    From the paper "Kernel Ridge Regression with Constraint of Helmholtz Equation for Sound Field Interpolation"
    (https://ieeexplore.ieee.org/document/8521334)
    """

    def __init__(self, lr_res=4, hr_res=32, c=343.0, dtype=torch.get_default_dtype()):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.lr_res = int(lr_res)
        self.hr_res = int(hr_res)
        self.c = float(c)

        # low-res grid coordinates
        x_lr = torch.linspace(0.0, 1.0, self.lr_res, device=self.device, dtype=self.dtype)
        y_lr = torch.linspace(0.0, 1.0, self.lr_res, device=self.device, dtype=self.dtype)
        gx_lr, gy_lr = torch.meshgrid(x_lr, y_lr, indexing='ij')
        self.coords_lr = torch.stack([gx_lr.flatten(), gy_lr.flatten()], dim=1)  # [n_train, 2]

        # high-res grid
        x_hr = torch.linspace(0.0, 1.0, self.hr_res, device=self.device, dtype=self.dtype)
        y_hr = torch.linspace(0.0, 1.0, self.hr_res, device=self.device, dtype=self.dtype)
        gx_hr, gy_hr = torch.meshgrid(x_hr, y_hr, indexing='ij')
        self.coords_hr = torch.stack([gx_hr.flatten(), gy_hr.flatten()], dim=1)  # [n_test, 2]

        self.n_train = self.coords_lr.shape[0]

    def spherical_j0_matrix(self, x1, x2, k):
        #matrix [i,j] = j0(k * ||x1_i - x2_j||) where j0(x)=sin(x)/x [sinc]
        dists = torch.cdist(x1, x2)  # [N1, N2]
        arg = k * dists
        j0 = torch.sinc(arg / torch.pi)
        return j0

    def reconstruct(self, lr_data, freq_hz, reg_lambda=1e-3):
        #lr_data: torch.Tensor, shape [B, 2, H, W] where H=W=lr_res (real, imag channels)
        B, C, H, W = lr_data.shape

        # wavenumber k
        k_val = 2.0 * np.pi * float(freq_hz) / float(self.c)
        k = torch.tensor(k_val, dtype=self.dtype, device=self.device)

        real_part = lr_data[:, 0].reshape(B, -1).to(device=self.device, dtype=self.dtype)
        imag_part = lr_data[:, 1].reshape(B, -1).to(device=self.device, dtype=self.dtype)
        p_lr = torch.complex(real_part, imag_part) # [B, n_train]
        P = p_lr.T # [n_train, B]

        # kernel matrices
        K_tt = self.spherical_j0_matrix(self.coords_lr, self.coords_lr, k) # [n_train, n_train]
        K_st = self.spherical_j0_matrix(self.coords_hr, self.coords_lr, k) # [n_test, n_train]

        eye = torch.eye(self.n_train, dtype=self.dtype, device=self.device)
        A = K_tt + (float(reg_lambda) * eye)
        A_cplx = A.to(dtype=P.dtype)
        alpha = torch.linalg.solve(A_cplx, P)
        K_st_cplx = K_st.to(dtype=P.dtype)
        p_hr = (torch.matmul(K_st_cplx, alpha)).T  # [B, n_test]

        # reshape to [B, hr_res, hr_res]
        p_hr = p_hr.view(B, self.hr_res, self.hr_res)

        return torch.stack([p_hr.real, p_hr.imag], dim=1)


def complex_to_magnitude(x):
    #convert [2, H, W] real/imag to magnitude
    return torch.sqrt(x[0] ** 2 + x[1] ** 2)


def run_single_inference(metadata_path):
    print(f"Loading dataset from {metadata_path}...")
    dataset = SoundFieldDataset(path=metadata_path)

    # random sample
    idx = random.randint(0, len(dataset) - 1)
    gt_hr, lr_low, _ = dataset[idx]

    # physical frequency
    fs = getattr(dataset, 'fs', 16000)
    n_fft = getattr(dataset, 'n_fft', 2048)
    bin_idx = idx % dataset.bins_per_room
    # 0Hz is removed, offset bin index by 1
    freq_hz = (bin_idx + 1) * fs / n_fft

    print(f"\n--- Random Inference Test ---")
    print(f"Room Index: {idx // dataset.bins_per_room}")
    print(f"Data Index: {bin_idx} (Physical Bin: {bin_idx + 1})")
    print(f"Frequency: {freq_hz:.2f} Hz")
    print(f"Input LR: {lr_low.shape}")
    print(f"GT HR: {gt_hr.shape}")

    lr_res = dataset.lr_res
    hr_res = dataset.hr_res

    kernel = UenoKernel(lr_res=lr_res, hr_res=hr_res)
    lr_batch = lr_low.unsqueeze(0).to(kernel.device)
    recon_hr = kernel.reconstruct(lr_batch, freq_hz=freq_hz)
    recon_hr = recon_hr.squeeze(0).cpu()  # [2, highres, highres]
    lr_mag = complex_to_magnitude(lr_low)
    lr_grid = torch.nn.functional.interpolate(lr_mag.unsqueeze(0).unsqueeze(0), size=(hr_res, hr_res), mode='nearest').squeeze()
    gt_mag = complex_to_magnitude(gt_hr)
    recon_mag = complex_to_magnitude(recon_hr)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(lr_grid, origin='lower', cmap='inferno')
    ax[0].set_title("Input (Low Res)")
    ax[1].imshow(recon_mag, origin='lower', cmap='inferno')
    ax[1].set_title(f"Kernel Reconstruction\n({freq_hz:.1f} Hz)")
    ax[2].imshow(gt_mag, origin='lower', cmap='inferno')
    ax[2].set_title("Ground Truth")
    plt.suptitle(f"Ueno Kernel Inference Check - {freq_hz:.1f} Hz")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, default='datasets/dataset_4to32_500/test/metadata.json') #alt: datasets/dataset_8to64_1000/test/metadata.json
    args = parser.parse_args()

    run_single_inference(args.metadata)