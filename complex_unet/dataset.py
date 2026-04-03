from torch.utils.data import Dataset
import os
import sys
import numpy as np
import torchaudio
import random
import torch
import glob
import scipy
import copy


import utils

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)
from data import SoundFieldDataset as SFR3dataset

def norm_sf_complex(input_ten):
    gt_real_norm = (input_ten.real - input_ten.real.mean()) / input_ten.real.std()
    gt_imag_norm = (input_ten.imag - input_ten.imag.mean()) / input_ten.imag.std()
    norm_ten = gt_real_norm.type(torch.complex64) + 1j * gt_imag_norm.type(torch.complex64)
    return norm_ten


def generate_mask(hr_res, lr_res, channels):
    mask_slice = np.zeros((hr_res, hr_res, 1))
    scale = (hr_res - 1) / (lr_res - 1)
    for r in range(lr_res):
        for c in range(lr_res):
            mask_slice[round(r * scale), round(c * scale), 0] = 1

    mask = np.repeat(mask_slice, channels, axis=2)

    return mask

class SoundFieldDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        do_normalize=True,
    ):
        self.ds = SFR3dataset(path=metadata_path)
        self.hr_res = self.ds.hr_res
        self.lr_res = self.ds.lr_res
        self.do_normalize = do_normalize
        self.ds._init_memmaps()
        self.mask = generate_mask(self.hr_res, self.lr_res, self.ds.bins_per_room)

    def __len__(self):
        return self.ds.num_rooms

    def __getitem__(self, item):
        gt_list = []
        lr_up_list = []

        for bin_idx in range(self.ds.bins_per_room):
            gt_hr, lr_low, _ = self.ds[item * self.ds.bins_per_room + bin_idx]
            hr_c = gt_hr[0].numpy() + 1j * gt_hr[1].numpy()
            lr_c = lr_low[0].numpy() + 1j * lr_low[1].numpy()
            lr_up_list.append(utils.upsampling(lr_c, self.lr_res, self.hr_res))
            gt_list.append(hr_c)

        irregular_sf = torch.from_numpy(np.stack(lr_up_list, axis=0).astype(np.complex64))
        sf_gt = torch.from_numpy(np.stack(gt_list, axis=0).astype(np.complex64))
        mask = torch.from_numpy(self.mask).permute(2, 0, 1).to(torch.complex64)
        sf_masked = torch.cat((irregular_sf, mask), dim=0)

        return sf_masked, sf_gt