from torch.utils.data import Dataset
import pyroomacoustics as pra
from tqdm import tqdm
import numpy as np
import argparse
import shutil
import random
import torch
import json
import math
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lowres', type=int, default=4)
    parser.add_argument('--highres', type=int, default=32)
    parser.add_argument('--maxfreq', type=int, default=500)
    parser.add_argument('--len_rir', type=int, default=2048)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--fsampl', type=int, default=16000)
    parser.add_argument('--out_dir', type=str, default='dataset_4to32_500')
    parser.add_argument('--generate', type=int, choices=[0,1], default=0)
    parser.add_argument('--show', type=int, choices=[0,1], default=1)
    parser.add_argument('--seed', type=int, default=None)

    arguments, _ = parser.parse_known_args()
    return arguments

args = parse_args()

lowres = args.lowres
highres = args.highres
maxfreq = args.maxfreq
len_rir = args.len_rir
n_fft = args.n_fft
fsampl = args.fsampl
generate = args.generate
show = args.show
out_dir = args.out_dir

def generate_sound_field(room_dim, mic_region, grid_res=64, lr_res=16, max_freq=1500):
    """
    Generate an HR dense grid and an LR grid (coarse mic array).
    Returns:
      hr_tensor: torch.Tensor [num_freq_bins, 2, grid_res, grid_res]  (real/imag)
      lr_tensor: torch.Tensor [num_freq_bins, 2, lr_res, lr_res]      (real/imag)
      room_dim, mic_region, rt60, source_position
    """

    room = None
    rt60 = 0
    e_absorption = None

    while True:
        try:
            # Determine Room Dimensions
            if room_dim is None:
                # bounds to reliably hit 250m3 - 500m3
                # Above DIN 18041 threshold for non-small rooms and below the maximum recommended volume for A4 rooms
                l = round(random.uniform(6.0, 12.0), 2)
                w = round(random.uniform(6.0, 12.0), 2)
                h = round(random.uniform(2.5, 4.5), 2)

                volume = l * w * h

                # volume check
                if not (250 <= volume <= 500):
                    continue  # skip and try generating a new size

                current_dim = (l, w, h)
            else:
                current_dim = room_dim
                volume = current_dim[0] * current_dim[1] * current_dim[2]

            # T60 formula in standard DIN 18041 for room type A4: Teaching/Communication with more speakers
            # Max T60 (V=500m3) is 0.5617s
            # Also considered appropriate by UK's BB93 standard for classrooms, offices, kitchens, libraries...
            # as well as open office spaces in public areas as described by ISO 22955:2021
            rt60 = 0.26 * math.log10(volume) - 0.14

            # absorption based on Sabine's formula
            e_absorption, _ = pra.inverse_sabine(rt60, current_dim)

            # dense HR room (for dense mic grid)
            room = pra.ShoeBox(
                current_dim,
                fs=fsampl,
                max_order=30,
                materials=pra.Material(e_absorption)
            )
            room_dim = current_dim
            break

        except ValueError:
            # if inverse_sabine fails
            if room_dim is not None:
                print(f"Warning: Provided room_dim {room_dim} caused an acoustic error. Adjusting to random...")
                room_dim = None
            continue

    # mic region definition
    if mic_region is None:
        # 1m x 1m x 1m exclusion volume (at least 0.5m from walls/ceiling)
        mx0 = round(random.uniform(0.5, room_dim[0] - 1.5), 2)
        my0 = round(random.uniform(0.5, room_dim[1] - 1.5), 2)
        mz0 = round(random.uniform(0.5, room_dim[2] - 1.5), 2)

        # exclusion box for the source
        mic_region = (
            (mx0, round(mx0 + 1.0, 2)),
            (my0, round(my0 + 1.0, 2)),
            (mz0, round(mz0 + 1.0, 2))
        )

    (mx0, mx1), (my0, my1), (mz0, mz1) = mic_region

    # source placement
    while True:
        source_position = np.array([
            random.uniform(0.5, room_dim[0] - 0.5),
            random.uniform(0.5, room_dim[1] - 0.5),
            random.uniform(0.5, room_dim[2] - 0.5)
        ])
        # check if source is inside exclusion box
        in_x = mx0 <= source_position[0] <= mx1
        in_y = my0 <= source_position[1] <= my1
        in_z = mz0 <= source_position[2] <= mz1
        if not (in_x and in_y and in_z):
            break
    room.add_source(source_position.tolist())

    # dense microphone grid inside mic_region
    z_slice = (mz0 + mz1) / 2.0
    x_hr = np.linspace(mx0, mx1, grid_res)
    y_hr = np.linspace(my0, my1, grid_res)
    x_mesh_hr, y_mesh_hr = np.meshgrid(x_hr, y_hr)
    mics_hr = np.vstack((x_mesh_hr.flatten(), y_mesh_hr.flatten(), np.full(x_mesh_hr.size, z_slice)))
    room.add_microphone_array(pra.MicrophoneArray(mics_hr, room.fs))

    # compute RIRs for HR
    room.compute_rir()

    # Prepare FFT window
    window = np.hanning(len_rir)

    # Determine frequency bins up to max_freq (same for HR and LR)
    max_freq_idx = int(np.ceil(max_freq / room.fs * n_fft)) + 1
    num_freq_bins = min(max_freq_idx, n_fft // 2 + 1)

    # HR pressure maps container
    pressure_hr = np.zeros((num_freq_bins, grid_res, grid_res), dtype=np.complex64)

    # iterate over HR microphones and fill pressure_hr
    for mic_idx, rir in enumerate(room.rir):
        # each rir is a list (one per source); we used a single source so take rir[0]
        h = np.asarray(rir[0], dtype=float)
        if h.size >= len_rir:
            h_cut = h[:len_rir]
        else:
            h_cut = np.zeros(len_rir, dtype=float)
            h_cut[:h.size] = h

        # apply window to reduce spectral leakage
        h_windowed = h_cut * window
        hfreq = np.fft.fft(h_windowed, n=n_fft)[:num_freq_bins]

        row = mic_idx // grid_res
        col = mic_idx % grid_res
        pressure_hr[:, row, col] = hfreq

    # re-create the same room with a coarse mic grid inside the same mic_region
    room_lr = pra.ShoeBox(
        room_dim,
        fs=fsampl,
        max_order=30,
        materials=pra.Material(e_absorption)
    )
    room_lr.add_source(source_position.tolist())

    x_lr = np.linspace(mx0, mx1, lr_res)
    y_lr = np.linspace(my0, my1, lr_res)
    x_mesh_lr, y_mesh_lr = np.meshgrid(x_lr, y_lr)
    mics_lr = np.vstack((x_mesh_lr.flatten(), y_mesh_lr.flatten(), np.full(x_mesh_lr.size, z_slice)))
    room_lr.add_microphone_array(pra.MicrophoneArray(mics_lr, room_lr.fs))

    room_lr.compute_rir()

    pressure_lr = np.zeros((num_freq_bins, lr_res, lr_res), dtype=np.complex64)
    for mic_idx, rir in enumerate(room_lr.rir):
        h = np.asarray(rir[0], dtype=float)
        if h.size >= len_rir:
            h_cut = h[:len_rir]
        else:
            h_cut = np.zeros(len_rir, dtype=float)
            h_cut[:h.size] = h
        h_windowed = h_cut * window
        hfreq = np.fft.fft(h_windowed, n=n_fft)[:num_freq_bins]
        row = mic_idx // lr_res
        col = mic_idx % lr_res
        pressure_lr[:, row, col] = hfreq

    hr_real = torch.tensor(pressure_hr.real, dtype=torch.float32)
    hr_imag = torch.tensor(pressure_hr.imag, dtype=torch.float32)
    hr_tensor = torch.stack([hr_real, hr_imag], dim=1)  # [bins, 2, H_hr, W_hr]

    lr_real = torch.tensor(pressure_lr.real, dtype=torch.float32)
    lr_imag = torch.tensor(pressure_lr.imag, dtype=torch.float32)
    lr_tensor = torch.stack([lr_real, lr_imag], dim=1)  # [bins, 2, H_lr, W_lr]

    return hr_tensor, lr_tensor, room_dim, mic_region, rt60, source_position


def generate_and_save_dataset(num_rooms=100, output_dir='.', basename='sound_field_dataset', rooms_sub='dataset_rooms'):
    """
    Generate `num_rooms` rooms in two combined NumPy .npy memmaps inside `output_dir`.
    """

    os.makedirs(output_dir, exist_ok=True)
    out_rooms_dir = os.path.join(output_dir, rooms_sub)
    os.makedirs(out_rooms_dir, exist_ok=True)

    per_room_paths = []
    metadata_list = []
    assert num_rooms > 0
    bins = ch = hr_h = hr_w = lr_h = lr_w = 0

    for i in tqdm(range(num_rooms), desc=f"Generating {os.path.basename(output_dir)}"):
        hr_tensor, lr_tensor, room_dim, mic_region, rt60, source_position = generate_sound_field(
            None, None, grid_res=highres, lr_res=lowres, max_freq=maxfreq
        )

        # remove 0Hz bin from both HR and LR
        hr_tensor_no0 = hr_tensor[1:].clone().to(torch.float32)
        lr_tensor_no0 = lr_tensor[1:].clone().to(torch.float32)

        if i == 0:
            bins, ch, hr_h, hr_w = hr_tensor_no0.shape
            _, _, lr_h, lr_w = lr_tensor_no0.shape

        room_dict = {
            "hr": hr_tensor_no0,
            "lr": lr_tensor_no0,
        }

        room_path = os.path.join(out_rooms_dir, f"room_{i:05d}.pt")
        torch.save(room_dict, room_path)
        per_room_paths.append(room_path)

        metadata_list.append({
            "room_index": i,
            "room_dim": list(room_dim),
            "mic_region": [list(mic_region[0]), list(mic_region[1]), list(mic_region[2])],
            "t60": float(rt60),
            "source_position": [float(x) for x in source_position],
        })

        del hr_tensor, lr_tensor, hr_tensor_no0, lr_tensor_no0, room_dict

    combined_hr_path = os.path.join(output_dir, f"{basename}_hr.npy")
    combined_lr_path = os.path.join(output_dir, f"{basename}_lr.npy")

    mem_hr = np.lib.format.open_memmap(combined_hr_path, mode='w+', dtype='float32',
                                       shape=(num_rooms, bins, ch, hr_h, hr_w))
    mem_lr = np.lib.format.open_memmap(combined_lr_path, mode='w+', dtype='float32',
                                       shape=(num_rooms, bins, ch, lr_h, lr_w))

    for i, p in enumerate(tqdm(per_room_paths, desc="Writing to memmap", leave=False)):
        d = torch.load(p, map_location='cpu', weights_only=False)
        mem_hr[i] = d["hr"].numpy()[:bins]
        mem_lr[i] = d["lr"].numpy()[:bins]
        del d

    del mem_hr, mem_lr

    meta = {
        # dataset layout
        "n_rooms": num_rooms,
        "bins_per_room": bins,
        "hr_res": hr_h,
        "lr_res": lr_h,
        # simulation parameters needed to reproduce
        "fs": float(fsampl),
        "n_fft": int(n_fft),
        "len_rir": int(len_rir),
        "max_freq": int(maxfreq),
        "max_order": 30,
        # relative npy paths
        "npy_hr": os.path.relpath(combined_hr_path, output_dir),
        "npy_lr": os.path.relpath(combined_lr_path, output_dir),
        # per-room reproduction data
        "rooms": metadata_list,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as js:
        json.dump(meta, js, indent=2)

    shutil.rmtree(out_rooms_dir)

    return meta


class SoundFieldDataset(Dataset):
    def __init__(self, path='metadata.json'):
        with open(path, 'r') as f:
            meta = json.load(f)
        base_dir = os.path.dirname(os.path.abspath(path))
        self.hr_path = os.path.join(base_dir, meta['npy_hr'])
        self.lr_path = os.path.join(base_dir, meta['npy_lr'])
        self.bins_per_room = int(meta['bins_per_room'])
        self.num_rooms = int(meta['n_rooms'])
        self.hr_res = int(meta['hr_res'])
        self.lr_res = int(meta['lr_res'])
        self.total_samples = self.num_rooms * self.bins_per_room
        self.fsampl = float(meta['fs'])
        self.n_fft = int(meta['n_fft'])

        self.mem_hr = None
        self.mem_lr = None

    def __len__(self):
        return self.total_samples

    def _init_memmaps(self):
        """open memmaps only once per worker process."""
        if self.mem_hr is None:
            self.mem_hr = np.lib.format.open_memmap(self.hr_path, mode='r')
        if self.mem_lr is None:
            self.mem_lr = np.lib.format.open_memmap(self.lr_path, mode='r')

    def __getitem__(self, idx):
        self._init_memmaps()
        room_index = idx // self.bins_per_room
        bin_index = idx % self.bins_per_room

        arr_hr = np.array(self.mem_hr[room_index, bin_index], copy=True)
        arr_lr = np.array(self.mem_lr[room_index, bin_index], copy=True)

        gt_hr = torch.from_numpy(arr_hr).float()
        lr_low = torch.from_numpy(arr_lr).float()

        # ISOBEL-style scaling (Kristoffersen et al. 2021)
        slice_max = torch.max(torch.abs(gt_hr))
        # scales the values into the [-1, 1] range
        gt_hr = gt_hr / slice_max
        lr_low = lr_low / slice_max

        return gt_hr, lr_low



if __name__ == '__main__':

    dataset_specs = {
        'train': 5000,
        'val': 500,
        'test': 500
    }

    seed = args.seed if args.seed is not None else random.randint(0, 1000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    metadata_store = {}

    for dataset_type, num_rooms in dataset_specs.items():
        target_dir = os.path.join(out_dir, dataset_type)
        print(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        if generate:
            print(f"Starting generation: dataset_type={dataset_type}, num_rooms={num_rooms}, out_dir={target_dir}")
            metadata = generate_and_save_dataset(num_rooms=int(num_rooms),
                                                       output_dir=target_dir,
                                                       basename=f"sound_field_{dataset_type}")
            metadata_store[dataset_type] = metadata


    for dataset_type in dataset_specs.keys():
        if dataset_type not in metadata_store:
            metadata_path = os.path.join(out_dir, dataset_type, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata_store[dataset_type] = json.load(f)
                print(f"Recovered metadata for {dataset_type} from {metadata_path}")
            else:
                print(f"Warning: No metadata found for {dataset_type} at {metadata_path}")

    train_meta = metadata_store.get('train')
    if train_meta and train_meta.get('n_rooms', 0) > 0 and show:
        import matplotlib.pyplot as plt

        train_dir = os.path.join(out_dir, 'train')
        hr_memmap = np.lib.format.open_memmap(os.path.join(train_dir, train_meta['npy_hr']), mode='r')
        lr_memmap = np.lib.format.open_memmap(os.path.join(train_dir, train_meta['npy_lr']), mode='r')

        chosen_idx = random.randint(0, int(train_meta['n_rooms']) - 1)
        bin_idx = random.randint(0, int(train_meta['bins_per_room']) - 1)

        hr_bin = torch.from_numpy(np.array(hr_memmap[chosen_idx, bin_idx]))  # [2, H_hr, W_hr]
        lr_bin = torch.from_numpy(np.array(lr_memmap[chosen_idx, bin_idx]))  # [2, H_lr, W_lr]

        print(f"Displaying room {chosen_idx}, bin {bin_idx}")

        for title, mag in [
            (f"Low-res magnitude (room {chosen_idx}, bin {bin_idx})",  torch.sqrt(lr_bin[0]**2 + lr_bin[1]**2).numpy()),
            (f"High-res magnitude (room {chosen_idx}, bin {bin_idx})", torch.sqrt(hr_bin[0]**2 + hr_bin[1]**2).numpy()),
        ]:
            plt.figure()
            plt.title(title)
            plt.imshow(mag, origin='lower')
            plt.axis('off')
            plt.show()
    else:
        print("No train rooms available to display. Make sure that metadata.json files exist.")