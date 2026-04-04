from dataset import generate_mask
import sfun_torch
import numpy
import torch
import utils
import sys
import os

folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, folder)


def load_complexunet(dataset, device, config_path=None, model_path=None):

    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.json")
    config = utils.load_config(config_path)
    session_id = config["training"]["session_id"]

    if model_path is None:
        model_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", session_id, "ComplexUNet"))
    model = sfun_torch.ComplexUnet(config["training"], n_freq=dataset.bins_per_room).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def complexunet_infer(cvnn_model, dataset, room_idx, bin_idx, device):

    bins = dataset.bins_per_room
    hr_res = dataset.hr_res
    lr_res = dataset.lr_res

    lr_up_list = []
    for b in range(bins):
        gt_hr, lr_low, _ = dataset[room_idx * bins + b]
        lr_c = lr_low[0].numpy() + 1j * lr_low[1].numpy()
        lr_up_list.append(utils.upsampling(lr_c, lr_res, hr_res))

    irregular_sf = torch.from_numpy(numpy.stack(lr_up_list, axis=0).astype(numpy.complex64))
    mask_np = generate_mask(hr_res, lr_res, bins)
    mask = torch.from_numpy(mask_np).permute(2, 0, 1).to(torch.complex64)
    sf_masked = torch.cat((irregular_sf, mask), dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = cvnn_model(sf_masked)

    sf_magnitude = prediction[0, bin_idx].abs().unsqueeze(0).unsqueeze(0)
    return sf_magnitude