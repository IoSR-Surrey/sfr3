import json
import numpy as np
import copy
import torch
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

dir = '/nas/home/fronchini/complex-sound-field/figures'

def NMSE_fun(pred, gt):
    
    pred = pred.reshape(40, -1)
    gt = gt.reshape(40, -1)

    num = np.sum(np.power(np.abs(np.abs(gt) - np.abs(pred)),2),axis=-1)
    den = np.sum(np.power(np.abs(np.abs(gt)),2),axis=-1)
    
    nmse = num/den
    return nmse


def NMSE_complex_fun(pred, gt):
    pred = pred.reshape(40, -1)
    gt = gt.reshape(40, -1)

    num = np.sum(np.power(np.abs(gt - pred), 2), axis=-1)
    den = np.sum(np.power(np.abs(gt), 2), axis=-1)

    nmse = num / den
    return nmse


def SSIM_fun(pred, gt):
    
    res = np.zeros((40, 1))
    
    pred = pred.reshape(40, -1)
    gt = gt.reshape(40, -1)
    
    for freq in range(pred.shape[0]):
        data_range = np.abs(pred[freq, :]).max() - np.abs(pred[freq, :]).min()
        res[freq] = ssim(np.abs(pred[freq]), np.abs(gt[freq]), data_range=data_range)
    
    return res

   
 
def load_config(config_filepath):
    """ Load a session configuration from a JSON-formatted file.

    Args:
    config_filepath: string
    Returns: dict

    """

    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        print('No readable config file at path: ' + config_filepath)
    else:
        with config_file:
            return json.load(config_file)
        
def save_config(data, config_output_path):
    
    with open(config_output_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Configuration file saved in {config_output_path}")
    

def get_frequencies():
    """Loads the frequency numbers found at 'util/frequencies.txt'.

    Returns: list

    """
    freqs_path = 'util/frequencies.txt'
    with open(freqs_path) as f:
        freqs = [[int(freq) for freq in line.strip().split(' ')] for line in f.readlines()][0]

    return freqs

def preprocessing(factor, sf, mask):
    """ Perfom all preprocessing steps.

        Args:
        factor: int
        sf: np.ndarray
        mask: np.ndarray

        Returns: np.ndarray, np.ndarray

        """

    # Downsampling
    downsampled_sf = downsampling(factor, sf)

    # Masking
    masked_sf = apply_mask(downsampled_sf, mask)

    # # Scaling masked sound field - we'll see thia later
    # scaled_sf = scale(masked_sf) # normalizzare parte reale ed immaginaria separate

    # Upsampling no-scaled sound field and mask
    irregular_sf, mask = upsampling(factor, masked_sf, mask) # irregular_sf shape (32, 32, 40), # mask shape (32, 32, 40
    
    
    return irregular_sf, mask


def downsampling(dw_factor, input_sfs):
    """ Downsamples sound fields given a downsampling factor.

        Args:
        dw_factor: int
        input_sfs: np.ndarray

        Returns: np.ndarray

        """
        
    
    return input_sfs[0:input_sfs.shape[0]:dw_factor, 0:input_sfs.shape[1]:dw_factor, :] 
    # we are consideing only one item per time, they are probably considering more than 1?


def apply_mask(input_sf, mask):
    """ Apply masks to sound fields.

        Args:
        input_sfs: np.ndarray
        masks: np.ndarray

        Returns: np.ndarray

        """
    
    #masked_sfs = []
    #for sf, mk in zip(input_sfs, masks):
    
    aux_sf = copy.deepcopy(input_sf)
    aux_sf[mask==0] = 0
    
    for i in range(input_sf.shape[2]):
        #aux_max = aux_sf[:, :, i].max()
        aux_max = 0 
        input_sf[:, :, i][mask[:, :, i]==0] = aux_max
        #masked_sfs.append(sf)

    return input_sf

def upsampling(lr, lr_res, hr_res):

    upsampled = np.zeros((hr_res, hr_res), dtype=lr.dtype)
    scale = (hr_res - 1) / (lr_res - 1)
    for r in range(lr_res):
        for c in range(lr_res):
            upsampled[round(r * scale), round(c * scale)] = lr[r, c]

    return upsampled