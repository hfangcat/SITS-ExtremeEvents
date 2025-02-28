"""
Dataset class for RaVAEn dataset (hdf5 version)

For training, validation and testing, 
unlike synthetic dataset, in which we generate data based on one base image on-the-fly,
we need to load the data from the hdf5 file.

We saved the patches in RGB bands (not BGR) in the hdf5 file using the preprocessing script
`data/scripts/prepare_ravaen_data.py`, see README.md for more details.
"""

"""
Training: image sequence (t1 - t5) + same augmentation + z-score normalization
Validation: image sequence (t1 - t5) + z-score normalization
"""

# Import libraries
import os
import numpy as np
import random

import torch
from tqdm import tqdm
import h5py
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from .utils import gauss_noise, saturation, brightness, contrast


class RaVAEnHDF5Dataset(Dataset):
    """
    Dataset class for RaVAEn dataset (hdf5 version).

    Args:
    - hdf5_file_path (str): path to the hdf5 file
    - csv_file_path (str): path to the csv file
    - config (dict): Dictionary containing the configuration parameters.
                     In this case, following keys are used:
                     - base_img (str): choice of base image
                     - change_threshold (float): threshold for change ratio
                     Normalization:
                     - mean: mean value for z-score normalization
                     - std: standard deviation for z-score normalization
    - mode (str): mode of the dataset (train, val, test)

    Returns:
    - sample (dict): dicionary containing the following keys:
                     - t1, t2, t3, t4, t5: Images.
                     - label: change (1) or no change (0).
                     - change_ratio: the ratio of change pixels in the patch.
    """

    # The original hdf5 file is ~18GB, so it is okay to load the entire dataset into memory
    # The original 60 "thin" DGX-A100 nodes are each equipped with 8 NVIDIA® A100 Tensor Core GPUs, 2 AMD Epyc™ 7742 CPUs, 1 TB RAM and 15 TB of local NVMe SSD storage
    def __init__(self, hdf5_file_path, csv_file_path, mode, config):
        super().__init__()
        self.hdf5_file_path = hdf5_file_path
        self.csv_file_path = csv_file_path
        self.mode = mode
        self.config = config

        # Load the hdf5 file and csv file
        # considering our A100, it would be fine to load the entire dataset into memory
        # self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')
        self.csv_file = pd.read_csv(self.csv_file_path)

        self.t1 = []
        self.t2 = []
        self.t3 = []
        self.t4 = []
        self.t5 = []
        self.change_mask = []

        # 2024/4/24: add event
        self.event = []

        # 2024/4/19: Load the patches from the hdf5 file
        with h5py.File(self.hdf5_file_path, 'r') as hdf:
            for key in tqdm(hdf.keys()):
                # Check if the key is in the csv file
                col = key.split('_')[-1]
                row = key.split('_')[-2]
                event = '_'.join(key.split('_')[:-2])
                assert ((self.csv_file['event'] == event) & (self.csv_file['row'] == int(row)) & (self.csv_file['col'] == int(col))).any(), f"Key {key} is not in the csv file"

                t1 = hdf[key]['t1'][:].transpose(1, 2, 0)
                t2 = hdf[key]['t2'][:].transpose(1, 2, 0)
                t3 = hdf[key]['t3'][:].transpose(1, 2, 0)
                t4 = hdf[key]['t4'][:].transpose(1, 2, 0)
                t5 = hdf[key]['t5'][:].transpose(1, 2, 0)
                change_mask = hdf[key]['change'][:].transpose(1, 2, 0)

                self.t1.append(t1)
                self.t2.append(t2)
                self.t3.append(t3)
                self.t4.append(t4)
                self.t5.append(t5)
                self.change_mask.append(change_mask)

                # 2024/4/24: add event
                self.event.append(event)

            # Close the hdf5 file
            hdf.close()


    def __len__(self):
        return len(self.csv_file)
    

    def __getitem__(self, idx):
        t1 = self.t1[idx]
        t2 = self.t2[idx]
        t3 = self.t3[idx]
        t4 = self.t4[idx]
        t5 = self.t5[idx]

        change_mask = self.change_mask[idx]

        # 2024/4/24: add event
        event = self.event[idx]
        fire_list = ['fire_camp', 'fire_carr', 'fire_czu', 'fire_mallacoota', 'fire_riveaux']
        flood_list = ['EMSR260_02VIADANA', 'EMSR271_02FARKADONA', 'EMSR324_04LESPIGNAN', 'EMSR333_02PORTOPALO']
        hurricane_list = ['hurricane_dorian_bahamas', 'hurricane_irma_barbuda', 'hurricane_michael_sanblas2', 'hurricane_harvey_otey', 'hurricane_maria_stcroix']
        landslide_list = ['landslide_bigsur', 'landslide_fagraskogarfjall', 'landslide_santalucia', 'landslide_sierra', 'landslide_xinmo']
        if self.event[idx] in fire_list:
            event = 0
        elif self.event[idx] in flood_list:
            event = 1
        elif self.event[idx] in hurricane_list:
            event = 2
        elif self.event[idx] in landslide_list:
            event = 3
        else:
            assert False, "Unknown disaster type"

        # Calculate the change ratio
        change_ratio = (change_mask == 1).sum() / change_mask.size

        # Decide the change label (1: change, 0: no change)
        if change_ratio > self.config['change_threshold']:
            change = 1
        else:
            change = 0

        img_seq = [t1, t2, t3, t4, t5]

        """Get the same augmentation for the image time series (only for training set)"""
        if self.mode == 'train':
            # Similar to the synthetic dataset
            # we only consider Gaussian blur, brightness, contrast, and saturation

            # 3. Gaussian blur + 4. Brightness, contrast, saturation
            for i in range(len(img_seq)):
                if random.random() > 0.98:
                    if random.random() > 0.985:
                        img_seq[i] = gauss_noise(img_seq[i])
                    elif random.random() > 0.985:
                        img_seq[i] = cv2.blur(img_seq[i], (3, 3))
                elif random.random() > 0.98:
                    if random.random() > 0.985:
                        img_seq[i] = saturation(img_seq[i], 0.9+random.random()*0.2)
                    elif random.random() > 0.985:
                        img_seq[i] = brightness(img_seq[i], 0.9+random.random()*0.2)
                    elif random.random() > 0.985:
                        img_seq[i] = contrast(img_seq[i], 0.9+random.random()*0.2)

        """Z-score normalization"""
        for i in range(len(img_seq)):
            img_seq[i] = self.zscore_norm(img_seq[i], self.config['datasets']['norm']['mean'], self.config['datasets']['norm']['std'])

        """Choose base image"""
        if self.config['base_img'] == 't1':
            base_img = img_seq[0].copy()
        elif self.config['base_img'] == 't4':
            base_img = img_seq[3].copy()
        elif self.config['base_img'] == 'mean':
            base_img = ((img_seq[0] + img_seq[1] + img_seq[2] + img_seq[3]) / 4).copy()

        """Convert the images to tensors"""
        for i in range(len(img_seq)):
            img_seq[i] = torch.from_numpy(img_seq[i].transpose((2, 0, 1)).copy()).float()
        
        base_img = torch.from_numpy(base_img.transpose((2, 0, 1)).copy()).float()

        if change == 1:
            change_mask = torch.from_numpy(change_mask.transpose((2, 0, 1)).copy()).float()
        else:
            change_mask = torch.zeros((1, base_img.shape[1], base_img.shape[2])).float()

        change = torch.tensor(change).int()

        """ Return the sample """
        sample = {
            'base': base_img,
            't1': img_seq[0],
            't2': img_seq[1],
            't3': img_seq[2],
            't4': img_seq[3],
            't5': img_seq[4],
            'change': change,
            'mask': change_mask,
            # 2024/4/24: add event
            'event': event
        }

        return sample
    

    def zscore_norm(self, img, mean, std):
        """ Z-score normalization """
        # Input: numpy array [H, W, C]
        # mean: [C]
        # std: [C]
        # Output: numpy array [H, W, C]
        mean_expanded = np.expand_dims(mean, axis=(0, 1))
        std_expanded = np.expand_dims(std, axis=(0, 1))

        return (img - mean_expanded) / std_expanded