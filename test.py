"""
`test.py` is the script that tests the model on the test set.
"""

import yaml
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler
from functools import partial
from torch.utils.data import DataLoader


# import datasets class
from datasets.EuroSAT_hdf5_dataset import EuroSATHDF5Dataset
from datasets.ravaen_hdf5_dataset import RaVAEnHDF5Dataset

# import models class

# 1. ae_variants
from models.ae_variants import build_model as build_ae

# 2. vae_variants
from models.vae_variants import build_model as build_vae

# 3. siamese_variants (bi-temporal strategy)
from models.bi_siamconcat import build_model as build_bi_siamconcat
from models.bi_siamdiff import build_model as build_bi_siamdiff

# 4. siamese_variants (multi-temporal strategy)
from models.multi_siamconcat import build_model as build_multi_siamconcat
from models.multi_siamdiff import build_model as build_multi_siamdiff

# import engine class
from engine import test

from main import seed_everything, load_config, deep_update

import numpy as np
import os
import time
import datetime
import random
import gc
import math


def main(default_config, subconfig, model_path):
    # Load the config file
    config = load_config(default_config)
    deep_update(config, load_config(subconfig))

    # Update the format of mean and std in config file (from string to numpy array)
    mean_str = config['datasets']['norm']['mean'][0]
    std_str = config['datasets']['norm']['std'][0]
    config['datasets']['norm']['mean'] = np.array([float(x) for x in mean_str.split()])
    config['datasets']['norm']['std'] = np.array([float(x) for x in std_str.split()])

    print(config)

    # Set seed
    seed_everything(config['seed'])

    # 1. Load the test set
    if config['datasets']['test_path'].split('/')[-2] == 'EuroSAT_hdf5':
        data_test = EuroSATHDF5Dataset(config['datasets']['test_path'], mode='test', config=config)
        test_data_loader = DataLoader(data_test, batch_size=1, num_workers=0, shuffle=False)
    elif config['datasets']['test_path'].split('/')[-2] == 'ravaen_hdf5':
        data_test = RaVAEnHDF5Dataset(config['datasets']['test_path'], config['datasets']['test_csv'], mode='test', config=config)
        test_data_loader = DataLoader(data_test, batch_size=1, num_workers=0, shuffle=False)


    # 2. Load the model
    # 2.1 Load the ae_variants model
    if config['baseline'] == 'vanilla_ae':
        if config['timesteps'] == 5:
            model, criterion = build_ae(config)

    # 2.2 Load the vae_variants model
    elif config['baseline'] == 'vanilla_vae':
        if config['timesteps'] == 5:
            model, criterion = build_vae(config)

    # 2.3 Load the siamese_variants model (bi-temporal strategy)
    elif config['baseline'] == 'bi_siamconcat':
        model, criterion = build_bi_siamconcat(config)
    elif config['baseline'] == 'bi_siamdiff':
        model, criterion = build_bi_siamdiff(config)
        
    # 2.4 Load the siamese_variants model (multi-temporal strategy)
    elif config['baseline'] == 'multi_siamconcat':
        model, criterion = build_multi_siamconcat(config)
    elif config['baseline'] == 'multi_siamdiff':
        model, criterion = build_multi_siamdiff(config)

    model = nn.DataParallel(model).cuda()

    # Load the model
    checkpoint_path = os.path.join(model_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test the model
    dict = test(model, criterion, test_data_loader, config)

    return dict