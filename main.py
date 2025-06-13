"""
`main.py` is the main script that trains the model.
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
from engine import train_one_epoch, evaluate

import numpy as np
import os
import time
import datetime
import random
import gc
import math

import wandb


def cosine_decay(epoch, num_epochs, warmup_epochs):
    """Dacay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr_lambda = float(epoch) / float(max(1, warmup_epochs))
    else:
        progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        lr_lambda = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(progress))))

    return lr_lambda


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) #gpu
    torch.cuda.manual_seed_all(seed) #gpu

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def deep_update(original, update):
    """
    Recursively update a dict with the contents in another dict.
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            deep_update(original[key], value)
        else:
            original[key] = value


def main(config):
    print(config)
    seed_everything(config['seed'])

    wandb.init(project=config['log']['wandb_proj'])

    wandb.config.update(config, allow_val_change=True)

    # 1. Load the dataset
    if config['datasets']['train_path'].split('/')[-2] == 'EuroSAT_hdf5':
        # Load dataset
        data_train = EuroSATHDF5Dataset(config['datasets']['train_path'], mode='train', config=config)
        data_val = EuroSATHDF5Dataset(config['datasets']['val_path'], mode='val', config=config)
        train_data_loader = DataLoader(data_train, batch_size=config['batch_size'], num_workers=0, shuffle=True, pin_memory=True, drop_last=True)
        # TODO: change drop_last to False (need to pad the last batch)
        val_data_loader = DataLoader(data_val, batch_size=config['val_batch_size'], num_workers=0, shuffle=False, pin_memory=True, drop_last=True)
    elif config['datasets']['train_path'].split('/')[-2] == 'ravaen_hdf5':
        # Load RaVAEn dataset
        data_train = RaVAEnHDF5Dataset(config['datasets']['train_path'], config['datasets']['train_csv'], mode='train', config=config)
        data_val = RaVAEnHDF5Dataset(config['datasets']['val_path'], config['datasets']['val_csv'], mode='val', config=config)
        train_data_loader = DataLoader(data_train, batch_size=config['batch_size'], num_workers=0, shuffle=True, pin_memory=True, drop_last=True)
        val_data_loader = DataLoader(data_val, batch_size=config['val_batch_size'], num_workers=0, shuffle=False, pin_memory=True, drop_last=True)


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

    model.cuda()

    # 3. Load the optimizer
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=float(config['lr']), weight_decay=float(config['weight_decay']))

    # 4. Load the learning rate scheduler
    warmup_epochs = config['warmup_epochs']
    num_epochs = config['num_epochs']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: cosine_decay(epoch, num_epochs, warmup_epochs))

    gc.collect()
    torch.cuda.empty_cache()

    model = nn.DataParallel(model).cuda()
    scaler = GradScaler()

    # Train
    print('Start training...')
    start_time = time.time()
    for epoch in range(num_epochs):
        train_one_epoch(model, criterion, train_data_loader, optimizer, scheduler, scaler, epoch, config)
        
        # Evaluate
        print('Evaluating...')
        evaluate(model, criterion, val_data_loader, epoch, config)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SITS-ExtremeEvents')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--subconfig', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Load config file
    config = load_config(args.config)

    if args.subconfig:
        deep_update(config, load_config(args.subconfig))

    # Update the format of mean and std in config file (from string to numpy array)
    mean_str = config['datasets']['norm']['mean'][0]
    std_str = config['datasets']['norm']['std'][0]
    config['datasets']['norm']['mean'] = np.array([float(x) for x in mean_str.split()])
    config['datasets']['norm']['std'] = np.array([float(x) for x in std_str.split()])

    # Update seed
    config['seed'] = args.seed

    main(config)