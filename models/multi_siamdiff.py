"""
`multi_siamdiff.py' contains the implementation of FC-Siam-Diff model with multi-temporal Siamese difference strategy.

Multi-temporal Siamese difference strategy:
- The model takes multi-temporal images as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import random

from .vision_transformer import VisionTransformer as ViT

from .helper import build_encoder


class MultiSiamDiff(nn.Module):
    def __init__(self, encoder, dim):
        super().__init__()

        self.encoder = encoder

        self.non_linear = nn.ReLU()

        self.classifier = nn.Linear(dim, 1)


    def forward(self, sample):
        """
        Args:
            sample (dict): a dictionary of the input sample. 
            See datasets/EuroSAT_hdf5_dataset.py for more details.
            sample = {
                'base': base_img,
                't1': img_seq[0],
                't2': img_seq[1],
                't3': img_seq[2],
                't4': img_seq[3],
                't5': img_seq[4],
                'change': change,
                'mask': change_mask
            }
        """
        embedding_dict = {}
        diff_dict = {}

        for key in ['t1', 't2', 't3', 't4', 't5']:
            x = sample[key]

            # Encoder
            embedding, _ = self.encoder(x)
            embedding_dict[key] = embedding

        # Compute the difference between the base image and the other images
        for key in ['t1', 't2', 't3', 't4']:
            diff = torch.abs(torch.flatten(embedding_dict['t5'], start_dim=1) - torch.flatten(embedding_dict[key], start_dim=1))
            diff_dict[key] = diff

        # Aggregate the differences (mean)
        aggregated_diff = torch.stack([diff_dict[key] for key in ['t1', 't2', 't3', 't4']], dim=1).mean(dim=1)

        # Non-linear layer + classifier
        aggregated_diff = self.classifier(self.non_linear(aggregated_diff))

        return embedding_dict, aggregated_diff
    

class SetCriterion(nn.Module):
    def __init__(self, losses):
        """
        Args:
        - losses (list): a list of loss functions
        """
        super().__init__()
        self.losses = losses
        self.loss_fn = nn.BCEWithLogitsLoss()

    
    def forward(self, diff, sample):
        """
        Args:
        - diff (tensor): the aggregated difference tensor
        """
        loss = self.loss_fn(diff, sample['change'].unsqueeze(1).float())

        return loss
    

def build_model(config):
    encoder = build_encoder(config)

    encoder_dim = config['encoder']['embed_dim']*(config['datasets']['img_size'][0]//config['encoder']['patch_size'])*(config['datasets']['img_size'][1]//config['encoder']['patch_size'])
    
    model = MultiSiamDiff(encoder, dim=encoder_dim)

    criterion = SetCriterion(config['losses']['types'])

    return model, criterion