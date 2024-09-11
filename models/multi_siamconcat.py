"""
`multi_siamconcat.py` contains the implementation of FC-SiamConcat model with multi-temporal pairing strategy.

Multi-temporal pairing strategy:
- The model takes all multi-temporal images as input.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import random

from .vision_transformer import VisionTransformer as ViT

from .helper import build_encoder


class MultiSiamConcat(nn.Module):
    def __init__(self, encoder, dim):
        super().__init__()
        self.encoder = encoder

        self.non_linear = nn.ReLU()

        self.classifier = nn.Linear(5*dim, 1)


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
        
        for key in ['t1', 't2', 't3', 't4', 't5']:
            x = sample[key]

            # Encoder
            embedding, _ = self.encoder(x)
            embedding_dict[key] = embedding

            # Concatenate the embeddings
            if key == 't1':
                concat = torch.flatten(embedding, start_dim=1)
            else:
                concat = torch.cat((concat, torch.flatten(embedding, start_dim=1)), dim=1)

        # Non-linear layer + classifier
        concat = self.non_linear(concat)
        concat = self.classifier(concat)

        return embedding_dict, concat
    

class SetCriterion(nn.Module):
    def __init__(self, losses):
        """
        Args:
        - losses (dict): a dictionary of the loss functions
        """
        super().__init__()
        self.losses = losses
        self.loss_fn = nn.BCEWithLogitsLoss()

    
    def forward(self, concat, sample):
        """
        Args:
        - concat: the output of the classification model
        """
        loss = self.loss_fn(concat, sample['change'].unsqueeze(1).float())

        return loss
    

def build_model(config):
    encoder = build_encoder(config)

    encoder_dim = config['encoder']['embed_dim']*(config['datasets']['img_size'][0]//config['encoder']['patch_size'])*(config['datasets']['img_size'][1]//config['encoder']['patch_size'])

    model = MultiSiamConcat(encoder, dim=encoder_dim)

    criterion = SetCriterion(config['losses']['types'])

    return model, criterion