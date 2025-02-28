"""
`bi_siamconcat.py` contains the implementation of FC-SiamConcat model with bi-temporal pairing strategy.

Bi-temporal pairing strategy (with multi-temporal data):
- The model takes two of multi-temporal images as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import random

from .vision_transformer import VisionTransformer as ViT

from .helper import build_encoder


class BiSiamConcat(nn.Module):
    """
    Bi-temporal FC-Siam-Concat model.
    """
    def __init__(self, encoder, dim):
        """
        Args:
        - encoder (nn.Module): VisionTransformer

        Note: Why we only need one encoder?
        - The encoder is shared between the two inputs.
        - The two inputs are then passed through the encoder to get the feature maps.
        """
        super().__init__()
        self.encoder = encoder

        self.non_linear = nn.ReLU()

        self.classifier = nn.Linear(2*dim, 1)

    
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
        concat_dict = {}

        for key in ['t1', 't2', 't3', 't4', 't5']:
            x = sample[key]

            # Encoder
            embedding, position_encodings = self.encoder(x)
            embedding_dict[key] = embedding

        # Concatenate the feature embeddings
        for key in ['t1', 't2', 't3', 't4']:
            concat = torch.cat((torch.flatten(embedding_dict['t5'], start_dim=1), torch.flatten(embedding_dict[key], start_dim=1)), dim=1)

            concat = self.non_linear(concat)

            concat = self.classifier(concat)

            concat_dict[key] = concat

        return embedding_dict, concat_dict
    

class SetCriterion(nn.Module):
    def __init__(self, losses):
        """
        Args:
        - losses (dict): a dictionary of the loss functions
        """
        super().__init__()
        self.losses = losses
        self.loss_fn = nn.BCEWithLogitsLoss()

    
    def forward(self, concat_dict, sample):
        """
        Args:
        - concat_dict (dict): a dictionary of the concatenated feature embeddings
        """
        loss = 0

        for key in ['t1', 't2', 't3', 't4']:
            loss += self.loss_fn(concat_dict[key], sample['change'].unsqueeze(1).float())

        return loss / 4
    

def build_model(config):
    encoder = build_encoder(config)

    encoder_dim = config['encoder']['embed_dim']*(config['datasets']['img_size'][0]//config['encoder']['patch_size'])*(config['datasets']['img_size'][1]//config['encoder']['patch_size'])

    model = BiSiamConcat(encoder, dim=encoder_dim)

    criterion = SetCriterion(config['losses']['types'])

    return model, criterion