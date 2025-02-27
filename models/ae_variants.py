"""
`ae_variants.py` contains the implementation of the following autoencoder variants:
1. Vanilla Autoencoder: only reconstruction loss
2. Contrastive Autoencoder: reconstruction loss + contrastive loss
"""

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import random

from itertools import combinations

from .vision_transformer import VisionTransformer as ViT
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from .helper import UpsampleLayer, build_encoder, build_decoder


class VanillaAE(nn.Module):
    def __init__(self, encoder, decoder, upsampling_layer):
        """"
        Args:
        - encoder (nn.Module): VisionTransformer
        - decoder (nn.Module): TransformerDecoder
        - upsampling_layer (nn.Module): UpsampleLayer
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.upsampling_layer = upsampling_layer

    
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
        recon_output = {}
        embedding_dict = {}

        for key in ['t1', 't2', 't3', 't4', 't5']:
            x = sample[key]

            # Encoder
            embedding, position_encodings = self.encoder(x)

            # Decoder
            output = self.decoder(tgt=embedding,
                                  memory=embedding,
                                  tgt_mask=None,
                                  memory_mask=None,
                                  tgt_key_padding_mask=None,
                                  memory_key_padding_mask=None,
                                  pos=position_encodings,
                                  query_pos=position_encodings)[0]
            
            # Upsampling layer
            output = self.upsampling_layer(output)

            recon_output[key] = output
            embedding_dict[key] = embedding

        return recon_output, embedding_dict
    

class SetCriterion(nn.Module):
    """
    This class sets the criterion for the model.
    """
    def __init__(self, losses):
        """
        Initialize the SetCriterion module.

        Args:
            losses (list): a list of losses to be computed.
        """
        super().__init__()
        self.losses = losses

    
    def loss_recon_output(self, output, sample):
        """
        Compute the reconstruction loss.
        Objective: reconstruction loss of (mean image + relevant change + irrelevant change)

        Args:
            output (dict): a dictionary of the reconstruction output.
            output = {
                't1': output_t1,
                't2': output_t2,
                't3': output_t3,
                't4': output_t4,
                't5': output_t5
            }
            sample (dict): a dictionary of the input sample.
        """
        loss = 0
        for key in ['t1', 't2', 't3', 't4', 't5']:
            # output[key] -> (B, C, H, W)
            # sample[key] -> (B, C, H, W)
            loss += nn.MSELoss(reduction='mean')(output[key], sample[key])
        
        loss /= 5

        return loss
    

    @staticmethod
    def cosine_distance(x1, x2):
        """
        TODO: should we add a mlp layer to project the embedding to a lower dimension?
        for example, (B, H/P * W/P, D) -> (B, hidden_dim)?
        """
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)
        return 1 - torch.abs(F.cosine_similarity(x1, x2, dim=-1))
    

    def loss_relevant_embedding_contrastive(self, sample, embedding, temperature):
        """
        Compute the contrastive loss for relevant change embedding.
        
        negative pairs: t5/t1, t5/t2, t5/t3, t5/t4
        positive pairs: t1/t2, t1/t3, t1/t4, t2/t3, t2/t4, t3/t4

        consider positive and negative pairs according to each negative pair
        for example: t5/t1, consider t1/t2, t1/t3, t1/t4 as positive pairs
        we are more interested in the negative pairs, so we use cosine_distance instead of cosine_similarity
        """

        """
        why we need sample in the input?
        we need to figure out which sample is changed and which sample is unchanged
        """
        # 1. get the changed sample and unchanged sample
        indices_changed = []
        indices_unchanged = []

        for idx in range(len(sample)):
            if sample['change'][idx] == 1:
                indices_changed.append(idx)
            else:
                indices_unchanged.append(idx)

        # 2. get the changed embedding and unchanged embedding
        embedding_changed = {}
        embedding_unchanged = {}

        for key in ['t1', 't2', 't3', 't4', 't5']:
            embedding_changed[key] = embedding[key][indices_changed]
            embedding_unchanged[key] = embedding[key][indices_unchanged]

        # 3. compute the contrastive loss for changed embedding
        contrastive_loss = torch.tensor(0.0).cuda()

        if len(indices_changed) != 0:
            for key in ['t1', 't2', 't3', 't4']:
                neg = torch.exp(self.cosine_distance(embedding_changed['t5'], embedding_changed[key]) / temperature)
                # get a new list without the current key
                key_except = [k for k in ['t1', 't2', 't3', 't4'] if k != key]

                pos1 = torch.exp(self.cosine_distance(embedding_changed[key], embedding_changed[key_except[0]]) / temperature)
                pos2 = torch.exp(self.cosine_distance(embedding_changed[key], embedding_changed[key_except[1]]) / temperature)
                pos3 = torch.exp(self.cosine_distance(embedding_changed[key], embedding_changed[key_except[2]]) / temperature)

                # sum up the positive pairs
                pos = pos1 + pos2 + pos3

                contrastive_loss += torch.log(pos / neg).mean()

            contrastive_loss /= 4
            contrastive_loss /= len(indices_changed)

        # 4. compute the consistency loss for unchanged embedding
        consistency_loss = torch.tensor(0.0).cuda()

        if len(indices_unchanged) != 0:
            # Get all possible pairs of embedding_unchanged
            pairs = list(combinations(embedding_unchanged.keys(), 2))

            losses = []
            for pair in pairs:
                # Cosine distance loss (cd in embedding space)
                cd = self.cosine_distance(embedding_unchanged[pair[0]], embedding_unchanged[pair[1]])
                losses.append(torch.mean(cd))
            
            consistency_loss = sum(losses) / len(losses)
            consistency_loss /= len(indices_unchanged)

        return contrastive_loss, consistency_loss
    

    def forward(self, recon_output, embedding, sample, temperature):
        """
        Forward pass of the SetCriterion module.
        """
        loss = {}

        if 'loss_recon_output' in self.losses:
            loss['loss_recon_output'] = self.loss_recon_output(recon_output, sample)
        if 'loss_embedding' in self.losses:
            contrastive_loss, consistency_loss = self.loss_relevant_embedding_contrastive(sample, embedding, temperature)
            loss['loss_embedding_contrastive'] = contrastive_loss
            loss['loss_embedding_consistency'] = consistency_loss

        return loss
    

def build_model(config):
    """
    Build the ae_variants model.

    Note: We want to test two variants of autoencoder:
    1. only reconstruction loss
    2. reconstruction loss + contrastive loss
    """
    encoder = build_encoder(config)
    decoder = build_decoder(config)

    upsampling_layer = UpsampleLayer(embed_dim=config['encoder']['embed_dim'],
                                     in_chans=config['datasets']['in_chans'],
                                     image_size=config['datasets']['img_size'],
                                     patch_size=config['encoder']['patch_size'],
                                     num_layers=config['upsampling_layer']['num_layers'])
    
    model = VanillaAE(encoder, decoder, upsampling_layer)

    criterion = SetCriterion(config['losses']['types'])

    return model, criterion