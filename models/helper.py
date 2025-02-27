"""
Helper functions for the models.

Contains the following functions/classes:
1. UpsampleLayer √
2. build_encoder √
3. build_decoder √
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .vision_transformer import VisionTransformer as ViT
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer


class MLP(nn.Module):
    """
    Multi-layer perceptron. (general MLP, can specify the number of layers and hidden dimension)

    Modified from https://github.com/facebookresearch/detr/blob/main/models/detr.py#L289
    [linear + layer_norm + activation] x L + linear
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Args:
            input_dim (int): input dimension
            hidden_dim (int): hidden dimension
            output_dim (int): output dimension
            num_layers (int): number of layers
        """
        super().__init__()
        self.num_layers = num_layers
        # for example, if num_layers = 2, then hidden_dims = [hidden_dim]
        h = [hidden_dim] * (num_layers - 1)

        # Create alternating list of linear and layernorm layers
        layers = []
        # for example, if num_layers = 2, then
        # enumerate(zip([input_dim] + h, h + [output_dim])) -> [(0, (input_dim, hidden_dim)), (1, (hidden_dim, output_dim))]
        # last layer does not have layernorm
        for i, (n, k) in enumerate(zip([input_dim] + h, h + [output_dim])):
            layers.append(nn.Linear(n, k))
            if i < self.num_layers - 1:
                layers.append(nn.LayerNorm(k))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Apply ReLU activation to all but last layer
        # for example, if num_layers = 2, then self.layers = [linear, layernorm, linear]
        # i -> range(0, 3 - 1, 2) -> [0]
        # x -> linear + layernorm + relu + linear
        for i in range(0, len(self.layers) - 1, 2):
            x = self.layers[i](x)
            x = self.layers[i + 1](x)
            x = F.relu(x)
        # Apply final linear layer
        x = self.layers[-1](x)
        return x
        

class UpsampleLayer(nn.Module):
    """
    Upsample layer after the transformer decoder.

    Input: (B, H/P * W/P, D)
    Output: (B, C, H, W)

    Dimension: (B, H/P * W/P, D) -> (B, H/P, W/P, D) -> (B, H/P, W/P, CxPxP)
    -> (B, H/PxP, W/PxP, C) -> (B, C, H, W)
    """
    def __init__(self, embed_dim, in_chans, image_size, patch_size, num_layers):
        """
        Args:
            embed_dim: D
            in_chans: C
            image_size: H, W
            patch_size: P
            num_layers: L
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers

        # define the upsampling layer
        upsampling_dim = in_chans * patch_size * patch_size
        self.mlp = MLP(embed_dim, upsampling_dim, upsampling_dim, num_layers)
        
    
    def forward(self, x):
        B, _, D = x.shape
        H, W = self.image_size
        P = self.patch_size
        C = self.in_chans

        # (B, H/P * W/P, D) -> (B, H/P, W/P, D)
        x = x.reshape(B, H // P, W // P, D)
        # (B, H/P, W/P, D) -> (B, H/P, W/P, CxPxP)
        x = self.mlp(x)
        # (B, H/P, W/P, CxPxP) -> (B, H/P, W/P, C, P, P) -> (B, C, H/P, P, W/P, P)
        x = x.reshape(B, H // P, W // P, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        # (B, C, H/P, P, W/P, P) -> (B, C, H, W)
        x = x.reshape(B, C, H, W)

        return x


def build_encoder(config, **kwargs):
    return ViT(
        img_size=config['datasets']['img_size'],
        patch_size=config['encoder']['patch_size'],
        in_chans=config['datasets']['in_chans'],
        embed_dim=config['encoder']['embed_dim'],
        depth=config['encoder']['depth'],
        num_heads=config['encoder']['num_heads'],
        mlp_ratio=config['encoder']['mlp_ratio'],
        qkv_bias=config['encoder']['qkv_bias'],
        **kwargs)


def build_decoder(config):
    decoder_layer = TransformerDecoderLayer(
        d_model=config['decoder']['d_model'],
        nhead=config['decoder']['nhead'],
        dim_feedforward=config['decoder']['dim_feedforward'],
        dropout=config['decoder']['dropout'],
        activation=config['decoder']['activation'],
        normalize_before=config['decoder']['normalize_before'],
        use_self_attn=config['decoder']['use_self_attn']
    )
    decoder_norm = nn.LayerNorm(config['decoder']['d_model'])

    return TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=config['decoder']['num_layers'],
        norm=decoder_norm,
        return_intermediate=config['decoder']['return_intermediate'],
        init=True
    )