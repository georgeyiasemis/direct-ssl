# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Tuple

import torch.nn as nn

from direct.config.defaults import ModelConfig


@dataclass
class ImageDomainVisionTrasnformerConfig(ModelConfig):
    use_mask: bool = True
    average_img_size: int = 320
    patch_size: int = 10
    embedding_dim: int = 64
    depth: int = 8
    num_heads: int = 9
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: float = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    gpsa_interval: Tuple[int, int] = (-1, -1)
    locality_strength: float = 1.0
    use_pos_embedding: bool = True
