# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from typing import Optional

from direct.config.defaults import ModelConfig
from direct.nn.types import ActivationType, ModelName


@dataclass
class MRIVarSplitNetConfig(ModelConfig):
    num_steps_reg: int = 8
    num_steps_dc: int = 8
    image_init: str = "sense"
    no_parameter_sharing: bool = True
    image_model_architecture: str = ModelName.unet
    image_resnet_hidden_channels: int = 128
    image_resnet_num_blocks: int = 15
    image_resnet_batchnorm: bool = True
    image_resnet_scale: float = 0.1
    image_unet_num_filters: int = 32
    image_unet_num_pool_layers: int = 4
    image_unet_dropout: float = 0.0
    image_didn_hidden_channels: int = 16
    image_didn_num_dubs: int = 6
    image_didn_num_convs_recon: int = 9
    image_conv_hidden_channels: int = 64
    image_conv_n_convs: int = 15
    image_conv_activation: str = ActivationType.relu
    image_conv_batchnorm: bool = False
    image_uformer_patch_size: int = 256
    image_uformer_embedding_dim: int = 32
    image_uformer_encoder_depths: tuple[int, ...] = (2, 2, 2, 2)
    image_uformer_encoder_num_heads: tuple[int, ...] = (1, 2, 4, 8)
    image_uformer_bottleneck_depth: int = 2
    image_uformer_bottleneck_num_heads: int = 16
    image_uformer_win_size: int = 8
    image_uformer_mlp_ratio: float = 4.0
    image_uformer_qkv_bias: bool = True
    image_uformer_qk_scale: Optional[float] = None
    image_uformer_drop_rate: float = 0.0
    image_uformer_attn_drop_rate: float = 0.0
    image_uformer_drop_path_rate: float = 0.1
    image_uformer_patch_norm: bool = True
    image_uformer_shift_flag: bool = True
    image_uformer_modulator: bool = False
    image_uformer_cross_modulator: bool = False
    image_uformer_normalized: bool = True
