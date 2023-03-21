# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Callable, Tuple, Union

import torch
import torch.nn as nn

from direct.data.transforms import complex_multiplication, conjugate
from direct.nn.transformers.vision_transformers import VisionTransformer

__all__ = ["ImageDomainVisionTransformer"]


class ImageDomainVisionTransformer(nn.Module):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        use_mask: bool = True,
        average_img_size: Union[int, Tuple[int, int]] = 320,
        patch_size: Union[int, Tuple[int, int]] = 10,
        embedding_dim: int = 64,
        depth: int = 8,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        gpsa_interval: Tuple[int, int] = (-1, -1),
        locality_strength: float = 1.0,
        use_pos_embedding: bool = True,
    ):
        super().__init__()
        self.transformer = VisionTransformer(
            average_img_size=average_img_size,
            patch_size=patch_size,
            in_channels=4 if use_mask else 2,
            out_channels=2,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            norm_layer=norm_layer,
            gpsa_interval=gpsa_interval,
            locality_strength=locality_strength,
            use_pos_embedding=use_pos_embedding,
        )
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.use_mask = use_mask

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def compute_sense_init(self, kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        r"""Computes sense initialization :math:`x_{\text{SENSE}}`:
        .. math::
            x_{\text{SENSE}} = \sum_{k=1}^{n_c} {S^{k}}^* \times y^k
        where :math:`y^k` denotes the data from coil :math:`k`.
        Parameters
        ----------
        kspace: torch.Tensor
            k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        Returns
        -------
        input_image: torch.Tensor
            Sense initialization :math:`x_{\text{SENSE}}`.
        """
        input_image = complex_multiplication(
            conjugate(sensitivity_map),
            self.backward_operator(kspace, dim=self._spatial_dims),
        )
        input_image = input_image.sum(self._coil_dim)

        return input_image

    def forward(
        self, masked_kspace: torch.Tensor, sensitivity_map: torch.Tensor, sampling_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Performs forward pass of :class:`ImageDomainVisionTransformer`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).

        Returns
        -------
        out : torch.Tensor
            Prediction of output image of shape (N, height, width, complex=2).
        """
        inp = self.compute_sense_init(kspace=masked_kspace, sensitivity_map=sensitivity_map)
        if self.use_mask and sampling_mask is not None:
            sampling_mask_inp = torch.cat(
                [
                    sampling_mask,
                    torch.zeros(*sampling_mask.shape, device=sampling_mask.device),
                ],
                dim=self._complex_dim,
            ).to(inp.dtype)
            # project it in image domain
            sampling_mask_inp = self.backward_operator(sampling_mask_inp, dim=self._spatial_dims).squeeze(
                self._coil_dim
            )
            inp = torch.cat([inp, sampling_mask_inp], dim=self._complex_dim)

        out = self.transformer(inp.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)
