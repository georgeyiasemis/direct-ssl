# coding=utf-8
# Copyright (c) DIRECT Contributors

from math import ceil, floor
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from direct.data.transforms import apply_mask, expand_operator, reduce_operator
from direct.nn.transformers.vision_transformers import VisionTransformer

__all__ = ["MRITransformer", "ImageDomainVisionTransformer"]


def _pad(x: torch.Tensor, patch_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """Pad the input tensor with zeros to make its spatial dimensions divisible by the patch size.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).
    patch_size : Tuple[int, int]
        Patch size as a tuple of integers (patch_height, patch_width).

    Returns
    -------
    Tuple containing the padded tensor, and the number of pixels padded in the width and height dimensions respectively.
    """
    _, _, h, w = x.shape
    hp, wp = patch_size
    f1 = ((wp - w % wp) % wp) / 2
    f2 = ((hp - h % hp) % hp) / 2
    wpad = (floor(f1), ceil(f1))
    hpad = (floor(f2), ceil(f2))
    x = F.pad(x, wpad + hpad)

    return x, wpad, hpad


def _unpad(x: torch.Tensor, wpad: Tuple[int, int], hpad: Tuple[int, int]) -> torch.Tensor:
    """Remove the padding added to the input tensor by _pad method.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H_pad, W_pad).
    wpad : Tuple[int, int]
        Number of pixels padded in the width dimension as a tuple of integers (left_pad, right_pad).
    hpad : Tuple[int, int]
        Number of pixels padded in the height dimension as a tuple of integers (top_pad, bottom_pad).

    Returns
    -------
    Tensor with the same shape as the original input tensor, but without the added padding.
    """
    return x[..., hpad[0] : x.shape[-2] - hpad[1], wpad[0] : x.shape[-1] - wpad[1]]


def _norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize the input tensor by subtracting the mean and dividing by the standard deviation across each channel and pixel.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).

    Returns
    -------
    Tuple containing the normalized tensor, mean tensor and standard deviation tensor.
    """
    mean = x.reshape(x.shape[0], 1, 1, -1).mean(-1, keepdim=True)
    std = x.reshape(x.shape[0], 1, 1, -1).std(-1, keepdim=True)
    x = (x - mean) / std

    return x, mean, std


def _unnorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalize the input tensor by multiplying with the standard deviation and adding
    the mean across each channel and pixel.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).
    mean : torch.Tensor
        Mean tensor obtained during normalization.
    std : torch.Tensor
        Standard deviation tensor obtained during normalization.

    Returns
    -------
    Tensor with the same shape as the original input tensor, but denormalized.
    """
    return x * std + mean


class MRITransformer(nn.Module):
    """A PyTorch module that implements MRI image reconstruction using VisionTransformer."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_gradient_descent_steps: int,
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
        **kwargs,
    ):
        """Inits :class:`MRITransformer`.

        Parameters
        ----------
        forward_operator : Callable
            Forward operator function.
        backward_operator : Callable
            Backward operator function.
        num_gradient_descent_steps : int
            Number of gradient descent steps to perform.
        average_img_size : int or Tuple[int, int], optional
            Size to which the input image is rescaled before processing.
        patch_size : int or Tuple[int, int], optional
            Patch size used in VisionTransformer.
        embedding_dim : int, optional
            The number of embedding dimensions in the VisionTransformer.
        depth : int, optional
            The number of layers in the VisionTransformer.
        num_heads : int, optional
            The number of attention heads in the VisionTransformer.
        mlp_ratio : float, optional
            The ratio of MLP hidden size to embedding size in the VisionTransformer.
        qkv_bias : bool, optional
            Whether to include bias terms in the projection matrices in the VisionTransformer.
        qk_scale : float, optional
            Scale factor for query and key in the attention calculation in the VisionTransformer.
        drop_rate : float, optional
            Dropout probability for the VisionTransformer.
        attn_drop_rate : float, optional
            Dropout probability for the attention layer in the VisionTransformer.
        dropout_path_rate : float, optional
            Dropout probability for the intermediate skip connections in the VisionTransformer.
        norm_layer : nn.Module, optional
            Normalization layer used in the VisionTransformer.
        gpsa_interval : Tuple[int, int], optional
            Interval for performing Generalized Positional Self-Attention (GPSA) in the VisionTransformer.
        locality_strength : float, optional
            The strength of locality in the GPSA in the VisionTransformer.
        use_pos_embedding : bool, optional
            Whether to use positional embedding in the VisionTransformer.
        """
        super().__init__()
        self.transformers = nn.ModuleList(
            [
                VisionTransformer(
                    average_img_size=average_img_size,
                    patch_size=patch_size,
                    in_channels=2,
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
                for _ in range(num_gradient_descent_steps)
            ]
        )
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_gradient_descent_steps = num_gradient_descent_steps

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

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
        inp = reduce_operator(
            coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
            sensitivity_map=sensitivity_map,
            dim=self._coil_dim,
        )

        inp = inp.permute(0, 3, 1, 2)

        inp, wpad, hpad = _pad(inp, self.transformers[0].patch_size)
        inp, mean, std = _norm(inp)

        inp = inp.permute(0, 2, 3, 1)

        for _ in range(self.num_gradient_descent_steps):
            inp = self.forward_operator(
                expand_operator(inp, sensitivity_map, dim=self._coil_dim),
                dim=self._spatial_dims,
            )
            inp = self.backward_operator(
                apply_mask(inp, sampling_mask, return_mask=False),
                dim=self._spatial_dims,
            )
            inp = reduce_operator(inp, sensitivity_map, dim=self._coil_dim)
            inp += self.transformers[_](inp.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        inp = inp.permute(0, 3, 1, 2)
        out = _unnorm(inp, mean, std)
        out = _unpad(out, wpad, hpad)

        return out.permute(0, 2, 3, 1)


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
        **kwargs,
    ):
        super().__init__()
        self.tranformer = VisionTransformer(
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

    def pad(self, x):
        _, _, h, w = x.shape
        hp, wp = self.transformer.patch_size
        f1 = ((wp - w % wp) % wp) / 2
        f2 = ((hp - h % hp) % hp) / 2
        wpad = [floor(f1), ceil(f1)]
        hpad = [floor(f2), ceil(f2)]
        x = F.pad(x, wpad + hpad)

        return x, wpad, hpad

    def unpad(self, x, wpad, hpad):
        return x[..., hpad[0] : x.shape[-2] - hpad[1], wpad[0] : x.shape[-1] - wpad[1]]

    def norm(self, x):
        mean = x.reshape(x.shape[0], 1, 1, -1).mean(-1, keepdim=True)
        std = x.reshape(x.shape[0], 1, 1, -1).std(-1, keepdim=True)
        x = (x - mean) / std

        return x, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

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
        inp = reduce_operator(
            coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
            sensitivity_map=sensitivity_map,
            dim=self._coil_dim,
        )

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

        inp = inp.permute(0, 3, 1, 2)

        inp, wpad, hpad = _pad(inp, self.transformer.patch_size)
        inp, mean, std = _norm(inp)

        out = self.transformer(inp)

        out = _unnorm(out, mean, std)
        out = _unpad(out, wpad, hpad)

        return out.permute(0, 2, 3, 1)
