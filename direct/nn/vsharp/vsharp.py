# coding=utf-8
# Copyright (c) DIRECT Contributors


from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from direct.constants import COMPLEX_SIZE
from direct.data.transforms import apply_mask, expand_operator, reduce_operator
from direct.nn.get_nn_model_config import ModelName, _get_activation, _get_model_config
from direct.nn.types import ActivationType, InitType


class LagrangeMultipliersInitializer(nn.Module):
    """A convolutional neural network model that initializers the Lagrange multiplier of the vSHARPNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        dilations: tuple[int, ...],
        multiscale_depth: int = 1,
        activation: ActivationType = ActivationType.prelu,
    ):
        """Inits :class:`LagrangeMultipliersInitializer`.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        channels : tuple of ints
            Tuple of integers specifying the number of output channels for each convolutional layer in the network.
        dilations : tuple of ints
            Tuple of integers specifying the dilation factor for each convolutional layer in the network.
        multiscale_depth : int
            Number of multiscale features to include in the output. Default: 1.
        """
        super().__init__()

        # Define convolutional blocks
        self.conv_blocks = nn.ModuleList()
        tch = in_channels
        for curr_channels, curr_dilations in zip(channels, dilations):
            block = nn.Sequential(
                nn.ReplicationPad2d(curr_dilations),
                nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations),
            )
            tch = curr_channels
            self.conv_blocks.append(block)

        # Define output block
        tch = np.sum(channels[-multiscale_depth:])
        block = nn.Conv2d(tch, out_channels, 1, padding=0)
        self.out_block = nn.Sequential(block)

        self.multiscale_depth = multiscale_depth

        self.activation = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`LagrangeMultipliersInitializer`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width).
        """

        features = []
        for block in self.conv_blocks:
            x = F.relu(block(x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)

        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth :], dim=1)

        return self.activation(self.out_block(x), inplace=True)


class VSharpNet(nn.Module):
    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        num_steps: int,
        num_steps_dc_gd: int,
        image_init: str = InitType.sense,
        no_parameter_sharing: bool = True,
        image_model_architecture: ModelName = ModelName.unet,
        initializer_channels: tuple[int, ...] = (32, 32, 64, 64),
        initializer_dilations: tuple[int, ...] = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        initializer_activation: ActivationType = ActivationType.prelu,
        **kwargs,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.num_steps_dc_gd = num_steps_dc_gd

        self.no_parameter_sharing = no_parameter_sharing

        if image_model_architecture not in ["unet", "normunet", "resnet", "didn", "conv", "uformer"]:
            raise ValueError(f"Invalid value {image_model_architecture} for `image_model_architecture`.")

        image_model, image_model_kwargs = _get_model_config(
            image_model_architecture,
            in_channels=COMPLEX_SIZE * 3,
            out_channels=COMPLEX_SIZE,
            **{k.replace("image_", ""): v for (k, v) in kwargs.items() if "image_" in k},
        )

        self.denoiser_blocks = nn.ModuleList()
        for _ in range(num_steps if self.no_parameter_sharing else 1):
            self.denoiser_blocks.append(image_model(**image_model_kwargs))

        self.initializer = LagrangeMultipliersInitializer(
            COMPLEX_SIZE,
            COMPLEX_SIZE,
            channels=initializer_channels,
            dilations=initializer_dilations,
            multiscale_depth=initializer_multiscale,
            activation=initializer_activation,
        )

        self.lmbda = nn.Parameter(torch.ones(1, requires_grad=True))
        nn.init.trunc_normal_(self.lmbda, 0.0, 1.0, 0.0)

        self.learning_rate_eta = nn.Parameter(torch.ones(num_steps_dc_gd, requires_grad=True))
        nn.init.trunc_normal_(self.learning_rate_eta, 0.0, 1.0, 0.0)

        self.rho = nn.Parameter(torch.ones(1, requires_grad=True))
        nn.init.trunc_normal_(self.rho, 0, 0.1, 0.0)

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        if image_init not in ["sense", "zero_filled"]:
            raise ValueError(f"Unknown image_initialization. Expected 'sense' or 'zero_filled'. " f"Got {image_init}.")

        self.image_init = image_init

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`MRIVarSplitNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2). Default: None.
        sampling_mask: torch.Tensor

        Returns
        -------
        image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        if self.image_init == "sense":
            x = reduce_operator(
                coil_data=self.backward_operator(masked_kspace, dim=self._spatial_dims),
                sensitivity_map=sensitivity_map,
                dim=self._coil_dim,
            )
        else:
            x = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)

        z = x.clone()

        u = self.initializer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        for iz in range(self.num_steps):
            z = (self.lmbda / self.rho) * self.denoiser_blocks[iz if self.no_parameter_sharing else 0](
                torch.cat([z, x, u / self.rho], dim=self._complex_dim).permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)

            for ix in range(self.num_steps_dc_gd):
                dc = apply_mask(
                    self.forward_operator(expand_operator(x, sensitivity_map, self._coil_dim), dim=self._spatial_dims)
                    - masked_kspace,
                    sampling_mask,
                    return_mask=False,
                )
                dc = self.backward_operator(dc, dim=self._spatial_dims)
                dc = reduce_operator(dc, sensitivity_map, self._coil_dim)

                x = x - self.learning_rate_eta[ix] * (dc + self.rho * (x - z) + u)

            u = u + self.rho * (x - z)

        return x
