# coding=utf-8
# Copyright (c) DIRECT Contributors


from __future__ import annotations

from typing import Callable, Optional

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

        return self.activation(self.out_block(x))


class VSharpNet(nn.Module):
    """

    It solves the augmented Lagrangian derivation of the variable half quadratic splitting problem using ADMM:

    .. math ::
        \vec{z}^{t+1}  = \argmin_{\vec{z}}\, \lambda \, \mathcal{G}(\vec{z}) +
            \frac{\rho}{2} \big | \big | \vec{x}^{t} - \vec{z} + \frac{\vec{u}^t}{\rho} \big | \big |_2^2
             \quad \Big[\vec{z}\text{-step}\Big]
        \vec{x}^{t+1}  = \argmin_{\vec{x}}\, \frac{1}{2} \big | \big | \mathcal{A}_{\mat{U},\mat{S}}(\vec{x}) -
            \tilde{\vec{y}} \big | \big |_2^2 + \frac{\rho}{2} \big | \big | \vec{x} - \vec{z}^{t+1}
            + \frac{\vec{u}^t}{\rho} \big | \big |_2^2 \quad \Big[\vec{x}\text{-step}\Big]
        \vec{u}^{t+1} = \vec{u}^t + \rho (\vec{x}^{t+1} - \vec{z}^{t+1}) \quad \Big[\vec{u}\text{-step}\Big]

    """

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
        kspace_no_parameter_sharing: bool = True,
        kspace_model_architecture: Optional[ModelName] = None,
        auxiliary_steps: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.num_steps_dc_gd = num_steps_dc_gd

        self.no_parameter_sharing = no_parameter_sharing

        if image_model_architecture not in [
            "unet",
            "normunet",
            "resnet",
            "didn",
            "conv",
            "uformer",
            "vision_transformer",
        ]:
            raise ValueError(f"Invalid value {image_model_architecture} for `image_model_architecture`.")
        if kspace_model_architecture not in [
            "unet",
            "normunet",
            "resnet",
            "didn",
            "conv",
            "uformer",
            "vision_transformer",
            None,
        ]:
            raise ValueError(f"Invalid value {kspace_model_architecture} for `kspace_model_architecture`.")

        image_model, image_model_kwargs = _get_model_config(
            image_model_architecture,
            in_channels=COMPLEX_SIZE * 4 if kspace_model_architecture else COMPLEX_SIZE * 3,
            out_channels=COMPLEX_SIZE,
            **{k.replace("image_", ""): v for (k, v) in kwargs.items() if "image_" in k},
        )

        if kspace_model_architecture:
            self.kspace_no_parameter_sharing = kspace_no_parameter_sharing
            kspace_model, kspace_model_kwargs = _get_model_config(
                kspace_model_architecture,
                in_channels=COMPLEX_SIZE,
                out_channels=COMPLEX_SIZE,
                **{k.replace("kspace_", ""): v for (k, v) in kwargs.items() if "kspace_" in k},
            )
            self.kspace_denoiser = kspace_model(**kspace_model_kwargs)
            self.scale_k = nn.Parameter(torch.ones(1, requires_grad=True))
            nn.init.trunc_normal_(self.scale_k, 0, 0.1, 0.0)
        else:
            self.kspace_denoiser = None

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

        self.learning_rate_eta = nn.Parameter(torch.ones(num_steps_dc_gd, requires_grad=True))
        nn.init.trunc_normal_(self.learning_rate_eta, 0.0, 1.0, 0.0)

        self.rho = nn.Parameter(torch.ones(num_steps, requires_grad=True))
        nn.init.trunc_normal_(self.rho, 0, 0.1, 0.0)

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        if image_init not in ["sense", "zero_filled"]:
            raise ValueError(f"Unknown image_initialization. Expected 'sense' or 'zero_filled'. " f"Got {image_init}.")

        self.image_init = image_init

        if auxiliary_steps == -1:
            self.auxiliary_steps = list(range(num_steps - 1))
        else:
            self.auxiliary_steps = list(range(min(auxiliary_steps, num_steps - 1)))

        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = (2, 3)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
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
        out = []
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
            if self.kspace_denoiser:
                kspace_z = self.kspace_denoiser(
                    self.forward_operator(z.contiguous(), dim=[_ - 1 for _ in self._spatial_dims]).permute(0, 3, 1, 2)
                ).permute(0, 2, 3, 1)
                kspace_z = self.backward_operator(kspace_z.contiguous(), dim=[_ - 1 for _ in self._spatial_dims])

            z = self.denoiser_blocks[iz if self.no_parameter_sharing else 0](
                torch.cat(
                    [z, x, u / self.rho[iz]] + ([self.scale_k * kspace_z] if self.kspace_denoiser else []),
                    dim=self._complex_dim,
                ).permute(0, 3, 1, 2)
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

                x = x - self.learning_rate_eta[ix] * (dc + self.rho[iz] * (x - z) + u)

            if self.training:
                if iz in self.auxiliary_steps:
                    out.append(x)

            u = u + self.rho[iz] * (x - z)

            out.append(x)
        return out
