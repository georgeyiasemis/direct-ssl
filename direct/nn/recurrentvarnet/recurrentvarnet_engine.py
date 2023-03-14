# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.nn.mri_models import MRIModelEngine
from direct.nn.ssl.mri_models import *


class RecurrentVarNetEngine(MRIModelEngine):
    """Recurrent Variational Network Engine."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`RecurrentVarNetEngine."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_kspace = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )
        output_kspace = T.apply_padding(output_kspace, data.get("padding", None))

        output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims),
            dim=self._coil_dim,
        )  # shape (batch, height,  width)

        return output_image, output_kspace


class RecurrentVarNetSSDUEngine(SSDUMRIModelEngine):
    """Recurrent Variational Network SSDU Engine."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`RecurrentVarNetSSDUEngine`."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[None, torch.Tensor]:
        kspace = data["input_kspace"] if self.model.training else data["masked_kspace"]
        mask = data["input_sampling_mask"] if self.model.training else data["sampling_mask"]
        output_kspace = self.model(
            masked_kspace=kspace,
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=mask,
        )

        output_kspace = kspace + T.apply_mask(output_kspace, ~mask, return_mask=False)
        output_kspace = T.apply_padding(output_kspace, data.get("padding", None))

        return None, output_kspace


class RecurrentVarNetDualSSLEngine(DualSSLMRIModelEngine):
    """Recurrent Variational Network DualSSL Engine."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`RecurrentVarNetDualSSLEngine`."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(
        self,
        data: Dict[str, Any],
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_kspace = self.model(
            masked_kspace=masked_kspace,
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=sampling_mask,
        )
        output_kspace = T.apply_padding(output_kspace, data.get("padding", None))
        output_image = output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims),
            dim=self._coil_dim,
        )
        return output_image, output_kspace


class RecurrentVarNetDualSSL2Engine(DualSSL2MRIModelEngine):
    """Recurrent Variational Network DualSSL2 Engine."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`RecurrentVarNetDualSSL2Engine`."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(
        self,
        data: Dict[str, Any],
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_kspace = self.model(
            masked_kspace=masked_kspace,
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=sampling_mask,
        )
        output_kspace = T.apply_padding(output_kspace, data.get("padding", None))
        output_image = output_image = T.root_sum_of_squares(
            self.backward_operator(output_kspace, dim=self._spatial_dims),
            dim=self._coil_dim,
        )
        return output_image, output_kspace


class RecurrentVarNetN2NEngine(N2NMRIModelEngine):
    """Recurrent Variational Network N2N Engine."""

    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: str,
        forward_operator: Optional[Callable] = None,
        backward_operator: Optional[Callable] = None,
        mixed_precision: bool = False,
        **models: nn.Module,
    ):
        """Inits :class:`RecurrentVarNetN2NEngine`."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[None, torch.Tensor]:
        output_kspace = self.model(
            masked_kspace=data["noisier_kspace"] if self.model.training else data["masked_kspace"],
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=data["noisier_sampling_mask"] if self.model.training else data["sampling_mask"],
        )
        output_kspace = T.apply_padding(output_kspace, data.get("padding", None))
        return None, output_kspace
