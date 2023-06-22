# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.nn.mri_models import MRIModelEngine
from direct.nn.ssl.mri_models import *


class LPDNetEngine(MRIModelEngine):
    """LPDNet Engine."""

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
        """Inits :class:`LPDNetEngine."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, None]:
        data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

        output_image = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )  # shape (batch, height,  width)
        return output_image, None


class LPDNetSSDUEngine(SSDUMRIModelEngine):
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
        output_image = self.model(
            masked_kspace=kspace,
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=mask,
        )
        output_kspace = T.apply_padding(
            kspace + self._forward_operator(output_image, data["sensitivity_map"], ~mask),
            padding=data["padding"],
        )
        return None, output_kspace


class LPDNetDualSSLEngine(DualSSLMRIModelEngine):
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
        """Inits :class:`LPDNetDualSSLEngine`."""
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
    ) -> Tuple[torch.Tensor, None]:
        output_image = self.model(
            masked_kspace=masked_kspace,
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=sampling_mask,
        )
        return output_image, None


class LPDNetDualSSL2Engine(DualSSL2MRIModelEngine):
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
    ) -> Tuple[torch.Tensor, None]:
        output_image = self.model(
            masked_kspace=masked_kspace,
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=sampling_mask,
        )
        return output_image, None


class LPDNetN2NEngine(N2NMRIModelEngine):
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
        """Inits :class:`LPDNetN2NEngine`."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, None]:
        output_image = self.model(
            masked_kspace=data["noisier_kspace"] if self.model.training else data["masked_kspace"],
            sensitivity_map=data["sensitivity_map"],
            sampling_mask=data["noisier_sampling_mask"] if self.model.training else data["sampling_mask"],
        )
        return output_image, None
