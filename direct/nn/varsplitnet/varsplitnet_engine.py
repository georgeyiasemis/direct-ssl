# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import autocast

import direct.data.transforms as T
from direct.config import BaseConfig
from direct.engine import DoIterationOutput
from direct.nn.mri_models import MRIModelEngine
from direct.nn.ssl.mri_models import (
    DualSSL2MRIModelEngine,
    DualSSLMRIModelEngine,
    MixedLearningEngine,
    N2NMRIModelEngine,
    SSDUMRIModelEngine,
)
from direct.types import TensorOrNone
from direct.utils import detach_dict, dict_to_device


class MRIVarSplitNetEngine(MRIModelEngine):
    """MRIVarSplitNet Engine."""

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
        """Inits :class:`MRIVarSplitNetEngine`.

        Parameters
        ----------
        cfg: BaseConfig
            Configuration file.
        model: nn.Module
            Model.
        device: str
            Device. Can be "cuda:{idx}" or "cpu".
        forward_operator: Callable, optional
            The forward operator. Default: None.
        backward_operator: Callable, optional
            The backward operator. Default: None.
        mixed_precision: bool
            Use mixed precision. Default: False.
        **models: nn.Module
            Additional models.
        """
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
        )  # shape (batch, height,  width, complex[=2])

        output_kspace = None

        return output_image, output_kspace


class MRIVarSplitNetSSDUEngine(SSDUMRIModelEngine):
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
        """Inits :class:`MRIVarSplitNetEngine`."""
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
            padding=data.get("padding", None),
        )
        return None, output_kspace


class MRIVarSplitNetDualSSLEngine(DualSSLMRIModelEngine):
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
        """Inits :class:`MRIVarSplitNetDualSSLEngine`."""
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


class MRIVarSplitNetDualSSL2Engine(DualSSL2MRIModelEngine):
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


class MRIVarSplitNetN2NEngine(N2NMRIModelEngine):
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
        """Inits :class:`MRIVarSplitNetN2NEngine`."""
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


class MRIVarSplitNetMixedEngine(MixedLearningEngine):
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

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, TensorOrNone]:
        is_ssl_training = data["is_ssl_training"][0]

        if is_ssl_training and self.model.training:
            kspace, mask = data["input_kspace"], data["input_sampling_mask"]
        else:
            kspace, mask = data["masked_kspace"], data["sampling_mask"]

        output_image = self.model(
            masked_kspace=kspace,
            sampling_mask=mask,
            sensitivity_map=data["sensitivity_map"],
        )
        # Data consistency
        output_kspace = (
            T.apply_padding(
                kspace + self._forward_operator(output_image, data["sensitivity_map"], ~mask),
                padding=data.get("padding", None),
            )
            if self.model.training
            else None
        )

        return output_image, output_kspace

    def _do_iteration(
        self,
        data: Dict[str, Any],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        if loss_fns is None:
            loss_fns = {}

        loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}

        data = dict_to_device(data, self.device)

        with autocast(enabled=self.mixed_precision):
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

            output_image, output_kspace = self.forward_function(data)

            if self.model.training:
                is_ssl_training = data["is_ssl_training"][0]
                if is_ssl_training:
                    # Project predicted k-space onto target k-space if SSL
                    output_kspace = T.apply_mask(output_kspace, data["target_sampling_mask"], return_mask=False)
                    # # RSS if SSL
                    # output_image = T.root_sum_of_squares(output_kspace, self._coil_dim)
                    output_image = T.modulus(
                        T.reduce_operator(
                            self.backward_operator(output_kspace, dim=self._spatial_dims),
                            data["sensitivity_map"],
                            self._coil_dim,
                        )
                    )
                else:
                    # Modulus if supervised
                    output_image = T.modulus(output_image)

                # Compute k-space loss
                loss_dict = self.compute_loss_on_data(loss_dict, loss_fns, data, None, output_kspace)
                # Compute image loss
                loss_dict = self.compute_loss_on_data(loss_dict, loss_fns, data, output_image, None)

                loss = sum(loss_dict.values())  # type: ignore

                self._scaler.scale(loss).backward()

            else:
                output_image = T.modulus(output_image)

        loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict},
        )
