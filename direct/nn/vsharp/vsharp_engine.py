# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import autocast

from direct.config import BaseConfig
from direct.data import transforms as T
from direct.engine import DoIterationOutput
from direct.nn.mri_models import MRIModelEngine
from direct.nn.ssl.mri_models import SSDUMRIModelEngine
from direct.types import TensorOrNone
from direct.utils import detach_dict, dict_to_device


class VSharpNetEngine(MRIModelEngine):
    """VSharpNet Engine."""

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
        """Inits :class:`VSharpNetEngine`.

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

    def _do_iteration(
        self,
        data: Dict[str, Any],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """Performs forward method and calculates loss functions.

        Parameters
        ----------
        data : Dict[str, Any]
            Data containing keys with values tensors such as k-space, image, sensitivity map, etc.
        loss_fns : Optional[Dict[str, Callable]]
            Callable loss functions.
        regularizer_fns : Optional[Dict[str, Callable]]
            Callable regularization functions.

        Returns
        -------
        DoIterationOutput
            Contains outputs.
        """

        # loss_fns can be None, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        data = dict_to_device(data, self.device)

        output_image: TensorOrNone
        output_kspace: TensorOrNone

        with autocast(enabled=self.mixed_precision):
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

            output_images, output_kspace = self.forward_function(data)
            output_images = [T.modulus_if_complex(_, complex_axis=self._complex_dim) for _ in output_images]
            loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}

            auxiliary_loss_weights = torch.logspace(-1, 0, steps=len(output_images)).to(output_images[0])
            for i in range(len(output_images)):
                loss_dict = self.compute_loss_on_data(
                    loss_dict, loss_fns, data, output_images[i], None, auxiliary_loss_weights[i]
                )

            loss_dict = self.compute_loss_on_data(
                loss_dict, loss_fns, data, None, output_kspace, auxiliary_loss_weights[i]
            )

            loss = sum(loss_dict.values())  # type: ignore

        if self.model.training:
            self._scaler.scale(loss).backward()

        loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.

        output_image = output_images[-1]
        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict},
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, None]:
        data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

        output_images = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )  # shape (batch, height,  width, complex[=2])

        output_image = output_images[-1]
        output_kspace = data["masked_kspace"] + T.apply_mask(
            T.apply_padding(
                self.forward_operator(
                    T.expand_operator(output_image, data["sensitivity_map"], dim=self._coil_dim),
                    dim=self._spatial_dims,
                ),
                padding=data["padding"],
            ),
            ~data["sampling_mask"],
            return_mask=False,
        )

        return output_images, output_kspace


class VSharpNetSSDUEngine(SSDUMRIModelEngine):
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
        """Inits :class:`VSharpNetEngine`."""
        super().__init__(
            cfg,
            model,
            device,
            forward_operator=forward_operator,
            backward_operator=backward_operator,
            mixed_precision=mixed_precision,
            **models,
        )

    def forward_function(self, data: Dict[str, Any]) -> Tuple[TensorOrNone, TensorOrNone]:
        pass

    def _do_iteration(
        self,
        data: Dict[str, Any],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        if loss_fns is None:
            loss_fns = {}

        data = dict_to_device(data, self.device)

        kspace = data["input_kspace"] if self.model.training else data["masked_kspace"]
        mask = data["input_sampling_mask"] if self.model.training else data["sampling_mask"]
        loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}

        with autocast(enabled=self.mixed_precision):
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

            output_images = self.model(
                masked_kspace=kspace,
                sensitivity_map=data["sensitivity_map"],
                sampling_mask=mask,
            )

            # In SSDU training we use output kspace to compute loss
            if self.model.training:
                if len(output_images) > 1:
                    auxiliary_loss_weights = torch.logspace(-1, 0, steps=len(output_images)).to(output_images[0])
                else:
                    auxiliary_loss_weights = torch.ones(1).to(output_images[0])
                for i in range(len(output_images)):
                    # Data consistency
                    output_kspace = T.apply_padding(
                        kspace + self._forward_operator(output_images[i], data["sensitivity_map"], ~mask),
                        padding=data.get("padding", None),
                    )
                    # Project predicted k-space onto target k-space
                    output_kspace = T.apply_mask(output_kspace, data["target_sampling_mask"], return_mask=False)
                    loss_dict = self.compute_loss_on_data(
                        loss_dict, loss_fns, data, None, output_kspace, auxiliary_loss_weights[i]
                    )

                    # SENSE reconstruction
                    output_images[i] = T.modulus(
                        T.reduce_operator(
                            self.backward_operator(output_kspace, dim=self._spatial_dims),
                            data["sensitivity_map"],
                            self._coil_dim,
                        )
                    )
                    loss_dict = self.compute_loss_on_data(
                        loss_dict, loss_fns, data, output_images[i], None, auxiliary_loss_weights[i]
                    )
                output_image = output_images[i]

                loss = sum(loss_dict.values())  # type: ignore

                if self.model.training:
                    self._scaler.scale(loss).backward()

                loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.

            else:
                output_kspace = T.apply_padding(
                    kspace + self._forward_operator(output_images[-1], data["sensitivity_map"], ~mask),
                    padding=data.get("padding", None),
                )

                # SENSE reconstruction
                output_image = T.modulus(
                    T.reduce_operator(
                        self.backward_operator(output_kspace, dim=self._spatial_dims),
                        data["sensitivity_map"],
                        self._coil_dim,
                    )
                )

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict=loss_dict if self.model.training else {},
        )


class VSharpNetMixedEngine(MRIModelEngine):
    """VSharpNet Engine."""

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
        """Inits :class:`VSharpNetMixedEngine`.

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

    def forward_function(self, data: Dict[str, Any]) -> None:
        pass

    def _do_iteration(
        self,
        data: Dict[str, Any],
        loss_fns: Optional[Dict[str, Callable]] = None,
        regularizer_fns: Optional[Dict[str, Callable]] = None,
    ) -> DoIterationOutput:
        """Performs forward method and calculates loss functions.

        Parameters
        ----------
        data : Dict[str, Any]
            Data containing keys with values tensors such as k-space, image, sensitivity map, etc.
        loss_fns : Optional[Dict[str, Callable]]
            Callable loss functions.
        regularizer_fns : Optional[Dict[str, Callable]]
            Callable regularization functions.

        Returns
        -------
        DoIterationOutput
            Contains outputs.
        """

        # loss_fns can be None, e.g. during validation
        if loss_fns is None:
            loss_fns = {}

        loss_dict = {k: torch.tensor([0.0], dtype=data["target"].dtype).to(self.device) for k in loss_fns.keys()}

        data = dict_to_device(data, self.device)

        with autocast(enabled=self.mixed_precision):
            data["sensitivity_map"] = self.compute_sensitivity_map(data["sensitivity_map"])

            is_ssl_training = data["is_ssl_training"][0]

            if is_ssl_training and self.model.training:
                kspace, mask = data["input_kspace"], data["input_sampling_mask"]
            else:
                kspace, mask = data["masked_kspace"], data["sampling_mask"]

            output_images = self.model(
                masked_kspace=kspace,
                sampling_mask=mask,
                sensitivity_map=data["sensitivity_map"],
            )

            if self.model.training:
                if len(output_images) > 1:
                    auxiliary_loss_weights = torch.logspace(-1, 0, steps=len(output_images)).to(output_images[0])
                else:
                    auxiliary_loss_weights = torch.ones(1).to(output_images[0])

                for i in range(len(output_images)):
                    # Data consistency
                    output_kspace = T.apply_padding(
                        kspace + self._forward_operator(output_images[i], data["sensitivity_map"], ~mask),
                        padding=data["padding"],
                    )
                    if is_ssl_training:
                        # Project predicted k-space onto target k-space if SSL
                        output_kspace = T.apply_mask(output_kspace, data["target_sampling_mask"], return_mask=False)

                    # Compute k-space loss per auxiliary step
                    loss_dict = self.compute_loss_on_data(
                        loss_dict, loss_fns, data, None, output_kspace, auxiliary_loss_weights[i]
                    )

                    # SENSE reconstruction if SSL else modulus if supervised
                    output_images[i] = T.modulus(
                        T.reduce_operator(
                            self.backward_operator(output_kspace, dim=self._spatial_dims),
                            data["sensitivity_map"],
                            self._coil_dim,
                        )
                        if is_ssl_training
                        else output_images[i]
                    )

                    # Compute image loss per auxiliary step
                    loss_dict = self.compute_loss_on_data(
                        loss_dict, loss_fns, data, output_images[i], None, auxiliary_loss_weights[i]
                    )

                    loss = sum(loss_dict.values())  # type: ignore

                self._scaler.scale(loss).backward()

                output_image = output_images[-1]
            else:
                output_image = T.modulus(output_images[-1])

        loss_dict = detach_dict(loss_dict)  # Detach dict, only used for logging.

        return DoIterationOutput(
            output_image=output_image,
            sensitivity_map=data["sensitivity_map"],
            data_dict={**loss_dict},
        )
