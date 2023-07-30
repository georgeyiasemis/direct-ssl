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
from direct.nn.ssl.mri_models import SSDUMRIModelEngine
from direct.utils import detach_dict, dict_to_device, normalize_image
from direct.utils.events import get_event_storage


class ConjGradNetEngine(MRIModelEngine):
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
        """Inits :class:`ConjGradNetEngine`.

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
        output_image = self.model(
            masked_kspace=data["masked_kspace"],
            sampling_mask=data["sampling_mask"],
            sensitivity_map=data["sensitivity_map"],
        )  # shape (batch, height,  width)
        output_kspace = None
        return output_image, output_kspace


class ConjGradNetSSDUEngine(SSDUMRIModelEngine):
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
        """Inits :class:`ConjGradNetSSDUEngine`.

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


class ConjGradNetMixedEngine(MRIModelEngine):
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

    def forward_function(self, data: Dict[str, Any]) -> None:
        pass

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

            if self.model.training:
                # Data consistency
                output_kspace = T.apply_padding(
                    kspace + self._forward_operator(output_image, data["sensitivity_map"], ~mask),
                    padding=data["padding"],
                )

                if is_ssl_training:
                    # Project predicted k-space onto target k-space if SSL
                    output_kspace = T.apply_mask(output_kspace, data["target_sampling_mask"], return_mask=False)
                    # RSS if SSL
                    output_image = T.root_sum_of_squares(output_kspace, self._coil_dim)
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

    def log_first_training_example_and_model(self, data):
        storage = get_event_storage()
        self.logger.info(f"First case: slice_no: {data['slice_no'][0]}, filename: {data['filename'][0]}.")

        if "sampling_mask" in data:
            first_sampling_mask = data["sampling_mask"][0][0]
        elif "input_sampling_mask" in data:  # ssdu
            first_input_sampling_mask = data["input_sampling_mask"][0][0]
            first_target_sampling_mask = data["target_sampling_mask"][0][0]
            storage.add_image("train/input_mask", first_input_sampling_mask[..., 0].unsqueeze(0))
            storage.add_image("train/target_mask", first_target_sampling_mask[..., 0].unsqueeze(0))
            first_sampling_mask = first_target_sampling_mask | first_input_sampling_mask
        elif "theta_sampling_mask" in data:  # dualssl
            first_theta_sampling_mask = data["theta_sampling_mask"][0][0]
            first_lambda_sampling_mask = data["lambda_sampling_mask"][0][0]
            storage.add_image("train/theta_mask", first_theta_sampling_mask[..., 0].unsqueeze(0))
            storage.add_image("train/lambda_mask", first_lambda_sampling_mask[..., 0].unsqueeze(0))
            first_sampling_mask = first_theta_sampling_mask | first_lambda_sampling_mask
        else:  # noisier2noise
            first_noisier_sampling_mask = data["noisier_sampling_mask"][0][0]
            storage.add_image("train/noisier_mask", first_noisier_sampling_mask[..., 0].unsqueeze(0))
            first_sampling_mask = data["sampling_mask"][0][0]
        first_target = data["target"][0]

        if self.ndim == 3:
            first_sampling_mask = first_sampling_mask[0]
            slice_dim = -4
            num_slices = first_target.shape[slice_dim]
            first_target = first_target[num_slices // 2]
        elif self.ndim > 3:
            raise NotImplementedError

        storage.add_image("train/mask", first_sampling_mask[..., 0].unsqueeze(0))
        storage.add_image(
            "train/target",
            normalize_image(first_target.unsqueeze(0)),
        )

        if "initial_image" in data:
            storage.add_image(
                "train/initial_image",
                normalize_image(T.modulus(data["initial_image"][0]).unsqueeze(0)),
            )

        self.write_to_logs()
