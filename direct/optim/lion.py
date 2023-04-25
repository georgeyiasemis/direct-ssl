# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code copied and slightly adapted from https://github.com/google/automl/blob/master/lion/lion_pytorch.py
# Copyright 2023 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch implementation of the Lion optimizer."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(
        self, params: Iterable, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0
    ):
        """Init :class:`Lion`.

        Parameters
        ----------
        params : iterable
            Iterable of parameters to optimize or dicts defining parameter groups
        lr : float
            Learning rate. Default: 1e-4.
        betas : tuple[float, float]
            Coefficients used for computing running averages of gradient and its square. Default: (0.9, 0.99).
        weight_decay : float
            Weight decay coefficient. Default: 0.
        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step.

        Parameters
        ----------
        closure : Callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss: torch.Tensor
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
