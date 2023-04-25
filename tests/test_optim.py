# coding=utf-8
# Copyright (c) DIRECT Contributors

import pytest
import torch
from torch import nn

from direct.optim import AdamW, Lion


@pytest.mark.parametrize(
    "lr",
    [1e-4, 2e-5],
)
@pytest.mark.parametrize(
    "weight_decay",
    [0.0001, 0.001],
)
def test_optimizers(lr, weight_decay):
    model = nn.Linear(10, 1)
    for optimizer in [AdamW, Lion]:
        opt = optimizer(model.parameters(), lr=1e-4, weight_decay=1e-2)
        # forward and backwards
        loss = model(torch.randn(10))
        loss.backward()
        # optimizer step
        opt.step()
        opt.zero_grad()
