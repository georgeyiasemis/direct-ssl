# coding=utf-8
# Copyright (c) DIRECT Contributors

import torch


def freeze_module(module):
    """
    Freeze the parameters of a module in a PyTorch model.

    Parameters
    ----------
    module : torch.nn.Module
        The module whose parameters need to be frozen.

    Returns
    -------
    None
        This function modifies the parameters of the input module in-place.
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """
    Unfreeze the parameters of a module in a PyTorch model.

    Parameters
    ----------
    module : torch.nn.Module
        The module whose parameters need to be unfrozen.

    Returns
    -------
    None
        This function modifies the parameters of the input module in-place.
    """
    for param in module.parameters():
        param.requires_grad = True
