# coding=utf-8
# Copyright (c) DIRECT Contributors
import torch
import torch.nn as nn

__all__ = ("batch_snr", "SNRLoss")


def batch_snr(input_data, target_data, reduction="mean"):
    """This function is a torch implementation of SNR metric for batches.

    Parameters
    ----------
    input_data : torch.Tensor
    target_data : torch.Tensor
    reduction : str

    Returns
    -------
    torch.Tensor
    """
    batch_size = target_data.size(0)
    input_view = input_data.view(batch_size, -1)
    target_view = target_data.view(batch_size, -1)

    square_error = torch.sum(target_view**2, 1)
    square_error_noise = torch.sum((input_view - target_view) ** 2, 1)
    snr = 10.0 * (torch.log10(square_error) - torch.log10(square_error_noise))

    if reduction == "mean":
        return snr.mean()
    if reduction == "sum":
        return snr.sum()
    if reduction == "none":
        return snr
    raise ValueError(f"Reduction is either `mean`, `sum` or `none`. Got {reduction}.")


class SNRLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input_data, target_data):
        return batch_snr(input_data, target_data, reduction=self.reduction)
