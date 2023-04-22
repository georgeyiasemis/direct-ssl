# coding=utf-8
# Copyright (c) DIRECT Contributors

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

__all__ = ["besov_norm", "normalized_besov_norm", "BesovNormLoss", "NormalizedBesovNormLoss"]


def dct(x: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    """Computes the Discrete Cosine Transform (DCT-II) of the input signal.

    Parameters
    ----------
    x : torch.Tensor
        The input signal.
    norm : str, optional
        The normalization factor. Can be None or 'ortho'.

    Returns
    -------
    torch.Tensor
        The DCT-II of the input signal over the last dimension.
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def dct_2d(x: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    """Computes the 2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    Parameters
    ----------
    x : torch.Tensor
        The input signal.
    norm : str, optional
        The normalization factor. Can be None or 'ortho'.

    Returns
    -------
    torch.Tensor
        The DCT-II of the input signal over the last two dimensions.
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def besov_norm(img: torch.Tensor, s: int = 2, p: int = 2) -> torch.Tensor:
    """Computes the Besov norm of the input image.

    img : torch.Tensor
        The input image.
    s : int
        The scaling factor for the Besov decomposition.
    p : int
        The exponent of the Besov norm.

    Returns
    -------
    torch.Tensor
        The Besov norm of the input image.
    """
    # Compute the DCT of the image along the height and width dimensions
    # This converts the image to the frequency domain
    img_dct = dct_2d(img, norm="ortho")

    # Compute the number of scales based on the height and width dimensions of the image
    num_scales = int(np.log2(img.shape[2]))

    # Initialize an empty list to store the seminorms at each scale
    seminorms = []

    # Iterate over the different scales of the Besov norm decomposition
    for j in range(num_scales):
        # Compute the scaling factor for this scale
        scaling_factor = 2 ** (j * s)

        # Compute the Fourier coefficients of the image at the current scale
        # These correspond to the frequency band of the image at the current scale
        coeffs = img_dct[:, :, : img.shape[2] // (2**j), : img.shape[3] // (2**j)]

        # Compute the magnitude of the Fourier coefficients along the channel dimension
        # This corresponds to taking the modulus of the complex Fourier coefficients
        magnitude = coeffs.abs()

        # Raise the magnitude to the power of p and multiply by the scaling factor
        magnitude_p = scaling_factor * torch.pow(magnitude, p)

        # Compute the sum of the magnitude_p along the height, width, and channel dimensions
        # This corresponds to the Besov seminorm at the current scale
        seminorm = magnitude_p.sum((1, 2, 3))

        # Append the seminorm to the list of seminorms at each scale
        seminorms.append(seminorm)

    # Compute the overall Besov norm of the image by summing the seminorms at each scale
    besov_norm_ = torch.cat(seminorms).sum() ** (1 / p)

    return besov_norm_


def besov_norm(input: torch.Tensor, target: torch.Tensor, s: int = 2, p: int = 2) -> torch.Tensor:
    """Computes the Besov norm metric.

    input : torch.Tensor
            Tensor of shape (N, C, H, W).
    target : torch.Tensor
        Tensor of same shape as the input.
    s : int
        The scaling factor for the Besov decomposition.
    p : int
        The exponent of the Besov norm.

    Returns
    -------
    torch.Tensor
        The Besov norm metric between input and target.
    """
    return besov_norm(target - input, s, p)


def normalized_besov_norm(input: torch.Tensor, target: torch.Tensor, s: int = 2, p: int = 2) -> torch.Tensor:
    """Computes the normalized Besov norm metric.

    input : torch.Tensor
            Tensor of shape (N, C, H, W).
    target : torch.Tensor
        Tensor of same shape as the input.
    s : int
        The scaling factor for the Besov decomposition.
    p : int
        The exponent of the Besov norm.

    Returns
    -------
    torch.Tensor
        The Besov norm metric between input and target.
    """
    return besov_norm(target, input, s, p) / besov_norm(target, s, p)


class BesovNormLoss(nn.Module):
    """Computes the Besov norm loss."""

    def __int__(self, s: int = 2, p: int = 2):
        """Inits :class:`BesovNormLoss`.

        Parameters
        ----------
        s : int
            The scaling factor for the Besov decomposition.
        p : int
            The exponent of the Besov norm.

        """
        super().__init__()
        self.s = s
        self.p = p

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward method of :class:`BesovNormLoss`.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape (N, C, H, W).
        target : torch.Tensor
            Tensor of same shape as the input.
        """
        return besov_norm(input, target, self.s, self.p)


class NormalizedBesovNormLoss(nn.Module):
    """Computes the normalized Besov norm loss."""

    def __int__(self, s: int = 2, p: int = 2):
        """Inits :class:`NormalizedBesovNormLoss`.

        Parameters
        ----------
        s : int
            The scaling factor for the Besov decomposition.
        p : int
            The exponent of the Besov norm.

        """
        super().__init__()
        self.s = s
        self.p = p

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward method of :class:`NormalizedBesovNormLoss`.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape (N, C, H, W).
        target : torch.Tensor
            Tensor of same shape as the input.
        """
        return normalized_besov_norm(input, target, self.s, self.p)
