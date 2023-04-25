# coding=utf-8
# Copyright (c) DIRECT Contributors

import numpy as np
import pytest
import torch
from skimage.color import rgb2gray
from sklearn.datasets import load_sample_image

from direct.functionals import BesovNormLoss, NormalizedBesovNormLoss, besov_norm

# Load two images and convert them to grayscale
flower = rgb2gray(load_sample_image("flower.jpg"))[None].astype(np.float32)
china = rgb2gray(load_sample_image("china.jpg"))[None].astype(np.float32)


@pytest.mark.parametrize("image", [flower, china])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("s", [1.0, 2.0])
@pytest.mark.parametrize("p", [1.0, 1.5])
def test_nmse(image, reduction, s, p):
    image_batch = []
    image_noise_batch = []

    for sigma in range(1, 5):
        noise = sigma * np.random.rand(*image.shape)
        image_noise = (image + noise).astype(np.float32).clip(0, 255)

        image_batch.append(image)
        image_noise_batch.append(image_noise)

    image_batch_torch = torch.tensor(image_batch)
    image_noise_batch_torch = torch.tensor(image_noise_batch)

    besov_norm_loss = BesovNormLoss(reduction=reduction, s=s, p=p)(image_noise_batch_torch, image_batch_torch)
    normalized_besov_norm_loss = NormalizedBesovNormLoss(reduction=reduction, s=s, p=p)(
        image_noise_batch_torch, image_batch_torch
    )

    assert normalized_besov_norm_loss == besov_norm_loss / besov_norm(
        torch.zeros_like(image_batch_torch), image_batch_torch, reduction=reduction, s=s, p=p
    )
