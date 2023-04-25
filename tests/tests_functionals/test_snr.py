# coding=utf-8
# Copyright (c) DIRECT Contributors

import numpy as np
import pytest
import torch
from skimage.color import rgb2gray
from sklearn.datasets import load_sample_image

from direct.functionals.snr import SNRLoss

# Load two images and convert them to grayscale
flower = rgb2gray(load_sample_image("flower.jpg"))[None].astype(np.float32)
china = rgb2gray(load_sample_image("china.jpg"))[None].astype(np.float32)


@pytest.mark.parametrize("image", [flower, china])
def test_psnr(image):
    image_batch = []
    image_noise_batch = []
    single_image_psnr = []

    for sigma in range(1, 5):
        noise = sigma * np.random.rand(*image.shape)
        image_noise = (image + noise).astype(np.float32).clip(0, 255)

        image_torch = (torch.from_numpy(image).unsqueeze(0)).float()  # 1, C, H, W
        image_noise_torch = (torch.from_numpy(image_noise).unsqueeze(0)).float()  # 1, C, H, W

        image_batch.append(image_torch)
        image_noise_batch.append(image_noise_torch)

        psnr_torch = SNRLoss(reduction="none").forward(image_noise_torch, image_torch)
        assert psnr_torch == SNRLoss(reduction="sum").forward(image_noise_torch, image_torch)

        psnr_torch = psnr_torch.numpy().item()
        single_image_psnr.append(psnr_torch)

    image_batch = torch.cat(image_batch, dim=0)
    image_noise_batch = torch.cat(image_noise_batch, dim=0)
    psnr_batch = SNRLoss(reduction="mean").forward(
        image_noise_batch,
        image_batch,
    )
    assert np.allclose(psnr_batch, np.average(single_image_psnr), atol=5e-4)
