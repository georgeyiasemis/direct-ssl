import math

import numpy as np


def crop(image, crop_size):
    sx, sy, _, _ = image.shape
    cx, cy = crop_size
    start_x = math.floor(sx / 2)
    start_y = math.floor(sy / 2)
    return image[
        start_x + math.ceil(-cx / 2) : start_x + math.ceil(cx / 2),
        start_y + math.ceil(-cy / 2) : start_y + math.ceil(cy / 2),
    ]


def run4Ranking_2025(img, filetype):
    if not img.ndim in [3, 4]:
        raise ValueError(f"Input image must be 3D or 4D but got {img.ndim}D for filetype {filetype}.")

    # check whether this is 'blackblood', 't1w', 't2w'. This should be of shape (sx, sy, sz)
    is_blackblood_t1w_t2w = any(x.lower() in filetype for x in ["blackblood", "t1w", "t2w"])
    # check whether this is a mapping data
    is_mapping = any(x.lower() in filetype for x in ["t1map", "t2map", "t2smap", "t1mappost"])
    # detect T1rho modality (independent but same handling as mapping)
    is_T1rho = "t1rho" in filetype

    if is_blackblood_t1w_t2w:
        sx, sy, sz = img.shape
        st = 1
    else:
        sx, sy, sz, st = img.shape

    if sz < 3:
        start_z, end_z = 0, sz
    else:
        center = round(sz / 2 + 0.00001)
        start_z = max(0, center - 2)
        end_z = min(sz, center)

    if is_blackblood_t1w_t2w or st == 1:
        start_t, end_t = 0, st
    elif is_mapping or is_T1rho:
        start_t, end_t = 0, st
    else:
        start_t, end_t = 0, min(st, 3)

    img = np.reshape(img, (sx, sy, sz, st))

    img = img[:, :, start_z:end_z, start_t:end_t]

    crop_size_x = round(sx / 3)
    crop_size_y = round(sy / 2 + 0.00001)

    img = crop(np.abs(img), (crop_size_x, crop_size_y))
    img4ranking = img.astype(np.float32)

    if is_blackblood_t1w_t2w:
        # For blackblood, t1w, t2w, we crop the image to the center
        img4ranking = np.squeeze(img4ranking, axis=3)
    return img4ranking
