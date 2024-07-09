# coding=utf-8
# Copyright (c) DIRECT Contributors

import logging
import pathlib
from typing import Callable, DefaultDict, Dict, Optional, Union

import h5py  # type: ignore
import numpy as np
import scipy

from direct.cmrxrecon.fastmri import complex_abs, ifft2c, rss
from direct.cmrxrecon.run4ranking import run4Ranking

logger = logging.getLogger(__name__)


def write_output_to_mat(
    output: Union[Dict, DefaultDict],
    output_directory: pathlib.Path,
    task: str,
    volume_processing_func: Optional[Callable] = None,
    output_key: str = "reconstruction",
    create_dirs_if_needed: bool = True,
) -> None:
    """Write dictionary with keys filenames and values torch tensors to h5 files.

    Parameters
    ----------
    output: dict
        Dictionary with keys filenames and values torch.Tensor's with shape [depth, num_channels, ...]
        where num_channels is typically 1 for MRI.
    output_directory: pathlib.Path
    volume_processing_func: callable
        Function which postprocesses the volume array before saving.
    output_key: str
        Name of key to save the output to.
    create_dirs_if_needed: bool
        If true, the output directory and all its parents will be created.

    Notes
    -----
    Currently only num_channels = 1 is supported. If you run this function with more channels the first one
    will be used.
    """

    def set_type(s):
        if "aorta" in s:
            return "Aorta"
        if "cine" in s:
            return "Cine"
        if "map" in s:
            return "Mapping"
        if "tagging" in s:
            return "Tagging"
        if "blood" in s:
            return "BlackBlood"
        if "flow" in s:
            return "Flow2d"
        raise ValueError(f"Unknown type for {s}.")

    if create_dirs_if_needed:
        # Create output directory
        output_directory.mkdir(exist_ok=True, parents=True)

    for idx, (volume, _, filename) in enumerate(output):
        # Volume is (nz, nc, nt, nx, ny, 2)
        if isinstance(filename, pathlib.PosixPath):
            filename = filename.name

        patient_name = str(filename)[:4]
        file_name = str(filename)[5:]

        save_path = (
            output_directory / "MultiCoil" / set_type(file_name) / "ValidationSet" / task / patient_name / file_name
        )

        save_path.parent.mkdir(exist_ok=True, parents=True)
        logger.info(f"({idx + 1}/{len(output)}): Writing {save_path}...")

        volume = volume.permute(0, 1, 2, 4, 3, 5)
        reconstruction = ifft2c(volume)
        reconstruction = complex_abs(reconstruction)  # Compute absolute value to get a real image
        reconstruction = rss(reconstruction, dim=1)

        reconstruction = reconstruction.cpu().numpy()
        reconstruction = reconstruction.transpose(3, 2, 0, 1)

        if "blood" in file_name:
            reconstruction = reconstruction[..., 0]

        img4ranking = run4Ranking(reconstruction, file_name)

        scipy.io.savemat(save_path, {output_key: img4ranking})


def write_output_to_h5(
    output: Union[Dict, DefaultDict],
    output_directory: pathlib.Path,
    volume_processing_func: Optional[Callable] = None,
    output_key: str = "reconstruction",
    create_dirs_if_needed: bool = True,
) -> None:
    """Write dictionary with keys filenames and values torch tensors to h5 files.

    Parameters
    ----------
    output: dict
        Dictionary with keys filenames and values torch.Tensor's with shape [depth, num_channels, ...]
        where num_channels is typically 1 for MRI.
    output_directory: pathlib.Path
    volume_processing_func: callable
        Function which postprocesses the volume array before saving.
    output_key: str
        Name of key to save the output to.
    create_dirs_if_needed: bool
        If true, the output directory and all its parents will be created.

    Notes
    -----
    Currently only num_channels = 1 is supported. If you run this function with more channels the first one
    will be used.
    """
    if create_dirs_if_needed:
        # Create output directory
        output_directory.mkdir(exist_ok=True, parents=True)

    for idx, (volume, _, filename) in enumerate(output):
        # The output has shape (slice, 1, height, width)
        if isinstance(filename, pathlib.PosixPath):
            filename = filename.name

        logger.info(f"({idx + 1}/{len(output)}): Writing {output_directory / filename}...")

        reconstruction = volume.numpy()[:, 0, ...].astype(np.float32)

        if volume_processing_func:
            reconstruction = volume_processing_func(reconstruction)

        with h5py.File(output_directory / filename, "w") as f:
            f.create_dataset(output_key, data=reconstruction)
