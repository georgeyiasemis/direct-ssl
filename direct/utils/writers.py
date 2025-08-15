# coding=utf-8
# Copyright (c) DIRECT Contributors

import logging
import pathlib
import re
from typing import Callable, DefaultDict, Dict, Optional, Union

import h5py  # type: ignore
import numpy as np
import scipy

from direct.cmrxrecon2025.run4ranking import run4Ranking_2025
from direct.cmrxrecon.run4ranking import run4Ranking

logger = logging.getLogger(__name__)


def write_output_to_mat(
    output: Union[Dict, DefaultDict],
    output_directory: pathlib.Path,
    task: str,
    volume_processing_func: Optional[Callable] = None,
    output_key: str = "reconstruction",
    create_dirs_if_needed: bool = True,
    set_name: str = "ValidationSet",
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
        s = s.lower()
        if "cine" in s:
            return "Cine"
        if "t1w" in s:
            return "T1w"
        if "t2w" in s:
            return "T2w"
        if "map" in s:
            return "Mapping"
        if "blood" in s:
            return "BlackBlood"
        if "flow" in s:
            return "Flow2d"
        if "t1rho" in s:
            return "T1rho"
        if "perfusion" in s:
            return "Perfusion"
        if "lge" in s:
            return "LGE"
        raise ValueError(f"Unknown type for {s}.")

    pattern = re.compile(r"^(Center\d+)_(.+?)_(P\d+)_([A-Za-z0-9_]+\.mat)$")

    def match_name(filename, task):
        match = pattern.match(filename)
        if match:
            center, machine, patient, file = match.groups()
            typ = set_type(filename)
            path = (
                pathlib.Path("MultiCoil")
                / typ
                / set_name
                / (f"UnderSample_{task}" if set_name == "ValidationSet" else task)
                / center
                / machine
                / patient
                / file
            )

        else:
            raise ValueError(f"Filename did not match pattern: {filename}")
        return path

    if create_dirs_if_needed:
        # Create output directory
        output_directory.mkdir(exist_ok=True, parents=True)

    assert set_name in ["ValidationSet", "TestSet"], "Set name must be ValidationSet or TestSet"
    assert task in ["TaskR1", "TaskR2", "TaskS1", "TaskS2"], "Task must be TaskR1 or TaskR2 or TaskS1 or TaskS2"

    logger.info(f"Writing to {output_directory} with set name {set_name} and task {task}")

    for idx, (volume, _, filename, original_shape) in enumerate(output):
        name = pathlib.Path(filename).name

        tp = set_type(name)

        base_save_name = match_name(name, task)

        save_path = output_directory / base_save_name
        save_path.parent.mkdir(exist_ok=True, parents=True)

        logger.info(
            f"({idx + 1}/{len(output)}): Processing {save_path} with current shape {tuple(volume.shape)}"
            f" and original shape {tuple(original_shape)}..."
        )

        reconstruction = volume[:, 0].cpu().numpy()
        reconstruction = reconstruction.transpose(2, 3, 0, 1)

        # This is because for BlackBlood, T1w, T2w we have 3D images with shape (sx, sy, sz) so the dataset before
        # repeats each slice and stacks them to have a 4D image with shape (sx, sy, sz, st) where st = 12 for BlackBlood
        # and st = 9 for T1w, T2w.
        # So now we need to select only the central time
        if tp == "BlackBlood":
            reconstruction = reconstruction[..., 6]
            reconstruction = np.expand_dims(reconstruction, axis=-1)
        elif tp in ["T1w", "T2w"]:
            reconstruction = reconstruction[..., 4]
            reconstruction = np.expand_dims(reconstruction, axis=-1)

        img4ranking = reconstruction.reshape(*original_shape)

        logger.info(
            f"({idx + 1}/{len(output)}): Writing {save_path} with shape {tuple(img4ranking.shape)} and "
            f"original shape {tuple(original_shape)}..."
        )

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
