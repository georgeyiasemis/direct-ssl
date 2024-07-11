from __future__ import annotations

from tempfile import NamedTemporaryFile as NTF
from typing import Optional

import numpy as np
import torch


def view_as_windows_torch(image, shape, stride=None):
    """View tensor as overlapping rectangular windows, with a given stride.

    Parameters
    ----------
    image : `~torch.Tensor`
        4D image tensor, with the last two dimensions
        being the image dimensions
    shape : tuple of int
        Shape of the window.
    stride : tuple of int
        Stride of the windows. By default it is half of the window size.

    Returns
    -------
    windows : torch.Tensor
        Tensor of overlapping windows.
    """
    if stride is None:
        stride = (1 for s in shape)

    windows = image.clone()
    for ind, (shape_dim, stride_dim) in enumerate(zip(shape, stride)):
        windows = windows.unfold(ind + 2, shape_dim, stride_dim)
    return windows


class GrappaTorch(object):
    """GRAPPA object for computing GRAPPA weights and applying them to k-space data.

    This implementation is based on the GRAPPA algorithm as presented in [1]_. The difference lies in the fact that
    this implementation is based on PyTorch tensors. However, for efficiency reasons, the implementation still uses
    numpy's `memmap` to store temporary files to avoid overwhelming memory usage.

    Parameters
    ----------
    kspace : torch.Tensor
        The k-space data to compute GRAPPA weights for.
    kernel_size : tuple[int, int]
        The size of the GRAPPA kernel. Default is (5, 5).
    coil_axis : int
        The axis of the coil dimension. Default is -1.

    References
    ----------
    .. [1] https://github.com/cai2r/fastMRI_prostate/blob/main/fastmri_prostate/reconstruction/grappa.py
    """

    def __init__(self, kspace: torch.Tensor, kernel_size: tuple[int, int] = (5, 5), coil_axis: int = -1) -> None:
        super().__init__()
        """Initialize the GRAPPA object.
        
        Parameters
        ----------
        kspace : torch.Tensor
            The k-space data to compute GRAPPA weights for.
        kernel_size : tuple[int, int]
            The size of the GRAPPA kernel. Default is (5, 5).
        coil_axis : int
            The axis of the coil dimension. Default is -1.
        """
        self.kspace = kspace
        self.kernel_size = kernel_size
        self.coil_axis = coil_axis
        self.lamda = 0.01
        self.device = kspace.device

        self.kernel_var_dict = self.get_kernel_geometries()

    def get_kernel_geometries(self) -> dict[str, np.ndarray]:
        """Extract unique kernel geometries based on a slice of kspace data

        Returns
        -------
        geometries : dict
            A dictionary containing the following keys:
            - 'patches': an array of overlapping patches from the k-space data slice.
            - 'patch_indices': an array of unique patch indices.
            - 'holes_x': a dictionary of x-coordinates for holes in each patch.
            - 'holes_y': a dictionary of y-coordinates for holes in each patch.

        Notes
        -----
        This function extracts unique kernel geometries from a slice of k-space data. The geometries correspond to
        overlapping patches that contain at least one hole. A hole is defined as a region of k-space data where the
        absolute value of the complex signal is equal to zero. The function returns a dictionary containing
        information about the patches and holes, which can be used to compute weights for each geometry using the
        GRAPPA algorithm.
        """
        self.kspace = torch.moveaxis(self.kspace, self.coil_axis, -1)

        # Quit early if there are no holes
        if torch.sum((torch.abs(self.kspace[..., 0]) == 0).flatten()) == 0:
            return torch.moveaxis(self.kspace, -1, self.coil_axis)

        kx, ky = self.kernel_size[:]
        kx2, ky2 = int(kx / 2), int(ky / 2)
        num_coils = self.kspace.shape[-1]

        self.kspace = torch.nn.functional.pad(self.kspace, (0, 0, ky2, ky2, kx2, kx2), mode="constant")
        mask = torch.abs(self.kspace[..., 0]) > 0

        # Get all overlapping patches from the mask
        patches = view_as_windows_torch(mask[None, None], (kx, ky), (1, 1))[0, 0]
        patches_shape = patches.shape  # save shape for unflattening indices later
        patches = patches.reshape((-1, kx, ky))

        # Find the unique patches and associate them with indices
        patches, iidx = torch.unique(patches, return_inverse=True, dim=0)

        # Filter out geometries that don't have a hole at the center.
        # These are all the kernel geometries we actually need to
        # compute weights for.
        validP = torch.nonzero(~patches[:, kx2, ky2]).squeeze()
        # ignore empty patches
        invalidP = torch.nonzero(torch.all(patches == 0, dim=(1, 2)))
        validP = torch.tensor([p for p in validP if p not in invalidP])

        # Give it back its coil dimension
        patches = patches.unsqueeze(-1).repeat(1, 1, 1, num_coils)

        holes_x = {}
        holes_y = {}

        for ii in validP:
            # x, y define where top left corner is, so move to ctr,
            # also make sure they are iterable as 1d array
            idx = torch.unravel_index(torch.nonzero(iidx == ii, as_tuple=False), patches_shape[:2])
            x, y = idx[0] + kx2, idx[1] + ky2
            x = x.reshape(-1)
            y = y.reshape(-1)

            holes_x[ii.item()] = x.cpu().numpy()
            holes_y[ii.item()] = y.cpu().numpy()

        return {
            "patches": patches.cpu().numpy(),
            "patch_indices": validP.cpu().numpy(),
            "holes_x": holes_x,
            "holes_y": holes_y,
        }

    def compute_weights(self, calib: torch.Tensor) -> dict[int, np.ndarray]:
        """Compute GRAPPA weights for the given calibration data.

        Parameters
        ----------
        calib : torch.Tensor
            The calibration data of shape (num_coils, num_rows_cal, num_cols_cal) to compute GRAPPA weights for.

        Returns
        -------
        weights : dict
            A dictionary containing the GRAPPA weights for each patch index.

        Notes
        -----
        The GRAPPA algorithm is used to estimate the missing k-space data in undersampled MRI acquisitions.
        The algorithm used to compute the GRAPPA weights involves first extracting patches from the calibration data,
        and then solving a linear system to estimate the weights. The resulting weights are stored in a dictionary
        where the key is the patch index. The equation to solve for the weights involves taking the product of the
        sources and the targets in the patch domain, and then regularizing the matrix using Tikhonov regularization.
        The function uses numpy's `memmap` to store temporary files to avoid overwhelming memory usage.
        """
        calib = torch.moveaxis(calib, self.coil_axis, -1)
        kx, ky = self.kernel_size[:]
        kx2, ky2 = int(kx / 2), int(ky / 2)
        num_coils = calib.shape[-1]

        calib = torch.nn.functional.pad(calib, (0, 0, ky2, ky2, kx2, kx2), mode="constant")

        # Store windows in temporary files so we don't overwhelm memory
        with NTF() as fA:
            # Get all overlapping patches of ACS
            try:
                patches = np.memmap(
                    fA,
                    dtype=np.complex64,
                    mode="w+",
                    shape=(calib.shape[0] - 2 * kx, calib.shape[1] - 2 * ky, 1, kx, ky, num_coils),
                )
                patches[:] = (
                    view_as_windows_torch(calib[None, None], (kx, ky, num_coils))[0, 0]
                    .reshape((-1, kx, ky, num_coils))
                    .cpu()
                    .numpy()
                )
            except ValueError:
                patches = (
                    view_as_windows_torch(calib[None, None], (kx, ky, num_coils))[0, 0]
                    .reshape((-1, kx, ky, num_coils))
                    .cpu()
                    .numpy()
                )

            weights = {}
            for ii in self.kernel_var_dict["patch_indices"]:
                # Get the sources by masking all patches of the ACS and
                # get targets by taking the center of each patch. Source
                # and targets will have the following sizes:
                #     S : (# samples, N possible patches in ACS)
                #     T : (# coils, N possible patches in ACS)
                # Solve the equation for the weights: using numpy.linalg.solve,
                # and Tikhonov regularization for better conditioning:
                #     SW = T
                #     S^HSW = S^HT
                #     W = (S^HS)^-1 S^HT
                #  -> W = (S^HS + lamda I)^-1 S^HT
                S = torch.from_numpy(patches[:, self.kernel_var_dict["patches"][ii, ...]]).to(calib.device)
                T = torch.from_numpy(patches[:, kx2, ky2, :]).to(calib.device)

                ShS = torch.conj(S).permute(1, 0) @ S
                ShT = torch.conj(S).permute(1, 0) @ T
                lamda0 = self.lamda * torch.norm(ShS) / ShS.shape[0]

                weights[ii] = (
                    torch.linalg.solve(ShS + lamda0 * torch.eye(ShS.shape[0], device=self.device), ShT)
                    .permute(1, 0)
                    .cpu()
                    .numpy()
                )

        return weights

    def apply_weights(self, kspace: np.ndarray, weights: dict[int, np.ndarray]) -> np.ndarray:
        """Applies the computed GRAPPA weights to the k-space data.

        Parameters
        ----------
        kspace : numpy.ndarray
            The k-space data to apply the weights to.

        weights : dict
            A dictionary containing the GRAPPA weights to apply.

        Returns
        -------
        numpy.ndarray
            The reconstructed data after applying the weights.
        """
        # Put the coil dimension at the end
        kspace = np.moveaxis(kspace, self.coil_axis, -1)

        # Get shape of kernel
        kx, ky = self.kernel_size[:]
        kx2, ky2 = int(kx / 2), int(ky / 2)

        # adjustment factor for odd kernel size
        adjx = np.mod(kx, 2)
        adjy = np.mod(ky, 2)

        # Pad kspace data
        kspace = np.pad(kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode="constant")

        with NTF() as frecon:
            # Initialize recon array
            recon = np.memmap(frecon, dtype=kspace.dtype, mode="w+", shape=kspace.shape)

            for ii in self.kernel_var_dict["patch_indices"]:
                for xx, yy in zip(self.kernel_var_dict["holes_x"][ii], self.kernel_var_dict["holes_y"][ii]):
                    # Collect sources for this hole and apply weights
                    S = kspace[xx - kx2 : xx + kx2 + adjx, yy - ky2 : yy + ky2 + adjy, :]

                    S = S[self.kernel_var_dict["patches"][ii, ...]]
                    recon[xx, yy, :] = (weights[ii] @ S[:, None]).squeeze()
            return np.moveaxis((recon[:] + kspace)[kx2:-kx2, ky2:-ky2, :], -1, self.coil_axis)


def grappa_reconstruction_torch(
    kspace_data: torch.Tensor, calib_data: torch.Tensor, kernel_geometry_slice: int = 0
) -> torch.Tensor:
    """Perform GRAPPA technique on pytorch k-space tensor.

    Parameters
    ----------
    kspace_data : torch.Tensor
        Input k-space data with shape (num_coils, num_slices, num_rows, num_cols).
    calib_data : torch.Tensor
        Calibration data for GRAPPA with shape (num_coils, num_slices, num_rows_cal, num_cols_cal).

    Returns
    -------
    ksapce_post_grappa : torch.Tensor
        k-space with shape (num_coils, num_slices, num_rows, num_cols).

    """
    kspace_data = kspace_data.permute(1, 3, 0, 2)  # (num_slices, num_cols, num_coils, num_rows)
    calib_data = calib_data.permute(1, 3, 0, 2)  # (num_slices, num_cols_cal, num_coils, num_rows_cal)

    grappa_obj = GrappaTorch(kspace_data[kernel_geometry_slice], kernel_size=(5, 5), coil_axis=1)

    kspace_post_grappa = torch.zeros(kspace_data.shape, dtype=kspace_data.dtype)
    for slice_num in range(kspace_data.shape[0]):
        # calculate GRAPPA weights for each slice
        grappa_weight_dict = grappa_obj.compute_weights(calib_data[slice_num])
        # apply GRAPPA weights to each slice
        kspace_post_grappa[slice_num] = torch.from_numpy(
            grappa_obj.apply_weights(kspace_data[slice_num].cpu().numpy(), grappa_weight_dict)
        ).to(kspace_data.device)

    return kspace_post_grappa.permute(2, 0, 3, 1)  # (num_coils, num_slices, num_rows, num_cols)


def grappa_reconstruction_torch_batch(
    kspace_data: torch.Tensor, calib_data: torch.Tensor, kernel_geometry_slice: Optional[int | tuple] = None
) -> torch.Tensor:
    """Perform GRAPPA technique on pytorch k-space tensor.

    Parameters
    ----------
    kspace_data : torch.Tensor
        Input k-space data with shape (batch_size, num_coils, num_slices, num_rows, num_cols).
    calib_data : torch.Tensor
        Calibration data for GRAPPA with shape (batch_size, num_coils, num_slices, num_rows_cal, num_cols_cal).
    kernel_geometry_slice : int or tuple, optional
        The slice of the kernel geometry to use for each batch. If an integer is provided, the same slice will be used
        for all batches. If a tuple is provided, the slice will be used for each batch. Default is None.

    Returns
    -------
    ksapce_post_grappa : torch.Tensor
        k-space with shape (batch_size, num_coils, num_slices, num_rows, num_cols).
    """
    batch_size = kspace_data.shape[0]
    if isinstance(kernel_geometry_slice, int):
        kernel_geometry_slice = (kernel_geometry_slice,) * batch_size
    if kernel_geometry_slice is None:
        kernel_geometry_slice = [kspace_data[_].shape[0] // 2 for _ in range(batch_size)]
    kspace_data = torch.view_as_complex(kspace_data)
    kspace_grappa = torch.zeros(kspace_data.shape, dtype=kspace_data.dtype)
    calib_data = torch.view_as_complex(calib_data)
    for batch_idx in range(batch_size):
        kspace_grappa[batch_idx] = grappa_reconstruction_torch(
            kspace_data[batch_idx], calib_data[batch_idx], kernel_geometry_slice[batch_idx]
        )
    return torch.view_as_real(kspace_grappa)
