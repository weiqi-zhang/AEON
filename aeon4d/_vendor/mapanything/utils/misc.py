"""
Miscellaneous utility functions.
"""


def invalid_to_zeros(arr, valid_mask, ndim=999):
    """
    Replace invalid values in an array with zeros based on a validity mask.

    Parameters:
    - arr: Input array (typically a PyTorch tensor)
    - valid_mask: Boolean mask indicating valid elements (True) and invalid elements (False)
    - ndim: Maximum number of dimensions to keep; flattens dimensions if arr.ndim > ndim

    Returns:
    - Tuple containing:
      - Modified array with invalid values replaced by zeros
      - nnz: Number of non-zero (valid) elements per sample in the batch
    """
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = (
            arr[..., 0].numel() // len(arr) if len(arr) else 0
        )  # Number of pixels per image
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz

