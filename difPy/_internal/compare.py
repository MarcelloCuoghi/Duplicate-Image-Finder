"""Image comparison utilities for difPy."""

import numpy as np
from PIL import Image


def compute_mse(tensor_A, tensor_B, rotate: bool = True) -> float:
    """Compute the Mean Squared Error between two image tensors.

    When rotate=True, returns the minimum MSE across 4 rotations (0°, 90°, 180°, 270°).
    """
    if rotate:
        mse_list = []
        for rot in range(4):
            if rot > 0:
                tensor_B = np.rot90(tensor_B)
            mse = np.square(np.subtract(tensor_A, tensor_B)).mean()
            mse_list.append(mse)
        return min(mse_list)
    else:
        return float(np.square(np.subtract(tensor_A, tensor_B)).mean())


def compare_shape(tensor_shape_A, tensor_shape_B) -> bool:
    """Check whether two image shapes are dimensionally equivalent (ignoring order)."""
    return sorted(tensor_shape_A) == sorted(tensor_shape_B)


def check_equality(tensor_A, tensor_B) -> bool:
    """Check whether two tensors are element-wise equal."""
    return bool((tensor_A == tensor_B).all())


def sort_imgs_by_size(img_list: list) -> list:
    """Sort images by resolution (highest first) for quality comparison."""
    imgs_sizes = []
    for img in img_list:
        with Image.open(img) as image:
            resolution = image.size
        img_size = (sum(resolution), img)
        imgs_sizes.append(img_size)
    return [file for _size, file in sorted(imgs_sizes, reverse=True)]
