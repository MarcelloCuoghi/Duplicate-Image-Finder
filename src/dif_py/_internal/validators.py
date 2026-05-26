"""Parameter validation for difPy."""

import os
from pathlib import Path

import numpy as np


def validate_directory(directory):
    """Validate and normalize the 'directory' parameter."""
    if len(directory) == 0:
        raise ValueError("Invalid directory parameter: no directory provided.")
    if all(isinstance(dir, list) for dir in directory):
        directory = np.array([item for sublist in directory for item in sublist])
    elif all(isinstance(dir, str) for dir in directory):
        directory = np.array(directory)
    else:
        raise ValueError(
            "Invalid directory parameter: directories must be of type LIST or STRING."
        )

    for dir in directory:
        dir = Path(dir)
        if not (os.path.isdir(dir) or os.path.isfile(dir)):
            raise FileNotFoundError(f'Directory "{dir!s}" does not exist')

    if len(set(directory)) != directory.size:
        raise ValueError(
            "Invalid directory parameters: invalid attempt to compare a directory with itself."
        )

    return sorted(directory)


def validate_recursive(recursive: bool) -> bool:
    """Validate the 'recursive' parameter."""
    if not isinstance(recursive, bool):
        raise TypeError(
            'Invalid value for "recursive" parameter: must be of type BOOL.'
        )
    return recursive


def validate_in_folder(in_folder: bool, recursive: bool) -> bool:
    """Validate the 'in_folder' parameter."""
    if not isinstance(in_folder, bool):
        raise TypeError(
            'Invalid value for "in_folder" parameter: must be of type BOOL.'
        )
    return in_folder


def validate_limit_extensions(limit_extensions: bool) -> bool:
    """Validate the 'limit_extensions' parameter."""
    if not isinstance(limit_extensions, bool):
        raise TypeError(
            'Invalid value for "limit_extensions" parameter: must be of type BOOL.'
        )
    return limit_extensions


def validate_similarity(similarity):
    """Validate and resolve the 'similarity' parameter to a numeric threshold."""
    if similarity not in ["duplicates", "similar"]:
        try:
            similarity = float(similarity)
            if similarity < 0:
                raise ValueError(
                    'Invalid value for "similarity" parameter: must be >= 0.'
                )
            return similarity
        except (ValueError, TypeError):
            raise ValueError(
                'Invalid value for "similarity" parameter: must be "duplicates", '
                '"similar" or of type INT or FLOAT.'
            )
    else:
        if similarity == "duplicates":
            return 0
        elif similarity == "similar":
            return 5


def validate_px_size(px_size: int) -> int:
    """Validate the 'px_size' parameter."""
    if not isinstance(px_size, int):
        raise TypeError('Invalid value for "px_size" parameter: must be of type INT.')
    if px_size < 10 or px_size > 5000:
        raise ValueError(
            'Invalid value for "px_size" parameter: must be between 10 and 5000.'
        )
    return px_size


def validate_rotate(rotate: bool) -> bool:
    """Validate the 'rotate' parameter."""
    if not isinstance(rotate, bool):
        raise TypeError('Invalid value for "rotate" parameter: must be of type BOOL.')
    return rotate


def validate_same_dim(same_dim: bool, similarity) -> bool:
    """Validate the 'same_dim' parameter."""
    if not isinstance(same_dim, bool):
        raise TypeError('Invalid value for "same_dim" parameter: must be of type BOOL.')
    return same_dim


def validate_show_progress(show_progress: bool) -> bool:
    """Validate the 'show_progress' parameter."""
    if not isinstance(show_progress, bool):
        raise TypeError(
            'Invalid value for "show_progress" parameter: must be of type BOOL.'
        )
    return show_progress


def validate_processes(processes: int) -> int:
    """Validate the 'processes' parameter."""
    if not isinstance(processes, int):
        raise TypeError('Invalid value for "processes" parameter: must be of type INT.')
    if processes < 1:
        raise ValueError('Invalid value for "processes" parameter: must be >= 1.')
    if processes > os.cpu_count():
        raise ValueError(
            'Invalid value for "processes" parameter: must be <= the number of CPU cores (os.cpu_count()).'
        )
    return processes


def validate_chunksize(chunksize) -> int | None:
    """Validate the 'chunksize' parameter."""
    if chunksize is None:
        return None
    if not isinstance(chunksize, int):
        raise TypeError(
            'Invalid value for "chunksize" parameter: must be of type INT or None.'
        )
    if chunksize < 1:
        raise ValueError('Invalid value for "chunksize" parameter: must be >= 1.')
    return chunksize


def validate_color_space(color_space: str) -> str:
    """Validate the 'color_space' parameter."""
    if not isinstance(color_space, str):
        raise TypeError(
            'Invalid value for "color_space" parameter: must be of type STR.'
        )
    if color_space not in ("rgb", "gray"):
        raise ValueError(
            'Invalid value for "color_space" parameter: must be "rgb" or "gray".'
        )
    return color_space


def validate_silent_del(silent_del: bool) -> bool:
    """Validate the 'silent_del' parameter."""
    if not isinstance(silent_del, bool):
        raise TypeError(
            'Invalid value for "silent_del" parameter: must be of type BOOL.'
        )
    return silent_del


def validate_move_to(dir: str):
    """Validate and create the 'move_to' destination directory."""
    if not isinstance(dir, str):
        raise TypeError('Invalid value for "move_to" parameter: must be of type STR')
    dir = Path(dir)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError:
            raise OSError(
                f'Invalid value for "move_to" parameter: "{dir!s}" does not exist.'
            )
    elif not os.path.isdir(dir):
        raise ValueError(
            f'Invalid value for "move_to" parameter: "{dir!s}" is not a directory.'
        )
    return dir
