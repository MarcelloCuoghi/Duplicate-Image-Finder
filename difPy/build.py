"""difPy Build: creates the image repository from directories."""

import os
import warnings
from datetime import datetime
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from ._internal import validators
from ._internal.stats import build_stats as _generate_build_stats
from ._internal.utils import initialize_multiprocessing, progress_bar

VALID_EXTENSIONS = frozenset([
    "apng",
    "bw",
    "cdf",
    "cur",
    "dcx",
    "dds",
    "dib",
    "emf",
    "eps",
    "fli",
    "flc",
    "fpx",
    "ftex",
    "fits",
    "gd",
    "gd2",
    "gif",
    "gbr",
    "icb",
    "icns",
    "iim",
    "ico",
    "im",
    "imt",
    "j2k",
    "jfif",
    "jfi",
    "jif",
    "jp2",
    "jpe",
    "jpeg",
    "jpg",
    "jpm",
    "jpf",
    "jpx",
    "mic",
    "mpo",
    "msp",
    "nc",
    "pbm",
    "pcd",
    "pcx",
    "pgm",
    "png",
    "ppm",
    "psd",
    "pixar",
    "ras",
    "rgb",
    "rgba",
    "sgi",
    "spi",
    "spider",
    "sun",
    "tga",
    "tif",
    "tiff",
    "vda",
    "vst",
    "wal",
    "webp",
    "xbm",
    "xpm",
])


class build:
    """Initialize difPy and build its image repository.

    Parameters
    ----------
    directory : str, list
        Paths of the directories or files to be searched.
    recursive : bool, optional
        Search recursively within directories (default True).
    in_folder : bool, optional
        If True, searches for matches in separate/isolated directories (default False).
    limit_extensions : bool, optional
        Limit search to known image file extensions (default True).
    px_size : int, optional
        Image compression size in pixels (default 50).
    show_progress : bool, optional
        Show progress bar in console (default True).
    processes : int, optional
        Number of worker processes for multiprocessing.
    """

    def __init__(
        self,
        *directory,
        recursive=True,
        in_folder=False,
        limit_extensions=True,
        px_size=50,
        show_progress=True,
        processes=os.cpu_count(),
    ):
        self._directory = validators.validate_directory(directory)
        self._recursive = validators.validate_recursive(recursive)
        self._in_folder = validators.validate_in_folder(in_folder, recursive)
        self._limit_extensions = validators.validate_limit_extensions(limit_extensions)
        self._px_size = validators.validate_px_size(px_size)
        self._show_progress = validators.validate_show_progress(show_progress)
        self._processes = validators.validate_processes(processes)

        initialize_multiprocessing()

        (
            self._tensor_dictionary,
            self._id_to_shape_dictionary,
            self._filename_dictionary,
            self._id_to_group_dictionary,
            self._group_to_id_dictionary,
            self._invalid_files,
            self.stats,
        ) = self._run()

    def _run(self):
        """Execute the full build workflow."""
        if self._show_progress:
            count = 0
            total_count = 3
            progress_bar(count, total_count, task="preparing files")

        start_time = datetime.now()

        valid_files, skipped_files = self._get_files()
        if self._show_progress:
            count += 1
            progress_bar(count, total_count, task="preparing files")

        (
            tensor_dictionary,
            id_to_shape_dictionary,
            filename_dictionary,
            id_to_group_dictionary,
            group_to_id_dictionary,
            invalid_files,
        ) = self._build_image_dictionaries(valid_files)

        end_time = datetime.now()
        if self._show_progress:
            count += 1
            progress_bar(count, total_count, task="preparing files")

        stats = _generate_build_stats(
            total_files=len(filename_dictionary),
            invalid_files=invalid_files,
            skipped_files=skipped_files,
            directory=self._directory,
            start_time=start_time,
            end_time=end_time,
            recursive=self._recursive,
            in_folder=self._in_folder,
            limit_extensions=self._limit_extensions,
            px_size=self._px_size,
            processes=self._processes,
        )

        if self._show_progress:
            count += 1
            progress_bar(count, total_count, task="preparing files")

        return (
            tensor_dictionary,
            id_to_shape_dictionary,
            filename_dictionary,
            id_to_group_dictionary,
            group_to_id_dictionary,
            invalid_files,
            stats,
        )

    def _get_files(self):
        """Search for files in the input directories."""
        valid_files_all = np.array([], dtype=object)
        skipped_files_all = np.array([], dtype=object)

        if self._in_folder:
            folder_files = []
            for dir in self._directory:
                if os.path.isdir(dir):
                    pattern = str(dir) + "/**/*" if self._recursive else str(dir) + "/*"
                    files = glob(pattern, recursive=self._recursive)
                elif os.path.isfile(dir):
                    files = [dir]
                else:
                    continue

                files = [f for f in files if not os.path.isdir(f)]
                valid_files, skip_files = self._validate_files(files)
                if len(valid_files) > 0:
                    folder_files.append(valid_files)
                if len(skip_files) > 0:
                    skipped_files_all = np.concatenate((skipped_files_all, skip_files))

            if folder_files:
                valid_files_all = np.array(folder_files, dtype=object)
        else:
            all_files = []
            for dir in self._directory:
                if os.path.isdir(dir):
                    pattern = str(dir) + "/**/*" if self._recursive else str(dir) + "/*"
                    files = glob(pattern, recursive=self._recursive)
                    files = [f for f in files if not os.path.isdir(f)]
                    all_files.extend(files)
                elif os.path.isfile(dir):
                    all_files.append(dir)

            valid_files, skip_files = self._validate_files(all_files)
            valid_files_all = np.array(valid_files, dtype=object)
            if len(skip_files) > 0:
                skipped_files_all = np.concatenate((skipped_files_all, skip_files))

        return valid_files_all, skipped_files_all

    def _validate_files(self, directory):
        """Validate file types from a list of paths."""
        valid_files = np.array([
            os.path.normpath(file) for file in directory if not os.path.isdir(file)
        ])
        if self._limit_extensions:
            valid_files, skip_files = self._filter_extensions(valid_files)
        else:
            warnings.warn(
                'Parameter "limit_extensions" is set to False. '
                "difPy result accuracy can not be guaranteed for file formats "
                'not covered by "limit_extensions"',
                stacklevel=2,
            )
            skip_files = []
        return valid_files, skip_files

    def _filter_extensions(self, directory_files):
        """Filter files to keep only those with known image extensions."""
        extensions = []
        for file in directory_files:
            try:
                ext = file.rsplit(".", 1)[-1].lower()
                extensions.append(ext)
            except (IndexError, AttributeError):
                extensions.append("_")

        mask = np.array([ext in VALID_EXTENSIONS for ext in extensions])
        keep_files = directory_files[mask]
        skip_files = directory_files[~mask]
        return keep_files, skip_files

    def _build_image_dictionaries(self, valid_files):
        """Build dictionaries of image tensors and metadata."""
        tensor_dictionary = {}
        id_to_shape_dictionary = {}
        filename_dictionary = {}
        invalid_files = {}
        id_to_group_dictionary = {}
        group_to_id_dictionary = {}
        count = 0

        if self._in_folder:
            for j in range(len(valid_files)):
                group_id = f"group_{j}"
                group_img_ids = []
                with Pool(processes=self._processes) as pool:
                    file_nums = [
                        (i, valid_files[j][i]) for i in range(len(valid_files[j]))
                    ]
                    for output in pool.starmap(self._generate_tensor, file_nums):
                        if isinstance(output, dict):
                            invalid_files.update(output)
                        else:
                            img_id = count
                            filename_idx, tensor, shape = output
                            group_img_ids.append(img_id)
                            id_to_group_dictionary[img_id] = group_id
                            id_to_shape_dictionary[img_id] = shape
                            filename_dictionary[img_id] = valid_files[j][filename_idx]
                            tensor_dictionary[img_id] = tensor
                        count += 1
                group_to_id_dictionary[group_id] = group_img_ids
        else:
            with Pool(processes=self._processes) as pool:
                file_nums = [(i, valid_files[i]) for i in range(len(valid_files))]
                for output in pool.starmap(self._generate_tensor, file_nums):
                    if isinstance(output, dict):
                        invalid_files.update(output)
                    else:
                        img_id = count
                        filename_idx, tensor, shape = output
                        id_to_shape_dictionary[img_id] = shape
                        filename_dictionary[img_id] = valid_files[filename_idx]
                        tensor_dictionary[img_id] = tensor
                    count += 1

        return (
            tensor_dictionary,
            id_to_shape_dictionary,
            filename_dictionary,
            id_to_group_dictionary,
            group_to_id_dictionary,
            invalid_files,
        )

    def _generate_tensor(self, num: int, file: str) -> dict | tuple:
        """Generate a compressed tensor from an image file."""
        try:
            warnings.simplefilter("error", UserWarning)
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            img = Image.open(file)
            if img.getbands() != ("R", "G", "B"):
                img = img.convert("RGB")
            shape = np.asarray(img).shape
            img = img.resize((self._px_size, self._px_size), resample=Image.BICUBIC)
            img = np.asarray(img)
            return (num, img, shape)
        except Exception as e:
            print(
                f"Error {e.__class__.__name__} loading image #{num} : '{file}' -> {e}"
            )
            if e.__class__.__name__ == "UnidentifiedImageError":
                return {
                    str(
                        Path(file)
                    ): "UnidentifiedImageError: file could not be identified as image."
                }
            else:
                return {str(Path(file)): str(e)}
