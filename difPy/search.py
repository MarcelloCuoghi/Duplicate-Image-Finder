"""difPy Search: finds duplicate/similar images in a built repository."""

import os
import warnings
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ._internal import validators
from ._internal.compare import (
    check_equality,
    compare_shape,
    compute_mse,
    sort_imgs_by_size,
)
from ._internal.stats import search_stats as _generate_search_stats
from ._internal.utils import initialize_multiprocessing, progress_bar


class search:
    """Search for duplicate/similar images in a difPy image repository.

    Parameters
    ----------
    difpy_obj : difPy.build
        A difPy object containing the built image repository.
    similarity : 'duplicates', 'similar', or float, optional
        Image comparison similarity threshold (MSE). Default is 'duplicates' (0).
    rotate : bool, optional
        Rotate images during comparison (default True).
    same_dim : bool, optional
        Only compare images with same dimensions (default True).
    show_progress : bool, optional
        Show progress bar in console (default True).
    processes : int, optional
        Number of worker processes for multiprocessing.
    chunksize : int, optional
        Batch size for multiprocessing with large datasets (> 5k images).
    """

    def __init__(
        self,
        difpy_obj,
        similarity="duplicates",
        rotate=True,
        same_dim=True,
        show_progress=True,
        processes=os.cpu_count(),
        chunksize=None,
    ):
        self._difpy_obj = difpy_obj
        self._similarity = validators.validate_similarity(similarity)
        self._rotate = validators.validate_rotate(rotate)
        self._same_dim = validators.validate_same_dim(same_dim, self._similarity)
        self._show_progress = validators.validate_show_progress(show_progress)
        self._processes = validators.validate_processes(processes)
        self._chunksize = validators.validate_chunksize(chunksize)
        self._in_folder = self._difpy_obj.stats["process"]["build"]["parameters"][
            "in_folder"
        ]

        initialize_multiprocessing()

        if self._show_progress:
            print("Initializing search...", end="\r")
        self.result, self.lower_quality, self.stats = self._run()

    def _run(self):
        """Execute the full search workflow."""
        start_time = datetime.now()

        if self._in_folder:
            result = self._search_infolder()
            result = self._format_result_infolder(result)
            lower_quality, duplicate_count, similar_count = (
                self._search_metadata_infolder(result)
            )
        else:
            result = self._search_union()
            result = self._format_result_union(result)
            lower_quality, duplicate_count, similar_count = self._search_metadata_union(
                result
            )

        end_time = datetime.now()

        stats = _generate_search_stats(
            build_stats=self._difpy_obj.stats,
            start_time=start_time,
            end_time=end_time,
            similarity=self._similarity,
            rotate=self._rotate,
            same_dim=self._same_dim,
            processes=self._processes,
            files_searched=len(self._difpy_obj._tensor_dictionary),
            duplicate_count=duplicate_count,
            similar_count=similar_count,
            chunksize=self._chunksize,
        )

        return result, lower_quality, stats

    # -------------------------------------------------------------------------
    # Union search (all directories merged)
    # -------------------------------------------------------------------------

    def _search_union(self):
        """Perform search in the union of all directories."""
        result_raw = []
        self._count = 0
        tensor_keys = list(self._difpy_obj._tensor_dictionary.keys())

        if len(tensor_keys) <= 5000:
            id_combinations = list(combinations(tensor_keys, 2))
            with Pool(processes=self._processes) as pool:
                output = pool.map(self._find_matches, id_combinations)
            for i in output:
                if i:
                    result_raw.append(i)
            self._count += 1
            if self._show_progress:
                progress_bar(self._count, 1, task="searching files")
        else:
            if self._chunksize is None:
                self._chunksize = max(1, round(1000000 / len(tensor_keys)))
            with Pool(processes=self._processes) as pool:
                for output in pool.imap_unordered(
                    self._find_matches_batch,
                    self._yield_comparison_group(),
                    self._chunksize,
                ):
                    if len(output) > 0:
                        result_raw = result_raw + output
                    self._count += 1
                    if self._show_progress:
                        progress_bar(
                            self._count,
                            len(tensor_keys) - 1,
                            task="searching files",
                        )

        return self._group_result_union(result_raw)

    # -------------------------------------------------------------------------
    # In-folder search (directories searched separately)
    # -------------------------------------------------------------------------

    def _search_infolder(self):
        """Perform search in isolated/separate directories."""
        result = []
        grouped_img_ids = list(self._difpy_obj._group_to_id_dictionary.values())
        self._count = 0

        with Pool(processes=self._processes) as pool:
            for ids in grouped_img_ids:
                if len(ids) <= 5000:
                    id_combinations = list(combinations(ids, 2))
                    output = pool.map(self._find_matches, id_combinations)
                    for i in output:
                        if i:
                            result.append(i)
                    self._count += 1
                else:
                    if self._chunksize is None:
                        self._chunksize = max(1, round(1000000 / len(ids)))
                    for output in pool.imap_unordered(
                        self._find_matches_batch,
                        self._yield_comparison_group(),
                        self._chunksize,
                    ):
                        if len(output) > 0:
                            result = result + output
                    self._count += 1
                if self._show_progress:
                    progress_bar(
                        self._count, len(grouped_img_ids), task="searching files"
                    )

        return result

    # -------------------------------------------------------------------------
    # Match finding
    # -------------------------------------------------------------------------

    def _find_matches(self, ids):
        """Search for a match between two images."""
        id_A, id_B = ids
        tensor_A = self._difpy_obj._tensor_dictionary[id_A]
        tensor_B = self._difpy_obj._tensor_dictionary[id_B]
        shape_A = self._difpy_obj._id_to_shape_dictionary[id_A]
        shape_B = self._difpy_obj._id_to_shape_dictionary[id_B]

        if self._same_dim:
            if not compare_shape(shape_A, shape_B):
                return False
            if check_equality(tensor_A, tensor_B):
                return (id_A, id_B, 0.0)
            mse = compute_mse(tensor_A, tensor_B, rotate=self._rotate)
            if mse <= self._similarity:
                return (id_A, id_B, mse)
        else:
            if check_equality(tensor_A, tensor_B):
                return (id_A, id_B, 0.0)
            mse = compute_mse(tensor_A, tensor_B, rotate=self._rotate)
            if mse <= self._similarity:
                return (id_A, id_B, mse)

    def _find_matches_batch(self, ids):
        """Search for matches among images in batches (for large datasets)."""
        result = []
        id_A = ids[0][0]
        tensor_A = self._difpy_obj._tensor_dictionary[id_A]
        ids_B_list = np.asarray([x[1] for x in ids])
        tensor_B_list = np.asarray([
            self._difpy_obj._tensor_dictionary[x[1]] for x in ids
        ])

        if self._same_dim:
            shape_A_list = [
                sorted(self._difpy_obj._id_to_shape_dictionary[id_A])
            ] * len(ids)
            shape_B_list = [
                sorted(self._difpy_obj._id_to_shape_dictionary[id_B])
                for id_B in ids_B_list
            ]
            same_shape = np.equal(shape_A_list, shape_B_list).all(axis=1)
            shape_index = np.where(same_shape)
            if len(shape_index) > 0:
                ids_B_list = ids_B_list[shape_index]
                tensor_B_list = tensor_B_list[shape_index]

        # Check for exact matches
        sum_B_list = [np.sum(tensor_B) for tensor_B in tensor_B_list]
        sum_A_list = [np.sum(tensor_A)] * len(sum_B_list)
        equals = np.equal(sum_A_list, sum_B_list)

        dupl_index = np.where(equals)
        non_dupl_index = np.where(equals == False)

        if len(dupl_index) > 0:
            for id_B in ids_B_list[dupl_index]:
                result.append((id_A, id_B, 0))
            tensor_B_list = tensor_B_list[non_dupl_index]
            ids_B_list = ids_B_list[non_dupl_index]

        if self._similarity > 0:
            mses = np.asarray([
                compute_mse(tensor_A, tensor_B, rotate=self._rotate)
                for tensor_B in tensor_B_list
            ])
            mse_index_sim = np.where(mses <= self._similarity)
            if len(mse_index_sim) > 0:
                for i, id_B in enumerate(ids_B_list[mse_index_sim]):
                    result.append((id_A, id_B, mses[i]))

        return result

    def _yield_comparison_group(self):
        """Yield lists of image ID pairs ready for batch comparison."""
        max_value = max(self._difpy_obj._tensor_dictionary.keys())
        missing_ids = set(range(max_value + 1)) - set(
            self._difpy_obj._tensor_dictionary.keys()
        )
        for i in range(max_value):
            if i in missing_ids:
                continue
            group = [(i, j) for j in range(i + 1, max_value) if j not in missing_ids]
            if group:
                yield group

    # -------------------------------------------------------------------------
    # Result formatting
    # -------------------------------------------------------------------------

    def _group_result_union(self, tuple_list):
        """Group raw match tuples into a result dictionary."""
        result = defaultdict(list)
        already_added = set()
        for k, *v in tuple_list:
            if v[0] not in already_added:
                result[k].append(v)
                already_added.add(v[0])
        return dict(result)

    def _group_result_infolder(self, tuple_list, folder_paths):
        """Group raw match tuples using folder paths."""
        result = defaultdict(dict)
        already_added = set()
        for k, *v in tuple_list:
            k_group = self._difpy_obj._id_to_group_dictionary[k]
            folder_path = folder_paths[k_group]
            if v[0] not in already_added:
                if k not in result[folder_path]:
                    result[folder_path][k] = []
                result[folder_path][k].append(v)
                already_added.add(v[0])
        return dict(result)

    def _format_result_union(self, result):
        """Replace image IDs with filenames in the result dictionary."""
        updated_result = {}
        for key, value in result.items():
            new_key = self._difpy_obj._filename_dictionary.get(key, key)
            new_value = [
                [self._difpy_obj._filename_dictionary.get(inner[0], inner[0]), inner[1]]
                for inner in value
            ]
            updated_result[new_key] = new_value
        return updated_result

    def _format_result_infolder(self, result):
        """Replace group/image IDs with file paths in the in-folder result."""
        folder_paths = self._get_paths_from_groups()
        result = self._group_result_infolder(result, folder_paths)

        updated_result = {}
        for group_id in result:
            for key, value in result[group_id].items():
                new_key = self._difpy_obj._filename_dictionary.get(key, key)
                new_value = [
                    [
                        self._difpy_obj._filename_dictionary.get(inner[0], inner[0]),
                        inner[1],
                    ]
                    for inner in value
                ]
                if group_id not in updated_result:
                    updated_result[group_id] = {}
                updated_result[group_id][new_key] = new_value
        return updated_result

    def _get_paths_from_groups(self):
        """Map group IDs to their parent folder paths."""
        folder_paths = {}
        for group_id, img_ids in self._difpy_obj._group_to_id_dictionary.items():
            if img_ids:
                first_img_path = self._difpy_obj._filename_dictionary[img_ids[0]]
                folder_paths[group_id] = os.path.dirname(first_img_path)
        return folder_paths

    # -------------------------------------------------------------------------
    # Search metadata / quality comparison
    # -------------------------------------------------------------------------

    def _search_metadata_union(self, result):
        """Compare image qualities and compute duplicate/similar counts."""
        duplicate_count, similar_count = 0, 0
        lower_quality = np.array([])

        for img, matches in result.items():
            match_group = [img]
            for img_match in matches:
                match_group.append(img_match[0])
                if self._similarity == 0 or img_match[1] == 0:
                    duplicate_count += 1
                else:
                    similar_count += 1
            match_group = sort_imgs_by_size(match_group)
            lower_quality = np.concatenate((lower_quality, match_group[1:]), axis=None)

        return list(set(lower_quality)), duplicate_count, similar_count

    def _search_metadata_infolder(self, result):
        """Compare image qualities for in-folder search results."""
        duplicate_count, similar_count = 0, 0
        lower_quality = np.array([])

        for group_id in result:
            for img, matches in result[group_id].items():
                match_group = [img]
                for img_match in matches:
                    match_group.append(img_match[0])
                    if self._similarity == 0 or img_match[1] == 0:
                        duplicate_count += 1
                    else:
                        similar_count += 1
                match_group = sort_imgs_by_size(match_group)
                lower_quality = np.concatenate(
                    (lower_quality, match_group[1:]), axis=None
                )

        return list(set(lower_quality)), duplicate_count, similar_count

    # -------------------------------------------------------------------------
    # Post-search actions
    # -------------------------------------------------------------------------

    def _delete_files(self):
        """Delete lower quality files and return the count of deletions."""
        deleted_files = 0
        for file in self.lower_quality:
            try:
                os.remove(file)
                deleted_files += 1
            except OSError:
                warnings.warn(f"Could not delete file: {file}", stacklevel=2)
        return deleted_files

    def move_to(self, destination_path):
        """Move lower quality images to a destination directory.

        Parameters
        ----------
        destination_path : str
            Path to move the lower_quality files to.
        """
        destination_path = validators.validate_move_to(destination_path)
        new_lower_quality = []
        for file in self.lower_quality:
            try:
                _head, tail = os.path.split(file)
                os.replace(file, os.path.join(destination_path, tail))
                new_lower_quality.append(
                    str(Path(os.path.join(destination_path, tail)))
                )
            except OSError:
                warnings.warn(f"Could not move file: {file}", stacklevel=2)
        print(
            f'Moved {len(self.lower_quality)} files(s) to "{Path(destination_path)!s}"'
        )
        self.lower_quality = new_lower_quality

    def delete(self, silent_del=False):
        """Delete the lower quality images found after search.

        Parameters
        ----------
        silent_del : bool, optional
            Skip user confirmation (default False).
        """
        silent_del = validators.validate_silent_del(silent_del)

        if len(self.lower_quality) > 0:
            if not silent_del:
                usr = input(
                    "Are you sure you want to delete all lower quality matched images? \n"
                    "! This cannot be undone. (y/n)"
                )
                if str(usr).lower() == "y":
                    deleted_files = self._delete_files()
                else:
                    print("Deletion canceled.")
                    return
            else:
                deleted_files = self._delete_files()

        print(f"Deleted {deleted_files} file(s)")
