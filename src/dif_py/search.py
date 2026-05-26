"""difPy Search: finds duplicate/similar images in a built repository."""

import os
import warnings
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tqdm import tqdm

from dif_py._internal import validators
from dif_py._internal.compare import (
    check_equality,
    compute_mse,
    sort_imgs_by_size,
)
from dif_py._internal.stats import search_stats as _generate_search_stats
from dif_py._internal.utils import initialize_multiprocessing

# ---------------------------------------------------------------------------
# Module-level worker state — populated once per worker process via the pool
# initializer.  Avoids pickling `self` (and its large tensor dictionaries)
# for every individual comparison task.
# ---------------------------------------------------------------------------

_worker_state: dict = {}


def _init_worker(
    tensor_dict, checksum_dict, norm_shape_dict, similarity, rotate, same_dim
):
    """Populate per-process globals once at pool start-up."""
    global _worker_state
    _worker_state = {
        "tensor": tensor_dict,
        "checksum": checksum_dict,
        "shape": norm_shape_dict,
        "similarity": similarity,
        "rotate": rotate,
        "same_dim": same_dim,
    }


def _find_matches_worker(ids: tuple):
    """Pairwise image comparison — module-level function for picklability."""
    id_A, id_B = ids
    s = _worker_state

    # Shape guard (fast dict lookup; pairs from same-dim generators already
    # satisfy this, but it acts as a safety net for the batch path).
    if s["same_dim"] and s["shape"][id_A] != s["shape"][id_B]:
        return None

    # Fast exact-match check via pre-computed checksum, then full equality.
    if s["checksum"][id_A] == s["checksum"][id_B] and check_equality(
        s["tensor"][id_A], s["tensor"][id_B]
    ):
        return (id_A, id_B, 0.0)

    # Skip MSE entirely when only exact duplicates are requested.
    if s["similarity"] == 0:
        return None

    mse = compute_mse(s["tensor"][id_A], s["tensor"][id_B], rotate=s["rotate"])
    if mse <= s["similarity"]:
        return (id_A, id_B, float(mse))
    return None


def _find_matches_batch_worker(group: list) -> list:
    """Batch image comparison for large datasets — module-level function."""
    s = _worker_state
    result = []
    id_A = group[0][0]
    tensor_A = s["tensor"][id_A]
    chk_A = s["checksum"][id_A]

    ids_B = np.asarray([pair[1] for pair in group])

    # Shape filter — eliminate candidates with incompatible dimensions.
    if s["same_dim"]:
        shape_A = s["shape"][id_A]
        mask = np.asarray([s["shape"][ib] == shape_A for ib in ids_B])
        ids_B = ids_B[mask]
        if ids_B.size == 0:
            return result

    # Exact-match detection via checksum equality.
    checksums_B = np.asarray([s["checksum"][ib] for ib in ids_B])
    eq_mask = checksums_B == chk_A
    for id_B in ids_B[np.where(eq_mask)[0]]:
        if check_equality(tensor_A, s["tensor"][id_B]):
            result.append((id_A, id_B, 0.0))

    if s["similarity"] > 0:
        neq_ids = ids_B[~eq_mask]
        if neq_ids.size > 0:
            mses = np.asarray([
                compute_mse(tensor_A, s["tensor"][ib], rotate=s["rotate"])
                for ib in neq_ids
            ])
            sim_indices = np.where(mses <= s["similarity"])[0]
            for idx in sim_indices:
                result.append((id_A, neq_ids[idx], float(mses[idx])))

    return result


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

    # Number of images above which streaming batch mode is used instead of
    # enumerating all pairs in memory upfront.
    _BATCH_THRESHOLD = 5_000

    def __init__(
        self,
        difpy_obj,
        similarity="duplicates",
        rotate=True,
        same_dim=True,
        show_progress=True,
        processes=None,
        chunksize=None,
    ):
        self._difpy_obj = difpy_obj
        self._similarity = validators.validate_similarity(similarity)
        self._rotate = validators.validate_rotate(rotate)
        self._same_dim = validators.validate_same_dim(same_dim, self._similarity)
        self._show_progress = validators.validate_show_progress(show_progress)
        if processes is None:
            processes = os.cpu_count() or 1
        self._processes = validators.validate_processes(processes)
        self._chunksize = validators.validate_chunksize(chunksize)
        self._in_folder = self._difpy_obj.stats["process"]["build"]["parameters"][
            "in_folder"
        ]

        initialize_multiprocessing()

        self.result, self.lower_quality, self.stats = self._run()

    # -------------------------------------------------------------------------
    # Initialisation helpers
    # -------------------------------------------------------------------------

    def _precompute(self):
        """Precompute per-image checksums and normalised shapes.

        Checksums (sum of all pixel values) provide O(1) rejection of
        non-duplicate pairs before the more expensive check_equality /
        compute_mse calls.  Normalised shapes allow shape-based pre-filtering
        without sorting inside hot worker loops.
        """
        self._checksums = {
            k: float(np.sum(v)) for k, v in self._difpy_obj._tensor_dictionary.items()
        }
        self._norm_shapes = {
            k: tuple(sorted(v))
            for k, v in self._difpy_obj._id_to_shape_dictionary.items()
        }

    def _make_pool(self):
        """Create a worker Pool pre-loaded with shared image data."""
        return Pool(
            processes=self._processes,
            initializer=_init_worker,
            initargs=(
                self._difpy_obj._tensor_dictionary,
                self._checksums,
                self._norm_shapes,
                self._similarity,
                self._rotate,
                self._same_dim,
            ),
        )

    def _run(self):
        """Execute the full search workflow."""
        start_time = datetime.now()

        # Precompute checksums and shapes once before spawning workers.
        self._precompute()

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
    # Pair generation helpers
    # -------------------------------------------------------------------------

    def _generate_pairs_same_dim(self, tensor_keys):
        """Yield only pairs whose images share the same normalised shape.

        Groups images by shape first, then emits combinations within each
        group — avoiding the O(N²) cross-shape checks inside workers.
        """
        shape_groups: dict = defaultdict(list)
        for key in tensor_keys:
            shape_groups[self._norm_shapes[key]].append(key)
        for group_keys in shape_groups.values():
            if len(group_keys) >= 2:
                yield from combinations(group_keys, 2)

    def _generate_pairs(self, tensor_keys):
        """Yield image ID pairs, respecting the same_dim setting."""
        if self._same_dim:
            yield from self._generate_pairs_same_dim(tensor_keys)
        else:
            yield from combinations(tensor_keys, 2)

    def _yield_comparison_group(self):
        """Yield batch groups for all images (large-dataset streaming)."""
        tensor_keys = self._difpy_obj._tensor_dictionary
        max_value = max(tensor_keys.keys())
        missing_ids = set(range(max_value + 1)) - set(tensor_keys.keys())
        for i in range(max_value):
            if i in missing_ids:
                continue
            group = [
                (i, j)
                for j in range(i + 1, max_value)
                if j not in missing_ids
                and (not self._same_dim or self._norm_shapes[i] == self._norm_shapes[j])
            ]
            if group:
                yield group

    def _yield_comparison_group_for_ids(self, ids):
        """Yield batch groups restricted to a specific set of image IDs.

        Used by the in-folder large-dataset path to avoid comparing images
        that belong to different folders.
        """
        sorted_ids = sorted(ids)
        for i, id_A in enumerate(sorted_ids):
            group = [
                (id_A, id_B)
                for id_B in sorted_ids[i + 1 :]
                if not self._same_dim
                or self._norm_shapes[id_A] == self._norm_shapes[id_B]
            ]
            if group:
                yield group

    # -------------------------------------------------------------------------
    # Union search (all directories merged)
    # -------------------------------------------------------------------------

    def _search_union(self):
        """Perform search across the union of all directories."""
        tensor_keys = list(self._difpy_obj._tensor_dictionary.keys())
        result_raw = []

        if len(tensor_keys) <= self._BATCH_THRESHOLD:
            id_combinations = list(self._generate_pairs(tensor_keys))
            total = len(id_combinations)
            if total == 0:
                return {}
            chunksize = max(1, min(1000, total // max(1, self._processes * 4)))
            with (
                self._make_pool() as pool,
                tqdm(
                    total=total,
                    desc="Searching images",
                    unit="pair",
                    disable=not self._show_progress,
                ) as pbar,
            ):
                for match in pool.imap_unordered(
                    _find_matches_worker, id_combinations, chunksize=chunksize
                ):
                    if match:
                        result_raw.append(match)
                    pbar.update()
        else:
            if self._chunksize is None:
                self._chunksize = max(1, round(1_000_000 / len(tensor_keys)))
            with (
                self._make_pool() as pool,
                tqdm(
                    total=len(tensor_keys) - 1,
                    desc="Searching images",
                    unit="batch",
                    disable=not self._show_progress,
                ) as pbar,
            ):
                for output in pool.imap_unordered(
                    _find_matches_batch_worker,
                    self._yield_comparison_group(),
                    self._chunksize,
                ):
                    result_raw.extend(output)
                    pbar.update()

        return self._group_result_union(result_raw)

    # -------------------------------------------------------------------------
    # In-folder search (directories searched separately)
    # -------------------------------------------------------------------------

    def _search_infolder(self):
        """Perform search within each directory in isolation."""
        result = []
        grouped_img_ids = list(self._difpy_obj._group_to_id_dictionary.values())

        with (
            self._make_pool() as pool,
            tqdm(
                total=len(grouped_img_ids),
                desc="Searching folders",
                unit="folder",
                disable=not self._show_progress,
            ) as pbar,
        ):
            for ids in grouped_img_ids:
                if len(ids) <= self._BATCH_THRESHOLD:
                    id_combinations = list(self._generate_pairs(ids))
                    if id_combinations:
                        chunksize = max(
                            1,
                            min(
                                500, len(id_combinations) // max(1, self._processes * 4)
                            ),
                        )
                        for match in pool.imap_unordered(
                            _find_matches_worker,
                            id_combinations,
                            chunksize=chunksize,
                        ):
                            if match:
                                result.append(match)
                else:
                    if self._chunksize is None:
                        self._chunksize = max(1, round(1_000_000 / len(ids)))
                    for output in pool.imap_unordered(
                        _find_matches_batch_worker,
                        self._yield_comparison_group_for_ids(ids),
                        self._chunksize,
                    ):
                        result.extend(output)
                pbar.update()

        return result

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
