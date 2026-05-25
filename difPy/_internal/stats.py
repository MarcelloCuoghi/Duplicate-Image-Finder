"""Statistics generation for difPy build and search processes."""

from pathlib import Path

import numpy as np


def build_stats(
    *,
    total_files: int,
    invalid_files: dict,
    skipped_files,
    directory,
    start_time,
    end_time,
    recursive: bool,
    in_folder: bool,
    limit_extensions: bool,
    px_size: int,
    processes: int,
) -> dict:
    """Generate statistics for the Build process."""
    for file in skipped_files:
        invalid_files[str(Path(file))] = "Unsupported file type"

    return {
        "directory": directory,
        "total_files": total_files + len(invalid_files),
        "invalid_files": {
            "count": len(invalid_files),
            "logs": invalid_files,
        },
        "process": {
            "build": {
                "duration": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "seconds_elapsed": np.round(
                        (end_time - start_time).total_seconds(), 4
                    ),
                },
                "parameters": {
                    "recursive": recursive,
                    "in_folder": in_folder,
                    "limit_extensions": limit_extensions,
                    "px_size": px_size,
                    "processes": processes,
                },
            }
        },
    }


def search_stats(
    *,
    build_stats: dict,
    start_time,
    end_time,
    similarity: float,
    rotate: bool,
    same_dim: bool,
    processes: int,
    files_searched: int,
    duplicate_count: int,
    similar_count: int,
    chunksize: int | None,
) -> dict:
    """Generate statistics for the Search process."""
    search_section = {
        "search": {
            "duration": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "seconds_elapsed": np.round((end_time - start_time).total_seconds(), 4),
            },
            "parameters": {
                "similarity_mse": similarity,
                "rotate": rotate,
                "same_dim": same_dim,
                "processes": processes,
                "chunksize": chunksize,
            },
            "files_searched": files_searched,
            "matches_found": {
                "duplicates": duplicate_count,
                "similar": similar_count,
            },
        }
    }
    build_stats["process"].update(search_section)
    return build_stats
