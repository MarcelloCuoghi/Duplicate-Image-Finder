"""Command-line interface for difPy."""

import argparse
import json
import os
from datetime import datetime

from ._internal.utils import convert_str_to_int, strtobool
from .build import build
from .search import search


def main():
    """Entry point for the difPy CLI."""
    parser = argparse.ArgumentParser(
        description="Find duplicate or similar images with difPy - "
        "https://github.com/elisemercury/Duplicate-Image-Finder"
    )
    parser.add_argument(
        "-D",
        "--directory",
        type=str,
        nargs="+",
        help="Paths of the directories to be searched. Default is working dir.",
        required=False,
        default=[os.getcwd()],
    )
    parser.add_argument(
        "-Z",
        "--output_directory",
        type=str,
        help="Output directory path for the difPy result files. Default is working dir.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-r",
        "--recursive",
        type=lambda x: bool(strtobool(x)),
        help="Search recursively within the directories.",
        required=False,
        choices=[True, False],
        default=True,
    )
    parser.add_argument(
        "-i",
        "--in_folder",
        type=lambda x: bool(strtobool(x)),
        help="Search for matches in the union of directories.",
        required=False,
        choices=[True, False],
        default=False,
    )
    parser.add_argument(
        "-le",
        "--limit_extensions",
        type=lambda x: bool(strtobool(x)),
        help="Limit search to known image file extensions.",
        required=False,
        choices=[True, False],
        default=True,
    )
    parser.add_argument(
        "-px",
        "--px_size",
        type=int,
        help="Compression size of images in pixels.",
        required=False,
        default=50,
    )
    parser.add_argument(
        "-s",
        "--similarity",
        type=convert_str_to_int,
        help="Similarity grade (mse).",
        required=False,
        default="duplicates",
    )
    parser.add_argument(
        "-ro",
        "--rotate",
        type=lambda x: bool(strtobool(x)),
        help="Rotate images during comparison process.",
        required=False,
        choices=[True, False],
        default=True,
    )
    parser.add_argument(
        "-dim",
        "--same_dim",
        type=lambda x: bool(strtobool(x)),
        help="Only compare images having the same dimensions (width x height).",
        required=False,
        choices=[True, False],
        default=True,
    )
    parser.add_argument(
        "-mv",
        "--move_to",
        type=str,
        help="Output directory path of lower quality images among matches.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--delete",
        type=lambda x: bool(strtobool(x)),
        help="Delete lower quality images among matches.",
        required=False,
        choices=[True, False],
        default=False,
    )
    parser.add_argument(
        "-sd",
        "--silent_del",
        type=lambda x: bool(strtobool(x)),
        help="Suppress the user confirmation when deleting images.",
        required=False,
        choices=[True, False],
        default=False,
    )
    parser.add_argument(
        "-p",
        "--show_progress",
        type=lambda x: bool(strtobool(x)),
        help="Show the real-time progress of difPy.",
        required=False,
        choices=[True, False],
        default=True,
    )
    parser.add_argument(
        "-proc",
        "--processes",
        type=convert_str_to_int,
        help="Number of worker processes for multiprocessing.",
        required=False,
        default=os.cpu_count(),
    )
    parser.add_argument(
        "-ch",
        "--chunksize",
        type=convert_str_to_int,
        help="Only relevant when dataset > 5k images. Sets the batch size for multiprocessing.",
        required=False,
        default=None,
    )
    args = parser.parse_args()

    # Output directory
    if args.output_directory is not None:
        dir = args.output_directory
        if not os.path.exists(dir):
            os.makedirs(dir)
    else:
        dir = os.getcwd()

    # Mutual exclusivity check
    if args.move_to is not None and args.delete is not False:
        raise SystemExit(
            '"move_to" and "delete" parameters are mutually exclusive. Please select one.'
        )

    # Run difPy
    dif = build(
        args.directory,
        recursive=args.recursive,
        in_folder=args.in_folder,
        limit_extensions=args.limit_extensions,
        px_size=args.px_size,
        show_progress=args.show_progress,
        processes=args.processes,
    )
    se = search(
        dif,
        similarity=args.similarity,
        rotate=args.rotate,
        same_dim=args.same_dim,
        processes=args.processes,
        chunksize=args.chunksize,
    )

    # Output files
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_file = f"difPy_{timestamp}_results.json"
    lq_file = f"difPy_{timestamp}_lower_quality.txt"
    stats_file = f"difPy_{timestamp}_stats.json"

    with open(os.path.join(dir, result_file), "w") as file:
        json.dump(se.result, file)
    with open(os.path.join(dir, stats_file), "w") as file:
        json.dump(se.stats, file)
    with open(os.path.join(dir, lq_file), "w") as file:
        file.write(f"{se.lower_quality}")

    if args.move_to is not None:
        se.move_to(args.move_to)

    if args.delete:
        se.delete(silent_del=args.silent_del)

    print(f"\n{result_file}\n{lq_file}\n{stats_file}\n\nsaved in '{dir}'.")


if __name__ == "__main__":
    main()
