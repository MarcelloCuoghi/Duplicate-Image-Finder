"""Command-line interface for difPy."""

import json
import os
from datetime import datetime
from typing import Annotated, Optional

import typer

from dif_py.build import build
from dif_py.search import search

app = typer.Typer(
    help="Find duplicate or similar images with difPy - "
    "https://github.com/elisemercury/Duplicate-Image-Finder",
    no_args_is_help=False,
)


@app.command()
def _command(
    directory: Annotated[
        Optional[list[str]],
        typer.Option(
            "-D",
            "--directory",
            help="Paths of directories to search. Default is working dir.",
        ),
    ] = None,
    output_directory: Annotated[
        Optional[str],
        typer.Option(
            "-Z",
            "--output-directory",
            help="Output directory for difPy result files. Default is working dir.",
        ),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive/--no-recursive", help="Search recursively within directories."
        ),
    ] = True,
    in_folder: Annotated[
        bool,
        typer.Option(
            "--in-folder/--no-in-folder",
            help="Search for matches in the union of directories.",
        ),
    ] = False,
    limit_extensions: Annotated[
        bool,
        typer.Option(
            "--limit-extensions/--no-limit-extensions",
            help="Limit search to known image file extensions.",
        ),
    ] = True,
    px_size: Annotated[
        int,
        typer.Option("-px", "--px-size", help="Compression size of images in pixels."),
    ] = 50,
    color_space: Annotated[
        str,
        typer.Option(
            "-cs",
            "--color-space",
            help="Color space for image comparison: 'gray' (grayscale, faster, recommended) or 'rgb'.",
        ),
    ] = "gray",
    similarity: Annotated[
        str,
        typer.Option(
            "-s",
            "--similarity",
            help="Similarity grade: 'duplicates', 'similar', or an integer MSE value.",
        ),
    ] = "duplicates",
    rotate: Annotated[
        bool,
        typer.Option("--rotate/--no-rotate", help="Rotate images during comparison."),
    ] = True,
    same_dim: Annotated[
        bool,
        typer.Option(
            "--same-dim/--no-same-dim",
            help="Only compare images with the same dimensions (width x height).",
        ),
    ] = True,
    move_to: Annotated[
        Optional[str],
        typer.Option(
            "-mv",
            "--move-to",
            help="Output directory for lower quality images among matches.",
        ),
    ] = None,
    delete: Annotated[
        bool,
        typer.Option(
            "--delete/--no-delete", help="Delete lower quality images among matches."
        ),
    ] = False,
    silent_del: Annotated[
        bool,
        typer.Option(
            "--silent-del/--no-silent-del",
            help="Suppress user confirmation when deleting images.",
        ),
    ] = False,
    show_progress: Annotated[
        bool,
        typer.Option(
            "--show-progress/--no-show-progress",
            help="Show the real-time progress of difPy.",
        ),
    ] = True,
    processes: Annotated[
        Optional[int],
        typer.Option(
            "-proc",
            "--processes",
            help="Number of worker processes for multiprocessing.",
        ),
    ] = None,
    chunksize: Annotated[
        Optional[int],
        typer.Option(
            "-ch",
            "--chunksize",
            help="Batch size for multiprocessing (relevant when dataset > 5k images).",
        ),
    ] = None,
) -> None:
    if directory is None:
        directory = [os.getcwd()]
    if processes is None:
        processes = os.cpu_count()

    # Convert similarity to int when a numeric string is passed (e.g. "-s 0")
    try:
        similarity_value: int | str = int(similarity)
    except ValueError:
        similarity_value = similarity

    # Mutual exclusivity check
    if move_to is not None and delete:
        raise typer.BadParameter(
            '"--move-to" and "--delete" are mutually exclusive. Please select one.'
        )

    # Output directory
    out_dir = output_directory or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # Run difPy
    dif = build(
        directory,
        recursive=recursive,
        in_folder=in_folder,
        limit_extensions=limit_extensions,
        px_size=px_size,
        color_space=color_space,
        show_progress=show_progress,
        processes=processes,
    )
    se = search(
        dif,
        similarity=similarity_value,
        rotate=rotate,
        same_dim=same_dim,
        processes=processes,
        chunksize=chunksize,
    )

    # Write output files
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    with open(os.path.join(out_dir, f"difPy_{timestamp}_results.json"), "w") as f:
        json.dump(se.result, f)
    with open(os.path.join(out_dir, f"difPy_{timestamp}_stats.json"), "w") as f:
        json.dump(se.stats, f)
    with open(os.path.join(out_dir, f"difPy_{timestamp}_lower_quality.txt"), "w") as f:
        f.write(f"{se.lower_quality}")

    if move_to is not None:
        se.move_to(move_to)

    if delete:
        se.delete(silent_del=silent_del)


def main() -> None:
    """Entry point for the difPy CLI."""
    app()


if __name__ == "__main__":
    main()
