"""Utility helpers for difPy."""

from multiprocessing import current_process, freeze_support


def initialize_multiprocessing():
    """Initialize multiprocessing support if running in the main process."""
    if current_process().name == "MainProcess":
        freeze_support()


def progress_bar(count: int, total_count: int, task: str = "processing images"):
    """Display a progress bar in the console."""
    if count == total_count:
        print(f"difPy {task}: [{count / total_count:.0%}]")
    else:
        print(f"difPy {task}: [{count / total_count:.0%}]", end="\r")
