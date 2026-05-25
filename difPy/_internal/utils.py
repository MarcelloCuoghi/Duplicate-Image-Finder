"""Utility helpers for difPy."""

import argparse
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


def convert_str_to_int(x):
    """Attempt to convert a string to int; return original value on failure."""
    try:
        return int(x)
    except (ValueError, TypeError):
        return x


def strtobool(v):
    """Convert a string representation of truth to True/False for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")
