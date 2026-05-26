"""Utility helpers for difPy."""

from multiprocessing import current_process, freeze_support


def initialize_multiprocessing():
    """Initialize multiprocessing support if running in the main process."""
    if current_process().name == "MainProcess":
        freeze_support()
