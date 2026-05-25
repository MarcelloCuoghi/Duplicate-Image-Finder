"""
difPy - Python package for finding duplicate and similar images.
2024 Elise Landman
https://github.com/elisemercury/Duplicate-Image-Finder
"""

from .build import build  # noqa: F401
from .search import search  # noqa: F401

if __name__ == "__main__":
    from .cli import main

    main()
