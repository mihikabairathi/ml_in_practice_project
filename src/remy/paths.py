"""Shared paths for the repository's editable installation."""

from pathlib import Path

# src/remy/paths.py -> repository root, independent of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
