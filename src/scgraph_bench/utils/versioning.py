"""Runtime version introspection for provenance manifests."""

from __future__ import annotations

import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def get_code_version() -> str | None:
    """Return '<sha>' or '<sha>+dirty' describing the current git source state.

    Untracked files do not count as dirty (artifact writes are expected at run time).
    Returns None when git is unavailable or the directory is not a repository.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        status = ""
    return f"{sha}+dirty" if status else sha


def get_torch_geometric_version() -> str | None:
    """Return the installed torch_geometric version, or None if unavailable."""
    try:
        import torch_geometric
    except ImportError:
        return None
    return str(torch_geometric.__version__)
