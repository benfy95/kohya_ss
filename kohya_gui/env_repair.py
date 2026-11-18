"""Remove broken package records left behind by old uv versions.

Upgrading an existing install with an old uv (seen with uv 0.7.21) can leave
the *old* version's ``<package>-<version>.dist-info`` folder behind, emptied
except for its ``licenses/`` subfolder and without a ``METADATA`` file. Python
still treats that folder as an installed package, and
``importlib.metadata.version()`` then returns ``None`` for it. diffusers reads
the transformers version this way at import time, so training crashed with::

    TypeError: expected string or bytes-like object, got 'NoneType'

This module is imported by ``kohya_gui.py`` before anything else and removes
exactly that residue. It is deliberately narrow:

* only inside a virtual environment (``sys.prefix != sys.base_prefix``), never
  a system Python;
* only ``*.dist-info`` folders that have **no** ``METADATA`` and contain nothing
  but an optional ``licenses`` folder (a valid record always has ``METADATA``).

Standard library only, so it can run before any third-party import.
"""

import os
import shutil
import sys
import sysconfig
from typing import Iterable, List, Optional

_ALLOWED_LEFTOVERS = {"licenses"}


def _site_packages_dirs() -> List[str]:
    paths = sysconfig.get_paths()
    dirs = []
    for key in ("purelib", "platlib"):
        path = paths.get(key)
        if path and os.path.isdir(path) and path not in dirs:
            dirs.append(path)
    return dirs


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def is_orphaned_dist_info(path: str) -> bool:
    """True for a ``*.dist-info`` folder with no METADATA that holds nothing
    but leftover license files."""
    if not path.endswith(".dist-info") or not os.path.isdir(path):
        return False
    try:
        entries = set(os.listdir(path))
    except OSError:
        return False
    if "METADATA" in entries:
        return False
    return entries <= _ALLOWED_LEFTOVERS


def find_orphaned_dist_info(site_dirs: Iterable[str]) -> List[str]:
    found = []
    for site_dir in site_dirs:
        try:
            names = sorted(os.listdir(site_dir))
        except OSError:
            continue
        for name in names:
            path = os.path.join(site_dir, name)
            if is_orphaned_dist_info(path):
                found.append(path)
    return found


def remove_orphaned_dist_info(
    site_dirs: Optional[Iterable[str]] = None, require_venv: bool = True
) -> List[str]:
    """Delete orphaned package records; returns the folders removed."""
    if require_venv and not in_virtualenv():
        return []
    if site_dirs is None:
        site_dirs = _site_packages_dirs()
    removed = []
    for path in find_orphaned_dist_info(site_dirs):
        try:
            shutil.rmtree(path)
            removed.append(path)
        except OSError as e:
            print(f"[kohya_ss] Could not remove broken package record {path}: {e}")
    return removed


def repair_environment() -> List[str]:
    """Run at GUI startup: remove the residue and tell the user why."""
    removed = remove_orphaned_dist_info()
    if removed:
        names = ", ".join(os.path.basename(p) for p in removed)
        print(
            f"[kohya_ss] Removed {len(removed)} broken package record(s) left by an "
            f"interrupted or old-uv upgrade: {names}"
        )
        print(
            "[kohya_ss] These made Python report no version for packages such as "
            "transformers. To avoid this in future updates, update uv: "
            "'uv self update' (or 'winget upgrade astral-sh.uv' if installed via winget)."
        )
    return removed
