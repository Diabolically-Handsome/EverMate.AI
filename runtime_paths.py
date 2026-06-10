"""Helpers for resource and writable path resolution.

These helpers keep EverMate working both:
- from the source tree
- from a frozen PyInstaller macOS app bundle
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "EverMate"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    return Path(__file__).resolve().parent


def bundle_root() -> Path:
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            return Path(meipass)
    return project_root()


def resource_path(*parts: str) -> Path:
    return bundle_root().joinpath(*parts)


def user_app_support_root(app_name: str = APP_NAME) -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / app_name
    if sys.platform == "win32":
        base = os.getenv("APPDATA") or str(home / "AppData" / "Roaming")
        return Path(base) / app_name
    return home / f".{app_name.lower()}"


def default_memory_dir() -> str:
    """Return the default memory root when MEMORY_DIR is not explicitly set."""

    if is_frozen():
        return str(user_app_support_root() / "memory")
    return "./memory"
