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
    base = os.getenv("XDG_DATA_HOME") or str(home / ".local" / "share")
    return Path(base) / app_name


def default_memory_dir() -> str:
    """Return the default memory root when MEMORY_DIR is not explicitly set.

    Always a fixed per-user location: a CWD-relative default silently splits
    the user's memory across directories depending on where the app was
    launched from.
    """

    return str(user_app_support_root() / "memory")


def migrate_legacy_memory_dir(target: str) -> bool:
    """One-time migration from the old CWD-relative ./memory layout.

    If `target` has no index yet but ./memory does, copy its contents over so
    users upgrading from older source checkouts keep their data. Returns True
    if a migration happened.
    """

    import shutil

    target_path = Path(target)
    legacy = Path.cwd() / "memory"
    try:
        if target_path.resolve() == legacy.resolve():
            return False
    except OSError:
        return False
    if (target_path / "index.sqlite").exists():
        return False
    if not (legacy / "index.sqlite").exists():
        return False
    target_path.mkdir(parents=True, exist_ok=True)
    for item in legacy.iterdir():
        dest = target_path / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    return True
