"""Tests for runtime_paths: per-user dirs and legacy memory migration.

These tests never touch the real user app-support dir: default_memory_dir is
only *computed* (string checks), and migrations run inside tmp_path.
"""

from __future__ import annotations

import os
import sys

from runtime_paths import (
    APP_NAME,
    default_memory_dir,
    migrate_legacy_memory_dir,
    user_app_support_root,
)


class TestUserAppSupportRoot:
    def test_is_absolute_and_named_after_app(self):
        root = user_app_support_root()
        assert root.is_absolute()
        assert root.name == APP_NAME

    def test_custom_app_name(self):
        assert user_app_support_root("OtherApp").name == "OtherApp"

    def test_platform_specific_location(self):
        root = str(user_app_support_root())
        if sys.platform == "darwin":
            assert os.path.join("Library", "Application Support") in root


class TestDefaultMemoryDir:
    def test_always_per_user_never_cwd_relative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d1 = default_memory_dir()
        assert os.path.isabs(d1)
        assert not d1.startswith(str(tmp_path))
        assert d1 == str(user_app_support_root() / "memory")

        # changing CWD must not change the result
        other = tmp_path / "elsewhere"
        other.mkdir()
        monkeypatch.chdir(other)
        assert default_memory_dir() == d1


class TestMigrateLegacyMemoryDir:
    def _make_legacy(self, root):
        legacy = root / "memory"
        (legacy / "chunks").mkdir(parents=True)
        (legacy / "index.sqlite").write_bytes(b"fake-db")
        (legacy / "chunks" / "00000001.txt").write_text("旧数据", encoding="utf-8")
        return legacy

    def test_copies_legacy_layout_into_empty_target(self, tmp_path, monkeypatch):
        cwd = tmp_path / "checkout"
        cwd.mkdir()
        self._make_legacy(cwd)
        monkeypatch.chdir(cwd)

        target = tmp_path / "target"
        assert migrate_legacy_memory_dir(str(target)) is True
        assert (target / "index.sqlite").read_bytes() == b"fake-db"
        assert (target / "chunks" / "00000001.txt").read_text(
            encoding="utf-8"
        ) == "旧数据"

    def test_skips_when_target_already_has_index(self, tmp_path, monkeypatch):
        cwd = tmp_path / "checkout"
        cwd.mkdir()
        self._make_legacy(cwd)
        monkeypatch.chdir(cwd)

        target = tmp_path / "target"
        target.mkdir()
        (target / "index.sqlite").write_bytes(b"existing")
        assert migrate_legacy_memory_dir(str(target)) is False
        assert (target / "index.sqlite").read_bytes() == b"existing"

    def test_skips_when_no_legacy_dir(self, tmp_path, monkeypatch):
        cwd = tmp_path / "empty-checkout"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        assert migrate_legacy_memory_dir(str(tmp_path / "target")) is False

    def test_skips_when_target_is_the_legacy_dir(self, tmp_path, monkeypatch):
        cwd = tmp_path / "checkout"
        cwd.mkdir()
        legacy = self._make_legacy(cwd)
        monkeypatch.chdir(cwd)
        assert migrate_legacy_memory_dir(str(legacy)) is False
