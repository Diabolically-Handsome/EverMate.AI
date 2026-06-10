"""Tests for models_config: boundary-aware model resolution, local-only policy."""

from __future__ import annotations

from models_config import is_local_model, resolve_installed_model


class TestIsLocalModel:
    def test_plain_models_are_local(self):
        assert is_local_model("gpt-oss:120b")
        assert is_local_model("deepseek-r1:8b")

    def test_cloud_models_are_not_local(self):
        assert not is_local_model("qwen3-cloud:480b")
        assert not is_local_model("deepseek-r1:8b-CLOUD")  # case-insensitive


class TestResolveInstalledModel:
    KEY = "deepseek_qwen3_8b"

    def test_exact_match(self):
        name, cands = resolve_installed_model(["deepseek-r1:8b"], self.KEY)
        assert name == "deepseek-r1:8b"
        assert "deepseek-r1:8b" in cands

    def test_quantization_suffix_matches(self):
        name, _ = resolve_installed_model(
            ["deepseek-r1:8b-0528-qwen3-q4_K_M"], self.KEY
        )
        assert name == "deepseek-r1:8b-0528-qwen3-q4_K_M"

    def test_boundary_prevents_80b_matching_8b(self):
        name, cands = resolve_installed_model(["deepseek-r1:80b"], self.KEY)
        assert name == ""
        assert cands  # candidates still reported for the error message

    def test_cloud_models_never_resolved(self):
        name, _ = resolve_installed_model(["deepseek-r1:8b-cloud"], self.KEY)
        assert name == ""

    def test_unknown_key(self):
        assert resolve_installed_model(["deepseek-r1:8b"], "no_such_key") == ("", [])

    def test_no_models_installed(self):
        name, cands = resolve_installed_model([], self.KEY)
        assert name == ""
        assert cands

    def test_candidate_priority_order(self):
        # qwen3_30b_a3b lists "qwen3:30b-a3b" before "qwen3:30b": when both
        # are installed, the first candidate wins.
        name, _ = resolve_installed_model(
            ["qwen3:30b", "qwen3:30b-a3b"], "qwen3_30b_a3b"
        )
        assert name == "qwen3:30b-a3b"

    def test_falls_back_to_later_candidate(self):
        # Only the second candidate is installed (plain "qwen3:30b" is not a
        # boundary-variant of "qwen3:30b-a3b", so it cannot match the first).
        name, _ = resolve_installed_model(["qwen3:30b"], "qwen3_30b_a3b")
        assert name == "qwen3:30b"
