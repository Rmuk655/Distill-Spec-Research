"""Unit tests for post-merge checkpoint cleanup."""

from __future__ import annotations

import os

import pytest

from algorithms.distillspec_gbv.trainer import (
    _POST_MERGE_CLEANUP_MARKER,
    cleanup_training_artifacts,
)


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("x")


def test_cleanup_removes_milestones_latest_and_optimizer(tmp_path):
    root = tmp_path / "kl_tree-gsm8k"
    root.mkdir()
    _touch(str(root / "adapter_config.json"))

    for step in ("01000", "01200"):
        ms = root / f"ckpt_step_{step}"
        ms.mkdir()
        _touch(str(ms / "adapter_config.json"))
        _touch(str(ms / "optimizer.pt"))

    latest = root / "ckpt_latest"
    latest.mkdir()
    _touch(str(latest / "adapter_config.json"))
    _touch(str(latest / "optimizer.pt"))

    best = root / "ckpt_best"
    best.mkdir()
    _touch(str(best / "adapter_config.json"))
    _touch(str(best / "optimizer.pt"))

    removed = cleanup_training_artifacts(str(root), force=True)

    assert "ckpt_latest/" in removed
    assert any(r.startswith("ckpt_step_") for r in removed)
    assert any("optimizer.pt" in r.replace("\\", "/") for r in removed)
    assert (best / "adapter_config.json").is_file()
    assert not latest.exists()
    assert not any(root.glob("ckpt_step_*"))
    assert (root / _POST_MERGE_CLEANUP_MARKER).is_file()


def test_cleanup_aggressive_when_merged_exists(tmp_path):
    root = tmp_path / "kl-gsm8k"
    root.mkdir()
    _touch(str(root / "adapter_model.safetensors"))
    _touch(str(root / "adapter_config.json"))
    _touch(str(root / "training_state.json"))
    best = root / "ckpt_best"
    best.mkdir()
    _touch(str(best / "adapter_config.json"))
    merged = tmp_path / "kl-gsm8k_merged"
    merged.mkdir()
    _touch(str(merged / "config.json"))

    removed = cleanup_training_artifacts(str(root), force=True)

    assert "ckpt_best/" in removed
    assert "adapter_model.safetensors" in removed
    assert "adapter_config.json" in removed
    assert "training_state.json" in removed
    assert (merged / "config.json").is_file()


def test_cleanup_skips_when_marker_present(tmp_path):
    root = tmp_path / "kl-gsm8k"
    latest = root / "ckpt_latest"
    latest.mkdir(parents=True)
    _touch(str(latest / "optimizer.pt"))
    _touch(str(root / _POST_MERGE_CLEANUP_MARKER))

    removed = cleanup_training_artifacts(str(root))

    assert removed == []
    assert (latest / "optimizer.pt").is_file()


def test_cleanup_only_cli_flag(tmp_path, monkeypatch):
    root = tmp_path / "kl-gsm8k"
    latest = root / "ckpt_latest"
    latest.mkdir(parents=True)
    _touch(str(latest / "optimizer.pt"))

    argv = [
        "trainer.py",
        "--cleanup_only",
        "--adapter",
        str(root),
        "--draft",
        "distilgpt2",
    ]
    monkeypatch.setattr("sys.argv", argv)

    from algorithms.distillspec_gbv import trainer as tr

    tr.main()
    assert not latest.exists()
