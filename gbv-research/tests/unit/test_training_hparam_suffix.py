"""Unit tests for checkpoint training-hyperparam fingerprint builder."""

from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.join(_HERE, "..", "..", "orchestration")
sys.path.insert(0, _ORCH)

import experiment as _exp


class TestTrainingFingerprint:
    def test_offline_deterministic(self):
        h = {}
        a, _ = _exp._training_fingerprint(
            h, "offline", effective_steps=1000, online_steps=500, online_max_tok=64,
        )
        b, _ = _exp._training_fingerprint(
            h, "offline", effective_steps=1000, online_steps=500, online_max_tok=64,
        )
        assert a == b
        assert len(a) == 8
        assert all(c in "0123456789abcdef" for c in a)

    def test_same_input_same_output(self):
        h = {"tree_K": 6, "tree_L": 10, "lr": 1e-4, "lora_r": 16, "teacher_temp": 0.9}
        a, canon_a = _exp._training_fingerprint(
            h, "tree", effective_steps=4000, online_steps=500, online_max_tok=64,
        )
        b, canon_b = _exp._training_fingerprint(
            h, "tree", effective_steps=4000, online_steps=500, online_max_tok=64,
        )
        assert a == b
        assert canon_a == canon_b
        assert canon_a["tree_K"] == 6
        assert canon_a["steps"] == 4000

    def test_offline_excludes_tree_keys(self):
        h = {"tree_K": 99, "tree_L": 99, "lr": 1e-4}
        offline_id, offline = _exp._training_fingerprint(
            h, "offline", effective_steps=1000, online_steps=500, online_max_tok=64,
        )
        h_flat = {"lr": 1e-4}
        offline_id2, _ = _exp._training_fingerprint(
            h_flat, "offline", effective_steps=1000, online_steps=500, online_max_tok=64,
        )
        assert offline_id == offline_id2
        assert "tree_K" not in offline

    def test_tree_profile_includes_tree_k_l(self):
        h = {"tree_K": 6, "tree_L": 10, "lr": 1e-4, "lora_r": 16, "teacher_temp": 0.7}
        _, canon = _exp._training_fingerprint(
            h, "tree", effective_steps=4000, online_steps=500, online_max_tok=64,
        )
        assert canon["tree_K"] == 6
        assert canon["tree_L"] == 10

    def test_online_profile_includes_online_k(self):
        h = {"online_K": 8, "online_steps": 500}
        _, canon = _exp._training_fingerprint(
            h, "online", effective_steps=4000, online_steps=600, online_max_tok=80,
        )
        assert canon["online_K"] == 8
        assert canon["online_steps"] == 600
        assert canon["max_new_tokens"] == 80

    def test_smoke_uses_effective_steps_not_yaml(self):
        h = {"train_steps": 4000}
        _, canon = _exp._training_fingerprint(
            h, "offline", effective_steps=10, online_steps=10, online_max_tok=30,
        )
        assert canon["steps"] == 10

    def test_profile_for_loss_dir_names(self):
        assert _exp._ckpt_profile_for("kl-gsm8k-q0.6b-q8b") == "offline"
        assert _exp._ckpt_profile_for("kl_tree-gsm8k-q0.6b-q8b") == "tree"
        assert _exp._ckpt_profile_for("online-gsm8k-q0.6b-q8b") == "online"
        assert _exp._ckpt_profile_for("online-ebe-gsm8k-q0.6b-q8b") == "online_ebe"
        assert _exp._ckpt_profile_for("online-kl-tree-gsm8k-q0.6b-q8b") == "online_tree"

    def test_nested_paths_applied_to_loss_ckpt_dirs(self):
        tp = {
            "tree_K": 4,
            "tree_L": 8,
            "lr": 1e-4,
            "lora_r": 16,
            "teacher_temp": 0.8,
            "train_steps": 4000,
        }
        steps = _exp.build_steps(
            draft="Qwen/Qwen3-0.6B",
            target="Qwen/Qwen3-8B",
            train_hparams=tp,
            smoke=False,
            losses_to_run=["kl", "kl_tree"],
            hw_tier="a100",
            force_eval=False,
        )
        offline_id, _ = _exp._training_fingerprint(
            tp, "offline", effective_steps=4000, online_steps=500, online_max_tok=64,
        )
        tree_id, _ = _exp._training_fingerprint(
            tp, "tree", effective_steps=4000, online_steps=500, online_max_tok=64,
        )

        def _out_path(step_id: str) -> str | None:
            for step in steps:
                if step["id"] != step_id:
                    continue
                cmd = step["cmd"]
                if "--output" in cmd:
                    return cmd[cmd.index("--output") + 1]
                if "--adapter" in cmd:
                    return cmd[cmd.index("--adapter") + 1]
            return None

        flat_train = _out_path("train_kl_gsm8k")
        tree_train = _out_path("train_kl_tree_gsm8k")
        flat_merge_dc = None
        tree_merge_dc = None
        for step in steps:
            if step["id"] == "merge_kl_gsm8k":
                flat_merge_dc = step["done_check"]
            if step["id"] == "merge_kl_tree_gsm8k":
                tree_merge_dc = step["done_check"]

        assert flat_train is not None and flat_train.endswith(os.path.join(offline_id))
        assert tree_train is not None and tree_train.endswith(os.path.join(tree_id))
        assert flat_merge_dc is not None and flat_merge_dc.endswith(
            os.path.join(f"{offline_id}_merged", "config.json"),
        )
        assert tree_merge_dc is not None and tree_merge_dc.endswith(
            os.path.join(f"{tree_id}_merged", "config.json"),
        )
        assert offline_id != tree_id

        train_step = next(s for s in steps if s["id"] == "train_kl_gsm8k")
        assert "_hparam_sidecar" in train_step
        assert train_step["_hparam_sidecar"]["profile"] == "offline"
        assert "tree_K" not in train_step["_hparam_sidecar"]

        for step in steps:
            if step["id"] == "eval_kl_gsm8k":
                requires = step.get("requires", "")
                assert offline_id in requires
                return
        pytest.fail("eval_kl_gsm8k step not found")

    def test_canonical_json_stable(self):
        h = {"lr": 1e-4, "lora_r": 16, "teacher_temp": 0.8}
        _, canon = _exp._training_fingerprint(
            h, "offline", effective_steps=4000, online_steps=500, online_max_tok=64,
        )
        payload = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        assert '"profile":"offline"' in payload or '"profile": "offline"' in payload.replace(" ", "")
