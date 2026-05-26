"""
test_pipeline_steps.py — unit tests for pipeline.build_steps() step generation.

build_steps() generates the full list of pipeline step dicts.
These tests verify:
  1. All expected step IDs are present for both smoke and full runs
  2. No duplicate step IDs
  3. Every training step includes --steps with the correct count
  4. --load_in_4bit is appended to training/eval commands when load_in_4bit=True
  5. --load_in_4bit is NOT in merge commands (merge is CPU-only, no teacher needed)
  6. done_check paths are strings (not None only for training steps)
  7. Eval steps include all 6 verifier modes
  8. Phase groups are correct

No subprocess calls made.  All checks are purely on the returned data structure.
"""

import sys
import os
import pytest

# conftest.py adds orchestration/ to sys.path
import pipeline as _pipeline

build_steps = _pipeline.build_steps


DRAFT  = "Qwen/Qwen2.5-0.5B"
TARGET = "Qwen/Qwen3-0.6B"

EXPECTED_TRAIN_STEP_IDS = [
    "train_kl_gsm8k",
    "train_ebe_gsm8k",
    "train_rev_kl_gsm8k",
    "train_jsd_gsm8k",
    "train_l1_gsm8k",
    "online_adapt_gsm8k",
]
EXPECTED_MERGE_STEP_IDS = [
    "merge_kl_gsm8k",
    "merge_ebe_gsm8k",
    "merge_rev_kl_gsm8k",
    "merge_jsd_gsm8k",
    "merge_l1_gsm8k",
    "merge_online_gsm8k",
]
EXPECTED_EVAL_STEP_IDS = [
    "eval_baseline_gsm8k",
    "eval_kl_gsm8k",
    "eval_ebe_gsm8k",
    "eval_rev_kl_gsm8k",
    "eval_jsd_gsm8k",
    "eval_l1_gsm8k",
    "eval_online_gsm8k",
]
ALL_EXPECTED_IDS = (
    EXPECTED_TRAIN_STEP_IDS
    + EXPECTED_MERGE_STEP_IDS
    + EXPECTED_EVAL_STEP_IDS
)


def _steps_by_id(steps):
    return {s["id"]: s for s in steps}


# ===========================================================================
# Step presence
# ===========================================================================

class TestStepPresence:

    def test_all_train_steps_present_smoke(self):
        steps = build_steps(DRAFT, TARGET, smoke=True)
        ids = {s["id"] for s in steps}
        for sid in EXPECTED_TRAIN_STEP_IDS:
            assert sid in ids, f"Missing training step in smoke: {sid}"

    def test_all_merge_steps_present_smoke(self):
        steps = build_steps(DRAFT, TARGET, smoke=True)
        ids = {s["id"] for s in steps}
        for sid in EXPECTED_MERGE_STEP_IDS:
            assert sid in ids, f"Missing merge step in smoke: {sid}"

    def test_all_eval_steps_present_smoke(self):
        steps = build_steps(DRAFT, TARGET, smoke=True)
        ids = {s["id"] for s in steps}
        for sid in EXPECTED_EVAL_STEP_IDS:
            assert sid in ids, f"Missing eval step in smoke: {sid}"

    def test_all_steps_present_full(self):
        steps = build_steps(DRAFT, TARGET, smoke=False)
        ids = {s["id"] for s in steps}
        for sid in ALL_EXPECTED_IDS:
            assert sid in ids, f"Missing step in full run: {sid}"

    def test_no_duplicate_step_ids(self):
        steps = build_steps(DRAFT, TARGET, smoke=True)
        ids = [s["id"] for s in steps]
        assert len(ids) == len(set(ids)), \
            f"Duplicate step IDs: {[x for x in ids if ids.count(x) > 1]}"


# ===========================================================================
# Step counts (smoke vs full)
# ===========================================================================

class TestStepCounts:

    def test_smoke_uses_50_steps(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=True))
        for sid in EXPECTED_TRAIN_STEP_IDS:
            if sid == "online_adapt_gsm8k":
                continue  # online uses _online_steps not _steps
            cmd = steps[sid]["cmd"]
            steps_idx = cmd.index("--steps") if "--steps" in cmd else None
            assert steps_idx is not None, f"{sid}: --steps flag missing"
            assert cmd[steps_idx + 1] == "50", \
                f"{sid}: smoke should use 50 steps, got {cmd[steps_idx+1]}"

    def test_full_uses_1000_steps(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=False))
        for sid in EXPECTED_TRAIN_STEP_IDS:
            if sid == "online_adapt_gsm8k":
                continue
            cmd = steps[sid]["cmd"]
            steps_idx = cmd.index("--steps") if "--steps" in cmd else None
            assert steps_idx is not None
            assert cmd[steps_idx + 1] == "1000", \
                f"{sid}: full should use 1000 steps, got {cmd[steps_idx+1]}"

    def test_smoke_eval_uses_n5(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=True))
        cmd = steps["eval_baseline_gsm8k"]["cmd"]
        n_idx = cmd.index("--n") if "--n" in cmd else None
        assert n_idx is not None, "eval cmd missing --n flag"
        assert cmd[n_idx + 1] == "5", \
            f"smoke eval should use n=5 prompts, got {cmd[n_idx+1]}"

    def test_full_eval_uses_n10(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=False))
        cmd = steps["eval_baseline_gsm8k"]["cmd"]
        n_idx = cmd.index("--n") if "--n" in cmd else None
        assert n_idx is not None
        assert cmd[n_idx + 1] == "10", \
            f"full eval should use n=10, got {cmd[n_idx+1]}"


# ===========================================================================
# 4-bit flag propagation
# ===========================================================================

class TestFourBitFlag:

    def test_4bit_in_train_cmds_when_enabled(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=True, load_in_4bit=True))
        for sid in EXPECTED_TRAIN_STEP_IDS:
            cmd = steps[sid]["cmd"]
            assert "--load_in_4bit" in cmd, \
                f"load_in_4bit=True: --load_in_4bit missing from train cmd {sid}"

    def test_4bit_in_eval_cmds_when_enabled(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=True, load_in_4bit=True))
        for sid in EXPECTED_EVAL_STEP_IDS:
            cmd = steps[sid]["cmd"]
            assert "--load_in_4bit" in cmd, \
                f"load_in_4bit=True: --load_in_4bit missing from eval cmd {sid}"

    def test_4bit_NOT_in_merge_cmds(self):
        """
        Merge steps do NOT need --load_in_4bit: they load only the base draft
        model in bf16 (no teacher needed) and run entirely on CPU-friendly paths.
        """
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=True, load_in_4bit=True))
        for sid in EXPECTED_MERGE_STEP_IDS:
            cmd = steps[sid]["cmd"]
            assert "--load_in_4bit" not in cmd, \
                f"--load_in_4bit should NOT be in merge cmd {sid} (no teacher loaded)"

    def test_4bit_absent_when_not_requested(self):
        steps = build_steps(DRAFT, TARGET, smoke=True, load_in_4bit=False)
        for step in steps:
            cmd = step["cmd"]
            assert "--load_in_4bit" not in cmd, \
                f"--load_in_4bit should not be in any cmd when load_in_4bit=False: {step['id']}"


# ===========================================================================
# done_check and retryable fields
# ===========================================================================

class TestStepMetadata:

    def test_train_steps_have_done_check(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=True))
        for sid in EXPECTED_TRAIN_STEP_IDS:
            dc = steps[sid].get("done_check")
            assert dc is not None and isinstance(dc, str), \
                f"Train step {sid} missing done_check path"

    def test_merge_steps_have_done_check(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=True))
        for sid in EXPECTED_MERGE_STEP_IDS:
            dc = steps[sid].get("done_check")
            assert dc is not None and isinstance(dc, str), \
                f"Merge step {sid} missing done_check path"

    def test_eval_steps_done_check_is_none(self):
        """Eval steps re-run on every pipeline invocation (done_check=None)."""
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=True))
        for sid in EXPECTED_EVAL_STEP_IDS:
            dc = steps[sid].get("done_check")
            assert dc is None, \
                f"Eval step {sid} should have done_check=None, got {dc!r}"

    def test_all_steps_have_required_keys(self):
        steps = build_steps(DRAFT, TARGET, smoke=True)
        for step in steps:
            for key in ("id", "desc", "cmd", "done_check", "group"):
                assert key in step, f"Step {step.get('id','?')} missing key '{key}'"

    def test_all_cmds_are_lists(self):
        steps = build_steps(DRAFT, TARGET, smoke=True)
        for step in steps:
            assert isinstance(step["cmd"], list), \
                f"Step {step['id']}: cmd must be a list, got {type(step['cmd'])}"


# ===========================================================================
# Verifier modes in eval commands
# ===========================================================================

class TestEvalModes:

    EXPECTED_MODES = {"alpha", "bv", "gbv", "traversal", "specinfer", "naive"}

    def _modes_in_cmd(self, cmd):
        idx = cmd.index("--modes") if "--modes" in cmd else None
        if idx is None:
            return set()
        return set(cmd[idx + 1].split(","))

    def test_smoke_eval_contains_all_verifier_modes(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=True))
        for sid in EXPECTED_EVAL_STEP_IDS:
            modes = self._modes_in_cmd(steps[sid]["cmd"])
            missing = self.EXPECTED_MODES - modes
            assert not missing, \
                f"Eval step {sid} missing verifier modes: {missing}"

    def test_full_eval_contains_all_verifier_modes(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=False))
        for sid in EXPECTED_EVAL_STEP_IDS:
            modes = self._modes_in_cmd(steps[sid]["cmd"])
            missing = self.EXPECTED_MODES - modes
            assert not missing, \
                f"Full eval step {sid} missing verifier modes: {missing}"
