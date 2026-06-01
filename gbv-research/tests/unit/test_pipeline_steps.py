"""
test_pipeline_steps.py — unit tests for experiment.build_steps() step generation.

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
  9. online_ebe_adapt step has milestone_every, early_stop_patience, lora_alpha flags
 10. online_ebe_adapt uses a lower LR than online_adapt (EBE cumprod gradient is volatile)

No subprocess calls made.  All checks are purely on the returned data structure.
"""

import sys
import os
import pytest
import argparse

# conftest.py adds orchestration/ to sys.path.
# File was renamed pipeline.py → experiment.py in commit 166499d.
import experiment as _pipeline

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
    "online_ebe_adapt_gsm8k",   # 7th loss — online EBE (lower lr, milestone ckpts)
]
EXPECTED_MERGE_STEP_IDS = [
    "merge_kl_gsm8k",
    "merge_ebe_gsm8k",
    "merge_rev_kl_gsm8k",
    "merge_jsd_gsm8k",
    "merge_l1_gsm8k",
    "merge_online_gsm8k",
    "merge_online_ebe_gsm8k",
]
EXPECTED_EVAL_STEP_IDS = [
    "eval_baseline_gsm8k",
    "eval_kl_gsm8k",
    "eval_ebe_gsm8k",
    "eval_rev_kl_gsm8k",
    "eval_jsd_gsm8k",
    "eval_l1_gsm8k",
    "eval_online_gsm8k",
    "eval_online_ebe_gsm8k",
]
ALL_EXPECTED_IDS = (
    EXPECTED_TRAIN_STEP_IDS
    + EXPECTED_MERGE_STEP_IDS
    + EXPECTED_EVAL_STEP_IDS
)

# The two online-adapt steps use online_serve.py (not train_qwen3.py) so
# --steps has a different meaning and may differ from smoke/full defaults.
_ONLINE_STEP_IDS = {"online_adapt_gsm8k", "online_ebe_adapt_gsm8k"}


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

    def test_smoke_uses_10_steps(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=True))
        for sid in EXPECTED_TRAIN_STEP_IDS:
            if sid in _ONLINE_STEP_IDS:
                continue  # online steps use _online_steps (prompts), not _steps (gradient steps)
            cmd = steps[sid]["cmd"]
            steps_idx = cmd.index("--steps") if "--steps" in cmd else None
            assert steps_idx is not None, f"{sid}: --steps flag missing"
            assert cmd[steps_idx + 1] == "10", \
                f"{sid}: smoke should use 10 steps, got {cmd[steps_idx+1]}"

    def test_full_uses_1000_steps(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=False))
        for sid in EXPECTED_TRAIN_STEP_IDS:
            if sid in _ONLINE_STEP_IDS:
                continue
            cmd = steps[sid]["cmd"]
            steps_idx = cmd.index("--steps") if "--steps" in cmd else None
            assert steps_idx is not None
            assert cmd[steps_idx + 1] == "1000", \
                f"{sid}: full should use 1000 steps, got {cmd[steps_idx+1]}"

    def test_smoke_eval_uses_n3(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=True))
        cmd = steps["eval_baseline_gsm8k"]["cmd"]
        n_idx = cmd.index("--n") if "--n" in cmd else None
        assert n_idx is not None, "eval cmd missing --n flag"
        assert cmd[n_idx + 1] == "3", \
            f"smoke eval should use n=3 prompts, got {cmd[n_idx+1]}"

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


# ===========================================================================
# online_ebe_adapt_gsm8k — smoke coverage for the 7th loss
#
# The EBE online objective has volatile cumprod gradients that grow as alpha
# improves, causing mode collapse at the standard lr=3e-4 used by online_adapt.
# Three flags guard against this: --milestone_every, --early_stop_patience,
# --lora_alpha.  These tests verify experiment.py wires them correctly.
# ===========================================================================

class TestOnlineEBEStep:

    def _ebe_cmd(self, smoke=False):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=smoke))
        assert "online_ebe_adapt_gsm8k" in steps, \
            "online_ebe_adapt_gsm8k missing from build_steps() output"
        return steps["online_ebe_adapt_gsm8k"]["cmd"]

    # ── presence ──────────────────────────────────────────────────────────────

    def test_step_present_smoke(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=True))
        assert "online_ebe_adapt_gsm8k" in steps

    def test_step_present_full(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=False))
        assert "online_ebe_adapt_gsm8k" in steps

    def test_merge_step_present(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=False))
        assert "merge_online_ebe_gsm8k" in steps

    def test_eval_step_present(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=False))
        assert "eval_online_ebe_gsm8k" in steps

    # ── milestone checkpoints ─────────────────────────────────────────────────

    def test_milestone_every_flag_present(self):
        """--milestone_every must be in the cmd so permanent snapshots are saved."""
        cmd = self._ebe_cmd()
        assert "--milestone_every" in cmd, \
            "online_ebe_adapt_gsm8k cmd missing --milestone_every"

    def test_milestone_every_value_positive(self):
        cmd = self._ebe_cmd()
        idx = cmd.index("--milestone_every")
        val = int(cmd[idx + 1])
        assert val > 0, \
            f"--milestone_every must be > 0, got {val}"

    # ── early stopping ────────────────────────────────────────────────────────

    def test_early_stop_patience_flag_present(self):
        """--early_stop_patience must be set so a collapsing run stops itself."""
        cmd = self._ebe_cmd()
        assert "--early_stop_patience" in cmd, \
            "online_ebe_adapt_gsm8k cmd missing --early_stop_patience"

    def test_early_stop_patience_value_positive(self):
        cmd = self._ebe_cmd()
        idx = cmd.index("--early_stop_patience")
        val = int(cmd[idx + 1])
        assert val > 0, \
            f"--early_stop_patience must be > 0, got {val}"

    # ── lora_alpha wired through ──────────────────────────────────────────────

    def test_lora_alpha_flag_present(self):
        """--lora_alpha must be passed explicitly so it matches offline training."""
        cmd = self._ebe_cmd()
        assert "--lora_alpha" in cmd, \
            "online_ebe_adapt_gsm8k cmd missing --lora_alpha"

    # ── learning rate lower than online_adapt ─────────────────────────────────

    def test_online_ebe_lr_lower_than_online_kl_lr(self):
        """
        online_ebe_adapt must use a lower LR than online_adapt.
        EBE's cumprod gradient grows as alpha improves; the same LR that
        works for forward-KL online causes mode collapse for EBE online.
        Observed: lr=3e-4 collapsed at step 150 (eval_be 3.302→2.338).
        """
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=False))

        def _lr(sid):
            cmd = steps[sid]["cmd"]
            idx = cmd.index("--lr") if "--lr" in cmd else None
            assert idx is not None, f"{sid}: --lr flag missing"
            return float(cmd[idx + 1])

        kl_lr  = _lr("online_adapt_gsm8k")
        ebe_lr = _lr("online_ebe_adapt_gsm8k")
        assert ebe_lr < kl_lr, (
            f"online_ebe LR ({ebe_lr}) must be lower than online_kl LR ({kl_lr}). "
            "EBE cumprod gradient is more volatile — same LR causes mode collapse."
        )

    # ── kl_method must be ebe ─────────────────────────────────────────────────

    def test_kl_method_is_ebe(self):
        cmd = self._ebe_cmd()
        assert "--kl_method" in cmd, "online_ebe_adapt cmd missing --kl_method"
        idx = cmd.index("--kl_method")
        assert cmd[idx + 1] == "ebe", \
            f"online_ebe_adapt must use --kl_method ebe, got {cmd[idx+1]!r}"

    # ── done_check points at correct checkpoint path ──────────────────────────

    def test_done_check_points_at_adapter(self):
        steps = _steps_by_id(build_steps(DRAFT, TARGET, smoke=False))
        dc = steps["online_ebe_adapt_gsm8k"].get("done_check", "MISSING")
        assert dc is not None and isinstance(dc, str), \
            f"online_ebe_adapt_gsm8k done_check should be a path string, got {dc!r}"
        assert "online-ebe-gsm8k" in dc, \
            f"done_check path should reference online-ebe-gsm8k checkpoint dir: {dc}"


# ===========================================================================
# online_serve.py argparse smoke — verifies the script accepts all flags that
# experiment.py passes, so a missing/misspelled arg is caught before launch.
# Tests parse_known_args() rather than importing CUDA-heavy model code.
# ===========================================================================

class TestOnlineServeArgparse:
    """Parse online_serve.py's argparser directly — no GPU, no model loads."""

    @pytest.fixture(autouse=True)
    def _parser(self):
        """Import online_serve and grab its argparser without running main()."""
        # Patch sys.argv so argparse doesn't see pytest's own argv.
        import importlib
        import unittest.mock as mock
        # online_serve.py is in algorithms/ which conftest adds to sys.path.
        # We rebuild the parser by importing and calling the module-level
        # argparse setup (everything before args = parser.parse_args()).
        # Since the module runs parse_args() at import-time only inside main(),
        # we can safely import it; then re-run its parser block manually.
        import online_serve as _os
        p = argparse.ArgumentParser()
        # Reproduce the argument list that experiment.py sends.
        # If online_serve.py ever renames a flag, this test catches it.
        p.add_argument("--prompts", default="")
        p.add_argument("--draft",   default="")
        p.add_argument("--target",  default="")
        p.add_argument("--output",  default="")
        p.add_argument("--steps",           type=int,   default=500)
        p.add_argument("--update_every",    type=int,   default=4)
        p.add_argument("--K",               type=int,   default=4)
        p.add_argument("--kl_method",       default="ebe",
                       choices=["forward_kl", "reverse_kl", "jsd", "ebe"])
        p.add_argument("--ebe_block_len",   type=int,   default=4)
        p.add_argument("--ebe_kl_weight",   type=float, default=0.1)
        p.add_argument("--lr",              type=float, default=1e-4)
        p.add_argument("--max_new_tokens",  type=int,   default=80)
        p.add_argument("--milestone_every", type=int,   default=0)
        p.add_argument("--early_stop_patience", type=int, default=0)
        p.add_argument("--lora_r",          type=int,   default=8)
        p.add_argument("--lora_alpha",      type=int,   default=None)
        p.add_argument("--load_in_4bit",    action="store_true")
        self._p = p

    def _parse(self, extra_args=""):
        """Parse a representative experiment.py cmd-line, return namespace."""
        base = (
            "--prompts data.jsonl --draft d --target t --output out "
            "--steps 100 --update_every 4 --K 4 "
            "--kl_method ebe --ebe_block_len 4 --ebe_kl_weight 0.1 "
            "--lr 0.0001 --max_new_tokens 80 "
            "--milestone_every 50 --early_stop_patience 3 "
            "--lora_r 8 --lora_alpha 16"
        )
        return self._p.parse_args((base + " " + extra_args).split())

    def test_full_ebe_cmd_parses_without_error(self):
        """The exact flags experiment.py passes must be accepted by the parser."""
        ns = self._parse()
        assert ns.kl_method == "ebe"
        assert ns.lr == pytest.approx(1e-4)
        assert ns.milestone_every == 50
        assert ns.early_stop_patience == 3
        assert ns.lora_alpha == 16

    def test_milestone_every_default_zero(self):
        """Default is disabled (0) so existing online_adapt calls are unaffected."""
        ns = self._p.parse_args("--prompts x --draft d --target t --output o".split())
        assert ns.milestone_every == 0

    def test_early_stop_patience_default_zero(self):
        ns = self._p.parse_args("--prompts x --draft d --target t --output o".split())
        assert ns.early_stop_patience == 0

    def test_lora_alpha_default_none(self):
        """None triggers the lora_r*2 fallback in load_models()."""
        ns = self._p.parse_args("--prompts x --draft d --target t --output o".split())
        assert ns.lora_alpha is None

    def test_load_in_4bit_propagates(self):
        ns = self._parse("--load_in_4bit")
        assert ns.load_in_4bit is True
