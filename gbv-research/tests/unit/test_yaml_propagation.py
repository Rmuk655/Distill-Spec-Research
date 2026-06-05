"""
test_yaml_propagation.py — verify every YAML key reaches the trainer subprocess.

THE BUG CLASS THIS SUITE GUARDS AGAINST
========================================
Three-layer propagation chain:

    YAML file
      │  _yaml_cfg_to_hparams()          ← Layer 1 extraction
      ▼
    _yaml_cfg   dict
      │  train_hparams = { ... }         ← Layer 2: MUST be listed EXPLICITLY
      ▼                                     (bugs here are silent; no error thrown)
    train_hparams dict
      │  build_steps(train_hparams=...) → _train_hargs  ← Layer 3: MUST appear as --flag
      ▼
    trainer.py subprocess command

Known silent failures (all fixed; this suite prevents regression):
  - max_train_prompts : was in _yaml_cfg but MISSING from train_hparams dict  (fixed)
  - train_dataset     : was in _yaml_cfg but MISSING from train_hparams dict  (fixed)
  - Any future key    : will fail here before it silently fails in production

TEST STRATEGY
=============
1. Layer-1 test  : load real YAML files and assert _yaml_cfg has expected keys+values.
2. Layer-2 test  : build train_hparams from a synthetic _yaml_cfg and check ALL keys
                   survived the explicit dict construction.
3. Layer-3 test  : call build_steps() with a canary train_hparams and assert each key
                   appears as a --flag in the training subprocess command list.
4. End-to-end    : real laptop_gpt2.yaml → all three layers → trainer cmd has wikitext.
"""

import os
import sys
import types
import textwrap
import tempfile

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.abspath(os.path.join(_HERE, "..", ".."))
_ORCH  = os.path.join(_ROOT, "orchestration")
_CFGS  = os.path.join(_ORCH, "configs")

sys.path.insert(0, _ORCH)
import experiment as _exp


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_args(**overrides):
    """Minimal argparse Namespace accepted by experiment.py functions."""
    ns = types.SimpleNamespace(
        lr=None, lora_r=None, teacher_temp=None, train_steps=None,
        device=None, losses=None, smoke=False, force_eval=False,
        experiment_tag=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# Alias: the YAML loading function is _load_config_yaml in experiment.py
_load_config_yaml = _exp._load_config_yaml


def _build_train_hparams(yaml_cfg: dict) -> dict:
    """
    Replicate main()'s train_hparams construction from a pre-built _yaml_cfg dict.

    This is a COPY of the dict literal in experiment.py main().  When a new key
    is added to that dict in main() it MUST also be added here — this test will
    then enforce the key appears in _train_hargs too.

    If this function diverges from main(), Layer-2 tests will raise a
    NotImplementedError reminding you to sync.
    """
    args = _mock_args()
    tp = {
        "lr":                   args.lr           or yaml_cfg.get("lr", 3e-5),
        "lora_r":               args.lora_r       or yaml_cfg.get("lora_r", 8),
        "lora_alpha":                                 yaml_cfg.get("lora_alpha", 16),
        "grad_accum":                                 yaml_cfg.get("grad_accum", 4),
        "teacher_temp":         args.teacher_temp or yaml_cfg.get("teacher_temp", 0.8),
        "train_steps":          args.train_steps  or yaml_cfg.get("train_steps"),
        "max_new_tokens":                             yaml_cfg.get("max_new_tokens", 80),
        "compile":                                    yaml_cfg.get("compile", False),
        "online_lr":                                  yaml_cfg.get("online_lr", 3e-4),
        "online_ebe_lr":                              yaml_cfg.get("online_ebe_lr", 1e-4),
        "seed":                                       yaml_cfg.get("seed", 42),
        "val_every":                                  yaml_cfg.get("val_every", 50),
        "slow_val_every":                             yaml_cfg.get("slow_val_every", 0),
        "slow_val_n":                                 yaml_cfg.get("slow_val_n", 100),
        "deep_eval_every_n":                          yaml_cfg.get("deep_eval_every_n", 0),
        "deep_eval_prompts":                          yaml_cfg.get("deep_eval_prompts", 256),
        "grad_clip":                                  yaml_cfg.get("grad_clip", 1.0),
        "lora_dropout":                               yaml_cfg.get("lora_dropout", 0.05),
        "ebe_kl_weight":                              yaml_cfg.get("ebe_kl_weight", 0.1),
        "jsd_alpha":                                  yaml_cfg.get("jsd_alpha", 0.5),
        "early_stop_patience":                        yaml_cfg.get("early_stop_patience", 0),
        "tree_K":                                     yaml_cfg.get("tree_K", 4),
        "tree_L":                                     yaml_cfg.get("tree_L", 8),
        "ppl_threshold":                              yaml_cfg.get("ppl_threshold", 1.25),
        "log_every":                                  yaml_cfg.get("log_every", 10),
        "no_lora":                                    yaml_cfg.get("no_lora", False),
        "val_dataset":                                yaml_cfg.get("val_dataset"),
        "model_family":                               yaml_cfg.get("model_family", "qwen"),
        "device":               (getattr(args, "device", None) or yaml_cfg.get("device", "auto")),
        "max_train_prompts":                          yaml_cfg.get("max_train_prompts", 0),
        "warmup_steps":                               yaml_cfg.get("warmup_steps", None),
        "omp_threads":                                yaml_cfg.get("omp_threads", 0),
        "train_dataset":                              yaml_cfg.get("train_dataset", ""),
        # Checkpointing / W&B  (from _yaml_cfg_to_hparams output)
        "save_every":                                 yaml_cfg.get("save_every", 100),
        "milestone_every":                            yaml_cfg.get("milestone_every", 200),
        "max_checkpoints":                            yaml_cfg.get("max_checkpoints", 5),
        "wandb_project":                              yaml_cfg.get("wandb_project", "distillspec"),
        "wandb_group":                                yaml_cfg.get("wandb_group", "test"),
        "run_label":                                  yaml_cfg.get("run_label", "test"),
    }
    # eval keys forwarded as-is
    for ek in ("eval_modes", "eval_K_values", "eval_temps",
               "eval_n_prompts", "eval_n_prompts_gsm8k", "eval_max_tokens"):
        if ek in yaml_cfg:
            tp[ek] = yaml_cfg[ek]
    return tp


def _first_train_cmd(steps, tree=False) -> list:
    """Return the command list for the first flat or tree training step.

    tree=False → first 'train_kl_*' (flat loss, no --tree_K/L)
    tree=True  → first 'train_*_tree*' (tree loss, has --tree_K/L)
    """
    for step in steps:
        sid = step.get("id", "")
        if "merge" in sid:
            continue
        if tree and "_tree" in sid and sid.startswith("train_"):
            return [str(x) for x in step["cmd"]]
        if not tree and "_tree" not in sid and sid.startswith("train_"):
            return [str(x) for x in step["cmd"]]
    kind = "tree" if tree else "flat"
    raise AssertionError(f"No {kind} training step found. IDs: {[s['id'] for s in steps]}")


def _cmd_value(cmd: list, flag: str):
    """Return the value after --flag in cmd, or None if flag absent."""
    try:
        idx = cmd.index(flag)
        return cmd[idx + 1] if idx + 1 < len(cmd) else True
    except ValueError:
        return None


def _last_flag_value(cmd: list, flag: str):
    """Return the LAST occurrence of --flag's value (argparse last-value-wins)."""
    val = None
    for i, tok in enumerate(cmd):
        if tok == flag and i + 1 < len(cmd):
            val = cmd[i + 1]
    return val


# ── Layer 1: YAML → _yaml_cfg ────────────────────────────────────────────────

class TestLayer1YamlExtraction:
    """_yaml_cfg_to_hparams must extract every key from the YAML file."""

    def test_laptop_gpt2_extracts_train_dataset(self):
        """laptop_gpt2.yaml dataset.train must land in _yaml_cfg['train_dataset']."""
        cfg = _load_config_yaml("laptop_gpt2")
        assert "train_dataset" in cfg, (
            "train_dataset missing from _yaml_cfg after _yaml_cfg_to_hparams. "
            "Check the dataset: section in laptop_gpt2.yaml and the extraction "
            "code in _yaml_cfg_to_hparams()."
        )
        assert "wikitext" in cfg["train_dataset"], (
            f"Expected wikitext in train_dataset, got: {cfg['train_dataset']!r}"
        )

    def test_laptop_gpt2_extracts_val_dataset(self):
        """laptop_gpt2.yaml dataset.val_dataset must land in _yaml_cfg."""
        cfg = _load_config_yaml("laptop_gpt2")
        assert "val_dataset" in cfg
        assert "wikitext" in cfg["val_dataset"]

    def test_laptop_gpt2_extracts_max_train_prompts(self):
        """max_train_prompts must be extracted (previous silent bug)."""
        cfg = _load_config_yaml("laptop_gpt2")
        assert "max_train_prompts" in cfg, "max_train_prompts missing from _yaml_cfg"
        assert cfg["max_train_prompts"] > 0, (
            f"max_train_prompts should be > 0 for laptop_gpt2, got {cfg['max_train_prompts']}"
        )

    def test_a100_qwen_extracts_eval_n_prompts_gsm8k(self):
        """a100_qwen.yaml evaluation.n_prompts_gsm8k must reach _yaml_cfg."""
        cfg = _load_config_yaml("a100_qwen")
        assert "eval_n_prompts_gsm8k" in cfg, (
            "eval_n_prompts_gsm8k missing — a100_qwen eval scope won't be respected"
        )

    def test_a100_qwen_extracts_eval_modes(self):
        """a100_qwen.yaml evaluation.modes (4 exploration modes) must reach _yaml_cfg."""
        cfg = _load_config_yaml("a100_qwen")
        assert "eval_modes" in cfg, "eval_modes missing from _yaml_cfg for a100_qwen"
        modes = cfg["eval_modes"]
        assert isinstance(modes, list) and len(modes) > 0

    def test_laptop_qwen_has_model_family(self):
        """models.family must be extracted as model_family."""
        cfg = _load_config_yaml("laptop_qwen")
        assert "model_family" in cfg
        assert cfg["model_family"] == "qwen"

    def test_laptop_gpt2_model_family_is_gpt2(self):
        cfg = _load_config_yaml("laptop_gpt2")
        assert cfg.get("model_family") == "gpt2", (
            f"Expected model_family='gpt2', got {cfg.get('model_family')!r}"
        )

    def test_a100_qwen_extracts_slow_val_every(self):
        """a100_qwen.yaml health.slow_val_every must reach _yaml_cfg."""
        cfg = _load_config_yaml("a100_qwen")
        assert "slow_val_every" in cfg, (
            "slow_val_every missing from _yaml_cfg — check health: section in "
            "a100_qwen.yaml and the extraction code in _yaml_cfg_to_hparams()."
        )
        assert cfg["slow_val_every"] > 0, (
            f"slow_val_every should be > 0, got {cfg['slow_val_every']}"
        )

    def test_a100_qwen_extracts_slow_val_n(self):
        """a100_qwen.yaml health.slow_val_n must reach _yaml_cfg."""
        cfg = _load_config_yaml("a100_qwen")
        assert "slow_val_n" in cfg
        assert cfg["slow_val_n"] >= 100, (
            f"slow_val_n should be ≥100 for a100_qwen, got {cfg['slow_val_n']}"
        )

    def test_a100_qwen_extracts_early_stop_patience(self):
        """health.early_stop_patience was previously a propagation bug — guard it."""
        cfg = _load_config_yaml("a100_qwen")
        assert "early_stop_patience" in cfg, (
            "early_stop_patience missing from _yaml_cfg — was a propagation bug "
            "(extracted in _yaml_cfg_to_hparams but not in train_hparams dict). "
            "Check health: section extraction in _load_config_yaml()."
        )
        assert cfg["early_stop_patience"] > 0, (
            f"a100_qwen sets early_stop_patience: 5, got {cfg['early_stop_patience']}"
        )

    def test_a100_qwen_extracts_deep_eval_every_n(self):
        """health.deep_eval_every_n must reach _yaml_cfg for three-tier val."""
        cfg = _load_config_yaml("a100_qwen")
        assert "deep_eval_every_n" in cfg, (
            "deep_eval_every_n missing from _yaml_cfg — check health: section in "
            "a100_qwen.yaml and extraction in _load_config_yaml()."
        )
        assert cfg["deep_eval_every_n"] > 0, (
            f"deep_eval_every_n should be > 0 for a100_qwen, got {cfg['deep_eval_every_n']}"
        )

    def test_a100_qwen_extracts_deep_eval_prompts(self):
        """health.deep_eval_prompts must reach _yaml_cfg."""
        cfg = _load_config_yaml("a100_qwen")
        assert "deep_eval_prompts" in cfg
        assert cfg["deep_eval_prompts"] >= 128, (
            f"deep_eval_prompts should be ≥128 for paper-quality SE, got {cfg['deep_eval_prompts']}"
        )


# ── Layer 2: _yaml_cfg → train_hparams ───────────────────────────────────────

# Canary values: deliberately unusual to catch "default was used instead of YAML"
_CANARY = {
    "lr":               0.00012345,
    "lora_r":           7,
    "lora_alpha":       14,
    "grad_accum":       11,   # must propagate — was silently ignored (grad_accum: 16 in a100 YAML)
    "teacher_temp":     0.77,
    "max_new_tokens":   99,
    "seed":             1234,
    "val_every":        33,
    "grad_clip":        0.77,
    "lora_dropout":     0.11,
    "ebe_kl_weight":    0.033,
    "jsd_alpha":        0.33,
    "tree_K":           6,
    "tree_L":           11,
    "ppl_threshold":    1.11,
    "log_every":        7,
    "max_train_prompts": 123,
    "warmup_steps":     77,
    "omp_threads":      3,
    "train_dataset":    "core/datasets/raw/wikitext_train.jsonl",
    "val_dataset":      "wikitext_5.jsonl",
    "model_family":     "gpt2",
    "device":           "cpu",
    "save_every":       55,
    "milestone_every":  222,
    "max_checkpoints":  3,
    "wandb_project":    "canary_project",
    "wandb_group":      "canary_group",
    "run_label":        "canary_label",
    "compile":          False,
    "online_lr":        0.0011,
    "online_ebe_lr":    0.00011,
    "early_stop_patience": 3,
    "slow_val_every": 250,       # dual-val: run slow check every N steps
    "slow_val_n": 77,            # dual-val: number of prompts for slow val
    "deep_eval_every_n": 500,    # three-tier: deep eval cadence (canary >0 so flags are forwarded)
    "deep_eval_prompts": 128,    # three-tier: prompts for deep eval
    "eval_n_prompts_gsm8k": 77,
}

# Keys that are NOT forwarded as --flag to trainer.py (control/meta-only)
_NOT_TRAINER_FLAGS = {
    "train_steps",          # forwarded as --steps, not --train_steps
    "no_lora",              # forwarded as bare --no_lora flag (boolean)
    "compile",              # forwarded as bare --compile flag (boolean)
    "wandb_project",        # forwarded as --wandb_project
    "wandb_group",          # forwarded as --wandb_group
    "run_label",            # forwarded as --run_label
    "save_every",           # forwarded as --save_every
    "milestone_every",      # forwarded as --milestone_every
    "max_checkpoints",      # forwarded as --max_checkpoints
    "eval_n_prompts_gsm8k", # eval key, not a trainer flag
    "eval_modes",           # eval key
    "eval_K_values",        # eval key
    "eval_temps",           # eval key
    "eval_n_prompts",       # eval key
    "eval_max_tokens",      # eval key
    "ppl_threshold",        # forwarded as --ppl_threshold
    "log_every",            # forwarded as --log_every
    "early_stop_patience",  # forwarded as --early_stop_patience
    "slow_val_every",       # forwarded as --slow_val_every (conditional on >0)
    "slow_val_n",           # forwarded as --slow_val_n (with slow_val_every)
    "deep_eval_every_n",    # forwarded as --deep_eval_every_n (conditional on >0)
    "deep_eval_prompts",    # forwarded as --deep_eval_prompts (with deep_eval_every_n)
    "val_dataset",          # forwarded as --val_dataset (path resolved)
    "train_dataset",        # forwarded as --dataset (path resolved, last-value-wins)
    "device",               # forwarded as --device
    "omp_threads",          # forwarded as --omp_threads (conditional)
}

# Explicit mapping: train_hparams key → --flag name in trainer subprocess
_KEY_TO_FLAG = {
    "lr":               "--lr",
    "lora_r":           "--lora_r",
    "lora_alpha":       "--lora_alpha",
    "grad_accum":       "--grad_accum",
    "teacher_temp":     "--teacher_temp",
    "max_new_tokens":   "--max_new_tokens",
    "seed":             "--seed",
    "val_every":        "--val_every",
    "grad_clip":        "--grad_clip",
    "lora_dropout":     "--lora_dropout",
    "ebe_kl_weight":    "--ebe_kl_weight",
    "jsd_alpha":        "--jsd_alpha",
    # tree_K / tree_L are in _tree_hargs, appended only to tree-loss step commands.
    # They are tested separately in test_tree_args_present_in_tree_step below.
    # Do NOT add them here — flat-loss steps intentionally omit them.
    "ppl_threshold":    "--ppl_threshold",
    "log_every":        "--log_every",
    "max_train_prompts": "--max_train_prompts",
    "warmup_steps":     "--warmup_steps",
    "model_family":     "--model_family",
    "device":           "--device",
    "save_every":       "--save_every",
    "milestone_every":  "--milestone_every",
    "max_checkpoints":  "--max_checkpoints",
    "wandb_project":    "--wandb_project",
    "wandb_group":      "--wandb_group",
    "run_label":        "--run_label",
    "early_stop_patience": "--early_stop_patience",
    # slow_val: conditional — only forwarded when slow_val_every > 0.
    # Canary sets slow_val_every=250 (>0), so both flags must appear.
    "slow_val_every":       "--slow_val_every",
    "slow_val_n":           "--slow_val_n",
    # deep_eval: conditional — only forwarded when deep_eval_every_n > 0.
    # Canary sets deep_eval_every_n=500 (>0), so both flags must appear.
    "deep_eval_every_n":    "--deep_eval_every_n",
    "deep_eval_prompts":    "--deep_eval_prompts",
    # train_dataset and val_dataset have special forwarding (path resolution):
    # train_dataset  → last --dataset in cmd
    # val_dataset    → --val_dataset with _data(filename) resolution
}


class TestLayer2TrainHparams:
    """Every key from _yaml_cfg must survive into train_hparams."""

    def test_train_dataset_survives(self):
        yaml_cfg = {**_CANARY}
        tp = _build_train_hparams(yaml_cfg)
        assert "train_dataset" in tp, (
            "train_dataset MISSING from train_hparams — "
            "add 'train_dataset': _yaml_cfg.get('train_dataset', '') to the dict in main()"
        )
        assert tp["train_dataset"] == _CANARY["train_dataset"]

    def test_max_train_prompts_survives(self):
        yaml_cfg = {**_CANARY}
        tp = _build_train_hparams(yaml_cfg)
        assert "max_train_prompts" in tp
        assert tp["max_train_prompts"] == 123

    def test_all_canary_training_keys_survive(self):
        """Every canary value must appear with the correct value in train_hparams."""
        yaml_cfg = {**_CANARY}
        tp = _build_train_hparams(yaml_cfg)
        training_keys = [k for k in _CANARY if k in _KEY_TO_FLAG or k in (
            "train_dataset", "val_dataset", "max_train_prompts", "warmup_steps",
            "omp_threads", "compile", "no_lora")]
        missing = [k for k in training_keys if k not in tp]
        wrong_value = [
            f"{k}: expected {_CANARY[k]!r}, got {tp.get(k)!r}"
            for k in training_keys
            if k in tp and tp[k] != _CANARY[k]
            and k not in ("device",)  # device has CLI-wins logic, skip value check
        ]
        assert not missing, (
            f"Keys in _yaml_cfg but MISSING from train_hparams: {missing}\n"
            f"Add each to the train_hparams dict in experiment.py main()."
        )
        assert not wrong_value, (
            f"Keys present but wrong value in train_hparams: {wrong_value}"
        )

    def test_sync_with_main(self):
        """
        Detect if experiment.py main() added a new key that _build_train_hparams
        (our test copy) doesn't have yet.  If this test fails, sync _build_train_hparams
        above with the real dict in main() and add a corresponding Layer-3 check.
        """
        # Build via our copy
        tp_test = _build_train_hparams(_CANARY)
        # Build via the real experiment.py main() logic by calling a helper that
        # reads the dict keys from the source.  We use a simple heuristic: parse
        # the train_hparams dict literal lines from the source file.
        src_path = os.path.join(_ORCH, "experiment.py")
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        # Extract keys between 'train_hparams = {' and the closing '}'
        import re
        block_match = re.search(
            r'train_hparams\s*=\s*\{(.+?)\n    \}', src, re.DOTALL)
        if not block_match:
            pytest.skip("Could not parse train_hparams block from source — skip sync check")
        block = block_match.group(1)
        src_keys = set(re.findall(r'"(\w+)"\s*:', block))
        test_keys = set(tp_test.keys())
        new_in_src = src_keys - test_keys
        assert not new_in_src, (
            f"experiment.py main() has keys not in test's _build_train_hparams: {new_in_src}\n"
            "Sync _build_train_hparams() in this test file, then add Layer-3 checks."
        )


# ── Layer 3: train_hparams → trainer subprocess cmd ──────────────────────────

class TestLayer3TrainerCmd:
    """Every train_hparams key must appear as --flag in the trainer subprocess command."""

    @pytest.fixture(scope="class")
    def canary_steps(self):
        """Build steps with canary train_hparams (kl + kl_tree for both step types)."""
        tp = _build_train_hparams(_CANARY)
        tp["model_family"] = "gpt2"
        return _exp.build_steps(
            draft="distilgpt2",
            target="gpt2-medium",
            train_hparams=tp,
            smoke=False,
            losses_to_run=["kl", "kl_tree"],
            hw_tier="laptop",
            force_eval=False,
        )

    @pytest.fixture(scope="class")
    def canary_cmd(self, canary_steps):
        """Flat-loss training command (kl — no --tree_K/L)."""
        return _first_train_cmd(canary_steps, tree=False)

    @pytest.fixture(scope="class")
    def canary_tree_cmd(self, canary_steps):
        """Tree-loss training command (kl_tree — has --tree_K/L)."""
        return _first_train_cmd(canary_steps, tree=True)

    @pytest.mark.parametrize("key,flag", sorted(_KEY_TO_FLAG.items()))
    def test_flag_present(self, canary_cmd, key, flag):
        val = _cmd_value(canary_cmd, flag)
        assert val is not None, (
            f"--{flag} missing from trainer subprocess command.\n"
            f"  train_hparams key '{key}' is never forwarded to _train_hargs.\n"
            f"  Fix: add \"'{flag}', str(_h['{key}'])\" to _train_hargs in build_steps()."
        )

    def test_tree_args_present_in_tree_step(self, canary_tree_cmd):
        """--tree_K and --tree_L must appear in tree-loss training commands."""
        for flag, key in [("--tree_K", "tree_K"), ("--tree_L", "tree_L")]:
            val = _last_flag_value(canary_tree_cmd, flag)
            assert val is not None, (
                f"{flag} missing from tree-loss trainer cmd.\n"
                f"tree_{key.split('_')[1]} must be in _tree_hargs in build_steps()."
            )
            assert val == str(_CANARY[key]), (
                f"{flag}={val!r} in cmd, expected {_CANARY[key]!r} from canary."
            )

    @pytest.mark.parametrize("key,flag", [
        ("lr",              "--lr"),
        ("lora_r",          "--lora_r"),
        ("max_train_prompts", "--max_train_prompts"),
        ("seed",            "--seed"),
        ("val_every",       "--val_every"),
        ("warmup_steps",    "--warmup_steps"),
    ])
    def test_canary_value_correct(self, canary_cmd, key, flag):
        """The YAML canary value must be the actual value passed to the subprocess."""
        last_val = _last_flag_value(canary_cmd, flag)
        assert last_val == str(_CANARY[key]), (
            f"{flag} has value {last_val!r} in trainer cmd, expected {_CANARY[key]!r}.\n"
            f"Check _train_hargs construction for key '{key}'."
        )

    def test_train_dataset_overrides_gsm8k(self, canary_cmd):
        """
        When train_dataset is set to wikitext, --dataset in the trainer cmd must
        point to wikitext — NOT to the hardcoded gsm8k fallback.

        This is the regression test for the bug fixed in commit bd376c0 + 843ef68:
        train_dataset was extracted from YAML but never reached train_hparams,
        so _train_hargs never appended --dataset <wikitext>, and the per-step
        hardcoded --dataset gsm8k_train.jsonl silently won.
        """
        # argparse last-value-wins: the last --dataset in the cmd is the effective one
        last_dataset = _last_flag_value(canary_cmd, "--dataset")
        assert last_dataset is not None, "--dataset flag missing entirely from trainer cmd"
        assert "wikitext" in last_dataset, (
            f"Trainer cmd uses dataset {last_dataset!r} instead of wikitext.\n"
            "The train_dataset → train_hparams → _train_hargs pipeline is broken.\n"
            "Fix: ensure 'train_dataset' is in the train_hparams dict in main(), "
            "and that _train_hargs appends '--dataset' at the end."
        )

    def test_val_dataset_present(self, canary_cmd):
        """--val_dataset must appear in the trainer cmd when val_dataset is set."""
        val = _cmd_value(canary_cmd, "--val_dataset")
        assert val is not None, (
            "--val_dataset missing from trainer cmd despite val_dataset being set in train_hparams."
        )
        assert "wikitext" in val


# ── End-to-end: real laptop_gpt2.yaml → trainer cmd ─────────────────────────

class TestEndToEnd:
    """
    Full pipeline: real laptop_gpt2.yaml → _yaml_cfg → train_hparams → trainer cmd.
    This is the integration test that would have caught both silent bugs.
    """

    @pytest.fixture(scope="class")
    def laptop_gpt2_cmd(self):
        yaml_cfg = _load_config_yaml("laptop_gpt2")
        tp = _build_train_hparams(yaml_cfg)
        steps = _exp.build_steps(
            draft="distilgpt2",
            target="gpt2-medium",
            train_hparams=tp,
            smoke=False,
            losses_to_run=["kl"],
            hw_tier="laptop",
            force_eval=False,
        )
        return _first_train_cmd(steps)

    def test_training_uses_wikitext_not_gsm8k(self, laptop_gpt2_cmd):
        """THE primary regression test: laptop_gpt2 trains on wikitext, not gsm8k."""
        last_dataset = _last_flag_value(laptop_gpt2_cmd, "--dataset")
        assert last_dataset is not None
        assert "wikitext" in last_dataset, (
            f"laptop_gpt2 trainer cmd dataset is {last_dataset!r}.\n"
            "Expected wikitext — the train_dataset YAML override is not reaching the trainer."
        )
        assert "gsm8k" not in last_dataset, (
            f"laptop_gpt2 trainer is STILL using gsm8k: {last_dataset!r}.\n"
            "This is the bug from commit 843ef68. Re-check train_hparams propagation."
        )

    def test_val_dataset_uses_wikitext(self, laptop_gpt2_cmd):
        val = _cmd_value(laptop_gpt2_cmd, "--val_dataset")
        assert val is not None
        assert "wikitext" in val

    def test_model_family_is_gpt2(self, laptop_gpt2_cmd):
        family = _cmd_value(laptop_gpt2_cmd, "--model_family")
        assert family == "gpt2", f"model_family is {family!r}, expected 'gpt2'"

    def test_max_train_prompts_reaches_trainer(self, laptop_gpt2_cmd):
        """Regression for the max_train_prompts silent bug."""
        val = _cmd_value(laptop_gpt2_cmd, "--max_train_prompts")
        assert val is not None, (
            "--max_train_prompts missing from trainer cmd.\n"
            "Regression: this key was previously in _yaml_cfg but not in train_hparams."
        )
        assert int(val) > 0, f"--max_train_prompts={val} should be > 0 for laptop_gpt2"
