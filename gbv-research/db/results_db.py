"""
results_db.py — SQLite persistence layer for all evaluation results.

Schema
------
runs         : one row per evaluation run (model × dataset × mode × hyperparams)
per_prompt   : per-prompt alpha / BE for each run (enables per-category breakdown)
train_runs   : training loss curves (step → loss)
"""

import sqlite3
import os
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# DB path resolution — portable across machines and cloud environments.
#
# Priority order (first non-empty wins):
#   1. SPECDIST_DB_PATH env var   — set by experiment.py when --storage_root is
#                                   given; propagated to all subprocesses so
#                                   every tool (train, eval, online) writes to
#                                   the same DB without knowing the cloud layout.
#   2. Module-relative default    — gbv-research/db/results.db  (local dev).
#
# Cloud launchers should set SPECDIST_DB_PATH to a path on persistent storage:
#   Colab:   /content/drive/MyDrive/specdist/results.db
#   Modal:   /vol/results.db
#   Kaggle:  /kaggle/working/specdist/results.db
#   RunPod:  /workspace/specdist/results.db
# ---------------------------------------------------------------------------
DB_PATH = (
    os.environ.get("SPECDIST_DB_PATH")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.db")
)

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_tag         TEXT    UNIQUE,     -- human-readable: 20260521_143022_ebe200_specinfer_diverse50_K3
    ts              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    draft_label     TEXT    NOT NULL,   -- 'baseline' | 'kl200' | 'ebe200' | custom
    draft_path      TEXT    NOT NULL,
    target_path     TEXT    NOT NULL,
    loss_name       TEXT    NOT NULL,   -- 'baseline' | 'kl' | 'ebe'
    train_steps     INTEGER NOT NULL DEFAULT 0,
    learning_rate   REAL    NOT NULL DEFAULT 0.0,
    lora_rank       INTEGER NOT NULL DEFAULT 0,
    dataset         TEXT    NOT NULL,   -- 'diverse50' | 'gsm8k' | 'humaneval' | ...
    n_prompts       INTEGER NOT NULL,
    mode            TEXT    NOT NULL,   -- 'alpha' | 'specinfer' | 'gbv' | 'traversal' | 'bv'
    K               INTEGER NOT NULL DEFAULT 1,
    L               INTEGER NOT NULL DEFAULT 5,
    temperature     REAL    NOT NULL DEFAULT 1.0,
    alpha_mean      REAL,
    alpha_std       REAL,
    alpha_ci95      REAL,
    block_eff       REAL,               -- mean block efficiency
    block_eff_std   REAL,
    throughput      REAL,               -- tok/sec
    ms_per_tok      REAL,
    peak_vram_mb    REAL,
    task_score      REAL,               -- exact match / pass@1 (0–1); NULL if not measured
    perplexity      REAL,               -- student model perplexity on dataset; NULL if not measured
    draft_latency_ms REAL,              -- avg draft model inference time per block (ms)
    verify_latency_ms REAL,             -- avg target verification time per block (ms)
    notes           TEXT,
    experiment_tag  TEXT,               -- free-text label for this experimental run, e.g.
                                        -- "v2 EBE loss with clipped accept weight" or
                                        -- "baseline laptop run May-22". Used to group and
                                        -- filter historical runs in the viz dashboard.
                                        -- Every run keeps its unique run_tag timestamp — this
                                        -- field is an additional human annotation layer.
    hw_tier         TEXT    NOT NULL DEFAULT 'laptop'
                                        -- hardware tier: 'laptop' | 'colab' | 'a100'
                                        -- laptop = smoke/correctness (Qwen2.5-0.5B -> Qwen3-0.6B)
                                        -- colab  = trend formation, 4-bit quantized target
                                        -- a100   = paper-quality, bf16, no quantization
);

CREATE INDEX IF NOT EXISTS idx_runs_label ON runs(draft_label, mode, dataset);

CREATE TABLE IF NOT EXISTS per_prompt (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    prompt_idx  INTEGER NOT NULL,
    category    TEXT,
    alpha       REAL,
    block_eff   REAL,
    gen_tokens  INTEGER,
    wall_ms     REAL
);

CREATE INDEX IF NOT EXISTS idx_pp_run ON per_prompt(run_id);

CREATE TABLE IF NOT EXISTS train_curves (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    label       TEXT    NOT NULL,   -- 'kl200' | 'ebe200' | 'ebe_lr1e5' ...
    loss_name   TEXT    NOT NULL,
    learning_rate REAL,
    lora_rank   INTEGER,
    step        INTEGER NOT NULL,
    loss        REAL    NOT NULL,
    accept_weight REAL,             -- EBE-specific
    split       TEXT    NOT NULL DEFAULT 'train'   -- 'train' | 'val'
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    # Migrate existing DBs that predate new columns — safe to run repeatedly
    # Migrate runs table
    for col, typedef, post_sql in [
        ("task_score",        "REAL", None),
        ("perplexity",        "REAL", None),
        ("draft_latency_ms",  "REAL", None),
        ("verify_latency_ms", "REAL", None),
        ("experiment_tag",    "TEXT",
         "CREATE INDEX IF NOT EXISTS idx_runs_etag ON runs(experiment_tag)"),
        ("hw_tier",           "TEXT NOT NULL DEFAULT 'laptop'",
         "CREATE INDEX IF NOT EXISTS idx_runs_hw_tier ON runs(hw_tier)"),
    ]:
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typedef}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        if post_sql:
            try:
                conn.execute(post_sql)
                conn.commit()
            except sqlite3.OperationalError:
                pass
    # Migrate train_curves table
    for col, typedef in [
        ("split", "TEXT NOT NULL DEFAULT 'train'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE train_curves ADD COLUMN {col} {typedef}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    return conn


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def insert_run(row: dict, hw_tier: str = "laptop") -> int:
    """Insert a run row and return its id. Unknown columns are silently dropped.

    hw_tier: 'laptop' | 'colab' | 'a100'  (default 'laptop' for backward compat)
    """
    valid_cols = {
        "run_tag", "ts", "draft_label", "draft_path", "target_path", "loss_name",
        "train_steps", "learning_rate", "lora_rank", "dataset", "n_prompts",
        "mode", "K", "L", "temperature",
        "alpha_mean", "alpha_std", "alpha_ci95",
        "block_eff", "block_eff_std",
        "throughput", "ms_per_tok", "peak_vram_mb",
        "task_score", "perplexity", "draft_latency_ms", "verify_latency_ms", "notes",
        "experiment_tag", "hw_tier",
    }
    # Inject hw_tier if not already set by the caller
    if "hw_tier" not in row:
        row = {**row, "hw_tier": hw_tier}
    row = {k: v for k, v in row.items() if k in valid_cols}
    if "ts" not in row:
        row["ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cols = list(row.keys())
    conn = _connect()
    cur = conn.execute(
        f"INSERT INTO runs ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
        [row[c] for c in cols],
    )
    conn.commit()
    run_id = cur.lastrowid
    conn.close()
    return run_id


def insert_per_prompt_batch(rows: list):
    if not rows:
        return
    conn = _connect()
    cols = list(rows[0].keys())
    conn.executemany(
        f"INSERT INTO per_prompt ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
        [[r[c] for c in cols] for r in rows],
    )
    conn.commit()
    conn.close()


def query_runs(filters: dict = None) -> list:
    conn = _connect()
    sql, params = "SELECT * FROM runs", []
    if filters:
        clauses = [f"{k}=?" for k in filters]
        sql += " WHERE " + " AND ".join(clauses)
        params = list(filters.values())
    sql += " ORDER BY ts ASC"
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


def query_per_prompt(run_id: int) -> list:
    conn = _connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM per_prompt WHERE run_id=? ORDER BY prompt_idx", (run_id,)
    ).fetchall()]
    conn.close()
    return rows


def distinct_values(col: str) -> list:
    conn = _connect()
    rows = conn.execute(f"SELECT DISTINCT {col} FROM runs ORDER BY {col}").fetchall()
    conn.close()
    return [r[0] for r in rows if r[0] is not None]


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def insert_train_step(label: str, loss_name: str, step: int, loss: float,
                      learning_rate: float = None, lora_rank: int = None,
                      accept_weight: float = None, split: str = "train"):
    """Log one training or validation step.

    split='train'  → training loss (default)
    split='val'    → validation loss — plotted as a dashed line in the dashboard.
                     If val loss rises after falling, the dashboard flags a red warning.
    """
    conn = _connect()
    conn.execute(
        """INSERT INTO train_curves
           (label, loss_name, learning_rate, lora_rank, step, loss, accept_weight, split)
           VALUES (?,?,?,?,?,?,?,?)""",
        (label, loss_name, learning_rate, lora_rank, step, loss, accept_weight, split),
    )
    conn.commit()
    conn.close()


def query_train_curves(label: str = None) -> list:
    conn = _connect()
    sql = "SELECT * FROM train_curves"
    params = []
    if label:
        sql += " WHERE label=?"
        params = [label]
    sql += " ORDER BY label, step"
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


def dedup_runs() -> int:
    """
    Remove duplicate runs caused by concurrent/repeated pipeline invocations.

    A duplicate is defined as two rows sharing the same
    (draft_label, dataset, mode, K, temperature).  For each group of
    duplicates we keep the row with the highest id (most recently inserted)
    and delete the rest — along with their per_prompt rows (CASCADE).

    Returns the number of rows deleted.
    """
    conn = _connect()
    # Find groups with more than one row
    dups = conn.execute("""
        SELECT draft_label, dataset, mode, K, temperature, COUNT(*) AS cnt, MAX(id) AS keep_id
        FROM runs
        GROUP BY draft_label, dataset, mode, K, temperature
        HAVING cnt > 1
    """).fetchall()

    deleted = 0
    for row in dups:
        # Delete all rows in this group EXCEPT the most-recently inserted one
        n = conn.execute("""
            DELETE FROM runs
            WHERE draft_label=? AND dataset=? AND mode=? AND K=? AND temperature=?
              AND id != ?
        """, (row["draft_label"], row["dataset"], row["mode"],
              row["K"], row["temperature"], row["keep_id"])).rowcount
        deleted += n
        print(f"  dedup: kept id={row['keep_id']}  "
              f"({row['draft_label']} | {row['dataset']} | {row['mode']} "
              f"K={row['K']} T={row['temperature']})  "
              f"deleted {n} duplicate(s)")

    conn.commit()
    conn.close()
    return deleted


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--dedup", action="store_true",
                    help="Remove duplicate run rows (keeps most-recently inserted per config)")
    _a = _p.parse_args()
    _connect()
    print(f"DB initialised at {DB_PATH}")
    if _a.dedup:
        n = dedup_runs()
        print(f"Dedup complete: {n} duplicate row(s) removed.")
