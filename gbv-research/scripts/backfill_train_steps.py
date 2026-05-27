"""
backfill_train_steps.py
=======================
One-time migration: set train_steps on existing eval_runs rows that were
recorded before evaluate.py had the --train_steps argument.

Rules (matching experiment.py _ec() logic):
  - draft_label == "baseline"                      → train_steps = 0  (already correct)
  - draft_label in online_labels, n_prompts <= 5   → train_steps = 50  (smoke run)
  - draft_label in online_labels, n_prompts >  5   → train_steps = 500
  - everything else, n_prompts <= 5                → train_steps = 50  (smoke run)
  - everything else, n_prompts >  5                → train_steps = 1000

Run once:
    python scripts/backfill_train_steps.py [--db db/results.db] [--dry-run]
"""

import argparse
import sqlite3
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GBV  = os.path.dirname(HERE)
DEFAULT_DB = os.path.join(GBV, "db", "results.db")

ONLINE_LABELS = {"online", "online_ebe", "online_ebe_single"}

def backfill(db_path: str, dry_run: bool = False) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    cur.execute("SELECT id, draft_label, n_prompts, train_steps FROM runs")
    rows = cur.fetchall()

    updates: list[tuple[int, int, int]] = []  # (new_steps, row_id, old_steps)
    skipped = 0

    for row in rows:
        label      = row["draft_label"] or ""
        n_prompts  = row["n_prompts"] or 0
        old_steps  = row["train_steps"] or 0

        if label == "baseline":
            skipped += 1
            continue   # correct already

        if label in ONLINE_LABELS:
            new_steps = 50 if n_prompts <= 5 else 500
        else:
            new_steps = 50 if n_prompts <= 5 else 1000

        if new_steps == old_steps:
            skipped += 1
            continue

        updates.append((new_steps, row["id"], old_steps))

    print(f"Rows to update : {len(updates)}")
    print(f"Rows unchanged : {skipped}")

    # Preview
    by_label: dict[str, dict] = {}
    for new_steps, row_id, old_steps in updates:
        cur.execute("SELECT draft_label, n_prompts FROM runs WHERE id=?", (row_id,))
        r = cur.fetchone()
        key = f"{r['draft_label']} (n={r['n_prompts']})"
        if key not in by_label:
            by_label[key] = {"old": old_steps, "new": new_steps, "count": 0}
        by_label[key]["count"] += 1

    print("\nSummary of changes:")
    for key, info in sorted(by_label.items()):
        print(f"  {key:<35} {info['old']:>6} -> {info['new']:>6}  ({info['count']} rows)")

    if dry_run:
        print("\n[dry-run] No changes written.")
        conn.close()
        return

    for new_steps, row_id, _ in updates:
        cur.execute("UPDATE runs SET train_steps=? WHERE id=?", (new_steps, row_id))

    conn.commit()
    conn.close()
    print(f"\n[ok] Updated {len(updates)} rows in {db_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db",      default=DEFAULT_DB, help="Path to results.db")
    p.add_argument("--dry-run", action="store_true", help="Preview changes, do not write")
    args = p.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: DB not found: {args.db}")
        raise SystemExit(1)

    backfill(args.db, dry_run=args.dry_run)
