# Training Dashboard

Live web UI for monitoring SpecDist experiments.

## Launch

```bash
# Run from gbv-research/
python dashboard/training_dashboard.py              # http://127.0.0.1:5000
python dashboard/training_dashboard.py --port 8080  # custom port
```

The dashboard can be started before, during, or after a pipeline run.
It reads files, never writes them.

## What it shows

- **Block efficiency table** — rows = trained models (baseline/kl/ebe/rev_kl/jsd/l1/online),
  cols = verifier modes (alpha/bv/gbv/traversal/specinfer/naive)
- **Training loss curves** — per-model loss vs step from W&B / results.db
- **Per-prompt breakdown** — acceptance rate per prompt for any run
- **Pipeline progress** — live step status (Phase 1→4), shows done/running/failed/pending
- **Live log tail** — streams `db/logs/pipeline_output.log` (training output, step boundaries)
- **Eval progress** — streams `db/logs/be_progress.log` (per-verifier-mode progress during eval)
- **GPU status** — live VRAM free/total via PyTorch

## Data sources (all read-only)

| Source | What it contains |
|--------|-----------------|
| `db/results.db` | All eval results — written by `orchestration/run_all.py` |
| `db/logs/pipeline_output.log` | Full pipeline log — training steps, val loss, step boundaries |
| `db/logs/be_progress.log` | Per-prompt block efficiency progress during eval steps |
| `orchestration/pipeline_state_*.json` | Step statuses (done/running/pending/failed) |

All paths are resolved relative to `gbv-research/` regardless of where
`training_dashboard.py` is launched from.

## API endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /api/runs` | All eval rows from results.db (supports `?col=val` filters) |
| `GET /api/dimensions` | Distinct values per column (for filter dropdowns) |
| `GET /api/per_prompt/<id>` | Per-prompt data for one run |
| `GET /api/train_curves?label=kl` | Training loss curve for a model |
| `GET /api/pipeline_status` | Live step statuses + recent DB entries |
| `GET /api/log_tail?source=pipeline&lines=80` | Last N lines of pipeline log |
| `GET /api/log_tail?source=eval&lines=80` | Last N lines of eval progress log |
| `GET /api/gpu_status` | Current VRAM usage |
