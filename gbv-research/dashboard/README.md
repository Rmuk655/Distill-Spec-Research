# Training Dashboard

Live web UI for monitoring SpecDist experiments.

```
python dashboard/training_dashboard.py        # http://localhost:5000
python dashboard/training_dashboard.py --port 8080
```

**What it shows:**
- Block efficiency heatmap — rows = trained models, cols = verifiers (GBV / traversal / specinfer / alpha)
- Training loss curves per model
- Per-prompt acceptance-rate breakdown
- Live pipeline log tail
- HW-tier filter (laptop / colab / a100) to compare experiments across machines

**Data sources (all read-only):**
- `db/results.db` — evaluation results written by `orchestration/run_all.py`
- `db/logs/pipeline_output.log` — live pipeline log
- `db/logs/be_progress.log` — per-prompt GBV progress during eval
