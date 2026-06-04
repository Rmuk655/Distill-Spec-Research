#!/usr/bin/env bash
# =============================================================================
# rerun_loss.sh — Re-train ONE loss and add new eval results for comparison.
#
# Usage:
#   bash deploy/rerun_loss.sh <loss> <tag> [config]
#
# Examples:
#   bash deploy/rerun_loss.sh kl          lora_r32_test
#   bash deploy/rerun_loss.sh kl_tree     new_tree_algo        a100_qwen
#   bash deploy/rerun_loss.sh traversal   lr_1e4_sweep
#
# What it does:
#   1. Deletes the checkpoint for this loss (forces re-training with current code)
#   2. KEEPS all old DB rows — old results remain for comparison
#   3. Resets only train+merge+eval steps for this loss
#   4. Re-runs with --force_eval so new rows are ADDED (not skipped)
#
# Old vs new results are distinguished by:
#   - run_tag (different timestamp)
#   - experiment_tag (the <tag> argument you pass)
#
# Compare in W&B: filter by experiment_tag
# Compare in dashboard: All Runs tab, sort by run_tag
# =============================================================================

set -euo pipefail

LOSS="${1:-}"
TAG="${2:-rerun_$(date +%Y%m%d_%H%M)}"
CONFIG="${3:-a100_qwen}"

if [ -z "${LOSS}" ]; then
    echo "Usage: bash deploy/rerun_loss.sh <loss> <tag> [config]"
    echo ""
    echo "Examples:"
    echo "  bash deploy/rerun_loss.sh kl          lora_r32_test"
    echo "  bash deploy/rerun_loss.sh kl_tree     new_tree_algo"
    echo "  bash deploy/rerun_loss.sh traversal   lr_sweep_1e4"
    echo ""
    echo "Loss names: kl, rev_kl, jsd, l1, ebe, ebe_single,"
    echo "            kl_tree, rev_kl_tree, jsd_tree,"
    echo "            bv_tree, gbv_tree, traversal_tree,"
    echo "            naive_tree, nss_tree, specinfer_tree, spectr_tree, khisti_tree"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GBV_DIR="$(dirname "$SCRIPT_DIR")"
[ -f "${HOME}/.specdist_env" ] && source "${HOME}/.specdist_env"
STORAGE="${STORAGE_ROOT:-${GBV_DIR}/db}"

# Derive pair tag from config YAML (same logic as _run_tag in experiment.py)
PAIR_TAG=$(python - <<PYEOF 2>/dev/null
import yaml, re, os, sys
sys.path.insert(0, '${GBV_DIR}/orchestration')
try:
    def _dm(p,c):
        r=dict(p)
        for k,v in c.items():
            if k=='_base': continue
            r[k]=_dm(r[k],v) if isinstance(v,dict) and isinstance(r.get(k),dict) else v
        return r
    def _lr(n, seen=None):
        if seen is None: seen=set()
        if n in seen: return {}
        seen=seen|{n}
        p=os.path.join('${GBV_DIR}/orchestration/configs',f'{n}.yaml')
        if not os.path.exists(p): return {}
        with open(p) as f: raw=yaml.safe_load(f) or {}
        base=raw.pop('_base',None)
        if base: return _dm(_lr(str(base),seen),raw)
        return raw
    raw=_lr('${CONFIG}')
    m=raw.get('models',{}); h=raw.get('hardware',{})
    d,t,q=m.get('draft',''),m.get('target',''),bool(h.get('load_in_4bit',False))
    _F=[('qwen','q'),('llama','l'),('gemma','g'),('mistral','m'),('phi','p')]
    def s(mid):
        name=mid.split('/')[-1].lower()
        if 'distilgpt2' in name: return 'dg2'
        if 'gpt2' in name:
            z='m' if 'medium' in name else ''
            return 'g2'+z
        zm=re.search(r'(\d+\.?\d*)\s*([bm])',name,re.I); sz=''
        if zm:
            nu,u=zm.group(1),zm.group(2).lower()
            if '.' in nu: nu=nu.rstrip('0').rstrip('.')
            sz=nu+u
        for k,p in _F:
            if k in name: return p+sz
        return name[:6]+sz
    print(f'{s(d)}-{s(t)}'+('nf4' if q else ''))
except Exception as e:
    print('unknown', file=__import__('sys').stderr)
    print('q0.6b-q8b')
PYEOF
)

STATE_FILE="${STORAGE}/pipeline_state_${CONFIG}-${PAIR_TAG}.json"
CKPT_BASE="${STORAGE}/checkpoints/${LOSS}-gsm8k-${PAIR_TAG}"

echo "========================================================================"
echo "  Rerun loss: ${LOSS}"
echo "  Tag:        ${TAG}   (use this in W&B / dashboard to find new results)"
echo "  Config:     ${CONFIG} (pair: ${PAIR_TAG})"
echo "  Storage:    ${STORAGE}"
echo "  Old rows:   KEPT for comparison"
echo "  New rows:   added alongside old (different run_tag)"
echo "========================================================================"
echo ""

# ── Step 1: Delete checkpoint (forces retraining) ────────────────────────────
echo "[1/3] Deleting checkpoint..."
if [ -d "${CKPT_BASE}" ] || [ -d "${CKPT_BASE}_merged" ]; then
    rm -rf "${CKPT_BASE}" "${CKPT_BASE}_merged"
    echo "      Deleted: ${CKPT_BASE}[_merged]"
else
    echo "      (no checkpoint found — will train from scratch)"
fi

# ── Step 2: Reset only this loss's state steps ───────────────────────────────
echo "[2/3] Resetting pipeline state for loss '${LOSS}'..."
python - <<PYEOF
import json, os, sys
LOSS = "${LOSS}"
STATE = "${STATE_FILE}"
if not os.path.exists(STATE):
    print(f"  State file not found: {STATE}")
    print(f"  (it will be created fresh on first run)")
    sys.exit(0)
with open(STATE) as f: s = json.load(f)
# Match steps that belong to this specific loss (not other losses with similar names)
targets = [
    f"train_{LOSS}_gsm8k",
    f"merge_{LOSS}_gsm8k",
    f"eval_{LOSS}_gsm8k",
]
reset_count = 0
for k in targets:
    if k in s['steps']:
        old = s['steps'][k]['status']
        s['steps'][k]['status'] = 'pending'
        print(f"  Reset: {k}  ({old} → pending)")
        reset_count += 1
    else:
        print(f"  Skip:  {k}  (not in state file)")
if reset_count == 0:
    print(f"  WARNING: No steps reset. Loss name '{LOSS}' may not match step names.")
    print(f"  Steps available: {[k for k in s['steps'].keys()][:8]}...")
with open(STATE, 'w') as f: json.dump(s, f, indent=2)
print(f"  {reset_count} steps reset to pending.")
PYEOF

# ── Step 3: Re-run with force_eval (adds new rows, keeps old) ────────────────
echo ""
echo "[3/3] Re-running loss '${LOSS}' with experiment_tag='${TAG}'..."
echo "      --force_eval: new eval rows added even if old ones exist"
echo "      Old DB rows for ${LOSS}: PRESERVED for comparison"
echo ""
python "${GBV_DIR}/deploy/aip_run.py" \
    --config "${CONFIG}" \
    --losses "${LOSS}" \
    --no_smoke \
    --force_eval \
    --experiment_tag "${TAG}"
