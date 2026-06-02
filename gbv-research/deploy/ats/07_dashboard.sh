#!/usr/bin/env bash
# =============================================================================
# 07_dashboard.sh — Start the SpecDist results dashboard locally.
# Reads from db/results.db — updated live as eval runs complete.
#
# Usage:
#   bash deploy/ats/07_dashboard.sh           # default port 5000
#   bash deploy/ats/07_dashboard.sh 8080      # custom port
#
# Access from your laptop (SSH tunnel):
#   ssh -L 5000:localhost:5000 user@ats-server   # then open localhost:5000
#
# Or if ATS server has a public IP / is on VPN:
#   Open http://<ats-server-ip>:5000
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PORT="${1:-5000}"

echo "=============================="
echo "  SpecDist Dashboard"
echo "  DB:   $ROOT/db/results.db"
echo "  Port: $PORT"
echo "=============================="
echo
echo "Access from your laptop:"
echo "  ssh -L $PORT:localhost:$PORT user@<ats-server>  →  open localhost:$PORT"
echo
echo "Press Ctrl+C to stop."
echo

cd "$ROOT"
python dashboard/training_dashboard.py --port "$PORT"
