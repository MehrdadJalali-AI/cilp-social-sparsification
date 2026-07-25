#!/bin/bash
set -euo pipefail
ROOT="/Users/mehrdadjalali/Documents/SRH_Research/CAILP-Social"
cd "$ROOT"
LOG="$ROOT/results/logs/queue_after_facebook.log"
echo "$(date): waiting for facebook grid PIDs to finish" | tee -a "$LOG"
while pgrep -f "run_full_grid.py --datasets facebook" >/dev/null 2>&1; do
  sleep 120
done
echo "$(date): facebook grid done; starting github + facebook ablations" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python3 scripts/run_full_grid.py \
  --datasets github --seeds 0 1 2 3 4 5 6 7 8 9 \
  --methods random dspar effective_resistance original_ilp neuralsparse ptdnet cailp_a31 cailp_multi \
  2>&1 | tee -a "$ROOT/results/logs/grid_github_all.log" &
GITHUB_PID=$!
PYTHONUNBUFFERED=1 python3 scripts/run_ablations_grid.py \
  --datasets facebook --seeds 0 1 2 --budgets 0.3 0.5 0.7 \
  2>&1 | tee -a "$ROOT/results/logs/ablation_facebook.log" &
ABL_PID=$!
wait $GITHUB_PID
echo "$(date): github grid finished exit=$?" | tee -a "$LOG"
wait $ABL_PID
echo "$(date): facebook ablations finished exit=$?" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python3 scripts/analyze_grid.py 2>&1 | tee -a "$ROOT/results/logs/analyze_final.log"
PYTHONUNBUFFERED=1 python3 scripts/make_grid_tables.py 2>&1 | tee -a "$ROOT/results/logs/analyze_final.log"
echo "$(date): analysis complete" | tee -a "$LOG"
