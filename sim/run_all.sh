#!/usr/bin/env bash
# Step-size sweep (40 generations each), then the 2000-generation run with the best sigma.
# Sweep score = benchmark practice + benchmark FFA of the final mean genome (fixed seeds).
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
W=${WORKERS:-24}
for s in 0.1 0.2 0.35; do
  $PY sim/evolve.py --out runs/sweep_s$s --gens 40 --sigma $s --ffa 8 --prac 1 --bench 40 --workers $W > runs/sweep_s$s.log 2>&1
done
BEST=$($PY - <<'PY'
import json
best = max((0.1, 0.2, 0.35), key=lambda s: (lambda r: r["bench_prac"] + r["bench_ffa"])(
    [json.loads(l) for l in open(f"runs/sweep_s{s}/log.jsonl")][-1]))
print(best)
PY
)
echo "best sigma: $BEST"
$PY sim/evolve.py --out runs/main --gens 2000 --sigma $BEST --ffa 8 --prac 2 --bench 25 --workers $W > runs/main.log 2>&1
