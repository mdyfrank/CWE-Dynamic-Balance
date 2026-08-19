#!/usr/bin/env bash
# sequential DP driver: one small block at a time, so the run has a checkpoint every ~1h and a crash
# costs one block rather than the sweep. BLAS-bound work must not overlap -- seven concurrent numpy
# jobs made each Bellman sweep 12x slower, which is how the first cost estimate for this run was
# wrong by an order of magnitude.
set -u
cd "$(dirname "$0")/code" || exit 1

while pgrep -f recompute_dynamic_dp.py > /dev/null 2>&1; do sleep 5; done
echo "=== DP start $(date -u +%H:%M:%S) ==="

for lo in 70000 70005 70010 70015 70020 70025 70030 70035 70040 70045 70050 70055; do
  hi=$((lo + 4))
  echo "=== BLOCK ${lo}-${hi} start $(date -u +%H:%M:%S) ==="
  python -u ha_dynamic_dp.py --seeds "${lo}-${hi}" --policies P_SB_uniform --deltas 0.95 \
      --out "../results/solver/_dp_shard_main_${lo}.json" \
    || { echo "=== BLOCK ${lo}-${hi} FAILED ==="; exit 1; }
  echo "=== BLOCK ${lo}-${hi} done $(date -u +%H:%M:%S) ==="
done

echo "=== DELTA NEIGHBOURS start $(date -u +%H:%M:%S) ==="
python -u ha_dynamic_dp.py --seeds 70000-70004 --policies P_SB_uniform --deltas 0.90,0.99 \
    --no-finite --out ../results/solver/_dp_shard_delta.json || { echo "=== DELTA FAILED ==="; exit 1; }

echo "=== ALL DP DONE $(date -u +%H:%M:%S) ==="
