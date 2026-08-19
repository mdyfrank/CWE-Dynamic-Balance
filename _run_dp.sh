#!/usr/bin/env bash
# sequential DP driver: one small block at a time, so the run has a checkpoint every ~1h and a crash
# costs one block rather than the sweep. BLAS-bound work must not overlap -- seven concurrent numpy
# jobs made each Bellman sweep 12x slower, which is how the first cost estimate for this run was
# wrong by an order of magnitude.
set -u
cd "$(dirname "$0")/code" || exit 1

while pgrep -f recompute_dynamic_dp.py > /dev/null 2>&1; do sleep 5; done
echo "=== DP start $(date -u +%H:%M:%S) ==="

# Restartable by construction: a block whose shard is already on disk is skipped, so re-running
# after a crash costs nothing and cannot half-overwrite a good shard. The first run needed this --
# it aborted on a cosmetic `relative_to` in the closing log line, AFTER the block had been written.
for lo in 70000 70005 70010 70015 70020 70025 70030 70035 70040 70045 70050 70055; do
  hi=$((lo + 4))
  shard="../results/solver/_dp_shard_main_${lo}.json"
  if [ -s "$shard" ]; then echo "=== BLOCK ${lo}-${hi} already on disk, skipped ==="; continue; fi
  echo "=== BLOCK ${lo}-${hi} start $(date -u +%H:%M:%S) ==="
  python -u ha_dynamic_dp.py --seeds "${lo}-${hi}" --policies P_SB_uniform --deltas 0.95 \
      --out "$shard" || { echo "=== BLOCK ${lo}-${hi} FAILED ==="; exit 1; }
  echo "=== BLOCK ${lo}-${hi} done $(date -u +%H:%M:%S) ==="
done

echo "=== DELTA NEIGHBOURS start $(date -u +%H:%M:%S) ==="
if [ -s ../results/solver/_dp_shard_delta.json ]; then
  echo "=== DELTA already on disk, skipped ==="
else
  python -u ha_dynamic_dp.py --seeds 70000-70004 --policies P_SB_uniform --deltas 0.90,0.99 \
      --no-finite --out ../results/solver/_dp_shard_delta.json \
    || { echo "=== DELTA FAILED ==="; exit 1; }
fi

echo "=== ALL DP DONE $(date -u +%H:%M:%S) ==="
