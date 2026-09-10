#!/bin/bash
# p1_parallel_run_phase4_scalar_n150_20260827.sh
#
# Full staged interval residual pipeline for the phase4 2026-08-27 scalar
# N=150 candidate, with parallel stage 1.
#
# Usage:
#   p1_parallel_run_phase4_scalar_n150_20260827.sh <workdir> <shard_size> <N_grid> <W> [logdir]

set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 <workdir> <shard_size> <N_grid> <W> [logdir]" >&2
  exit 2
fi

WORK=$1
SHARD=$2
N=$3
W=$4
LOGDIR=${5:-$WORK/logs}
RUNTIME=${RUNTIME:-apptainer}
SIF=${SIF:-/home/tb97/transit-cap-ccf/cap-ccf.sif}
BINDPATH=${BINDPATH:-/scratch:/scratch}
WORKER_THREADS=${WORKER_THREADS:-4}
MAIN_THREADS=${MAIN_THREADS:-8}
JL="$RUNTIME exec --bind $BINDPATH $SIF julia --startup-file=no --compiled-modules=no --project=."

mkdir -p "$WORK" "$LOGDIR"
cd "$(dirname "$0")/.."

echo "candidate=phase4_20260827_scalar_N150"
echo "workdir=$WORK"
echo "logdir=$LOGDIR"
echo "shard_size=$SHARD"
echo "N_grid=$N"
echo "workers=$W"
echo "worker_threads=$WORKER_THREADS"
echo "main_threads=$MAIN_THREADS"
echo "julia=$JL"

$JL -e 'using BabenkoRigorous' > "$LOGDIR/warmup.log" 2>&1

pids=()
for i in $(seq 1 "$W"); do
  JULIA_NUM_THREADS=$WORKER_THREADS /usr/bin/time -v \
    $JL scripts/stage_run_phase4_scalar_n150_20260827.jl p1part "$WORK" "$SHARD" "$N" "$i" "$W" \
    > "$LOGDIR/p1part_${i}of${W}.log" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  echo "one or more p1part workers failed; see $LOGDIR" >&2
  exit 1
fi

JULIA_NUM_THREADS=$MAIN_THREADS /usr/bin/time -v \
  $JL scripts/stage_run_phase4_scalar_n150_20260827.jl p1merge "$WORK" "$SHARD" "$N" "$W" \
  > "$LOGDIR/p1merge.log" 2>&1

JULIA_NUM_THREADS=$MAIN_THREADS /usr/bin/time -v \
  $JL scripts/stage_run_phase4_scalar_n150_20260827.jl fft "$WORK" "$SHARD" "$N" \
  > "$LOGDIR/fft.log" 2>&1

JULIA_NUM_THREADS=$MAIN_THREADS /usr/bin/time -v \
  $JL scripts/stage_run_phase4_scalar_n150_20260827.jl s3 "$WORK" "$SHARD" "$N" \
  > "$LOGDIR/s3.log" 2>&1

echo "ALL_STAGES_COMPLETE"
