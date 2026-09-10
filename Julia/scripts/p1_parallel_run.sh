#!/bin/bash
# p1_parallel_run.sh — full staged residual pipeline with parallel stage 1.
#
# Usage: p1_parallel_run.sh <workdir> <shard_size> <N_grid> <W> [logdir]
#
# Launches W concurrent p1part worker processes (contiguous shard chunks),
# waits, then p1merge -> fft -> s3. Every process is wrapped in
# /usr/bin/time -v; per-stage logs go to <logdir>. Peak RAM ~ W x per-worker
# churn (~3-4 GB at shard_size=1024); p1 wall time ~ 1/W.
#
# Environment overrides:
#   SIF          apptainer image   (default: /home/tb97/transit-cap-ccf/cap-ccf.sif)
#   BINDPATH     bind mount        (default: /scratch:/scratch)
#   WORKER_THREADS / MAIN_THREADS  Julia threads per worker / for fft+s3
set -u
WORK=$1; SHARD=$2; N=$3; W=$4
LOGDIR=${5:-$WORK/logs}
SIF=${SIF:-/home/tb97/transit-cap-ccf/cap-ccf.sif}
BINDPATH=${BINDPATH:-/scratch:/scratch}
WORKER_THREADS=${WORKER_THREADS:-4}
MAIN_THREADS=${MAIN_THREADS:-8}
JL="apptainer exec --bind $BINDPATH $SIF julia --project=."

mkdir -p "$WORK" "$LOGDIR"
cd "$(dirname "$0")/.."   # BabenkoRigorous root

# Warm the precompile cache once so W concurrent starts don't stampede.
$JL -e 'using BabenkoRigorous' > "$LOGDIR/warmup.log" 2>&1

for i in $(seq 1 "$W"); do
  JULIA_NUM_THREADS=$WORKER_THREADS /usr/bin/time -v \
    $JL scripts/stage_run.jl p1part "$WORK" "$SHARD" "$N" "$i" "$W" \
    > "$LOGDIR/p1part_${i}of${W}.log" 2>&1 &
done
wait

JULIA_NUM_THREADS=$MAIN_THREADS /usr/bin/time -v \
  $JL scripts/stage_run.jl p1merge "$WORK" "$SHARD" "$N" "$W" \
  > "$LOGDIR/p1merge.log" 2>&1
JULIA_NUM_THREADS=$MAIN_THREADS /usr/bin/time -v \
  $JL scripts/stage_run.jl fft "$WORK" "$SHARD" "$N" \
  > "$LOGDIR/fft.log" 2>&1
JULIA_NUM_THREADS=$MAIN_THREADS /usr/bin/time -v \
  $JL scripts/stage_run.jl s3 "$WORK" "$SHARD" "$N" \
  > "$LOGDIR/s3.log" 2>&1

echo ALL_STAGES_COMPLETE
