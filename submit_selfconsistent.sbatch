#!/bin/bash
#SBATCH --job-name=scalar-selfconsistent
#SBATCH --account=daniel_lab
#SBATCH --partition=free                 # no SU charge, but preemptible (PreemptMode=CANCEL, no grace) - see chain notes below
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16               # one Distributed worker per CPU; ~5 s/point since the Euclidean fix
#SBATCH --mem-per-cpu=2G                 # workers peaked at ~1.3 GB each in the 16-point run
#SBATCH --time=03:55:00                  # <-- keep this in sync with WALL_SECONDS below
#SBATCH --output=logs/scalar_selfconsistent_%j.out
#SBATCH --error=logs/scalar_selfconsistent_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# This script resubmits itself (via an explicit absolute path, not $0 - see
# note below) until the Julia driver reports convergence or hits max_iter.
# Each run picks up exactly where the last one left off, using checkpoints
# in $CKPT_DIR - see run_selfconsistent_checkpointed.jl.
# ---------------------------------------------------------------------------

# Must match --time above, in seconds, so the Julia driver knows its own
# wall-clock budget and can checkpoint + exit before SLURM kills it.
export SBATCH_WALL_SECONDS=$((3 * 3600 + 55 * 60))
# Stop starting new condensate points this long before the wall limit, so
# in-flight ones (slowest ~35 min) can finish and be checkpointed.
export SC_SAFETY_BUFFER=2400

# Hardcoded absolute path to THIS script, used for resubmission. Slurm does
# not reliably preserve $0 as the original submitted path inside a running
# job, so referencing it directly here is more robust than "sbatch $0".
SCRIPT_PATH="/data/homezvol0/lcondren/COFI_neutrino/submit_selfconsistent.sbatch"

PROJECT_DIR="/data/homezvol0/lcondren/COFI_neutrino"
CKPT_DIR="$PROJECT_DIR/checkpoints"
cd "$PROJECT_DIR"
mkdir -p "$CKPT_DIR" logs

# Safety cap: if something is broken (e.g. a persistent bug), stop the chain
# after this many resubmissions instead of silently burning allocation forever.
# The count is reset to 0 whenever a run's Julia driver exits cleanly (see the
# end of this script), so it only counts CONSECUTIVE runs that crashed or were
# killed (e.g. preempted on the free partition) - routine preemptions don't
# eat into the cap as long as runs in between make progress.
MAX_RESUBMISSIONS=50
COUNT_FILE="$CKPT_DIR/resubmit_count.txt"
COUNT=$( [ -f "$COUNT_FILE" ] && cat "$COUNT_FILE" || echo 0 )

if [ -f "$CKPT_DIR/CONVERGED" ] || [ -f "$CKPT_DIR/MAX_ITER_REACHED" ]; then
    echo "Already finished (CONVERGED or MAX_ITER_REACHED present). Not resubmitting."
    exit 0
fi

if [ "$COUNT" -ge "$MAX_RESUBMISSIONS" ]; then
    echo "Hit safety cap of $MAX_RESUBMISSIONS resubmissions without convergence."
    echo "Stopping the chain - check $CKPT_DIR for the latest state; something may be stuck."
    exit 1
fi
echo $((COUNT + 1)) > "$COUNT_FILE"

# Queue the next link in the chain BEFORE doing any work. --dependency=afterany
# means the next job starts once this one ends, whether it exits cleanly,
# hits the wall-time limit, or is killed - so the chain survives an unclean
# kill, not just a clean exit. This is also what handles preemption on the
# free partition: HPC3 cancels (not requeues) preempted jobs with no grace
# period, which satisfies afterany and lets this follow-up start.
NEXT_JOBID=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH")
echo "Queued follow-up job $NEXT_JOBID (dependency: afterany:$SLURM_JOB_ID)"
# Lets jobs waiting on this chain (submit_condensate_final.sbatch) find its newest link.
echo "$NEXT_JOBID" > "$CKPT_DIR/chain_next_jobid.txt"

module load julia/1.12.4
export JULIA_PROJECT="$PROJECT_DIR"      # inherited by the Distributed worker processes
export JULIA_NUM_THREADS=1               # parallelism is process-based; see the .jl driver
export OPENBLAS_NUM_THREADS=1
# Condensate tabulation points: grid spacing (z1-z0)/(N-1) must resolve phi's
# oscillation period (~6.3 in z); the 16-point run (spacing 6.6) aliased it.
# NOTE: changing this mid-chain orphans existing partial checkpoints.
export SC_N_POINTS="${SC_N_POINTS:-100}"
export SC_MAX_ITER="${SC_MAX_ITER:-20}"
# Condensate quadrature tolerance per point, and relative convergence criterion
# max|Δphi|/max|phi| (overrides the absolute SC_TOL when > 0).
export SC_RTOL="${SC_RTOL:-1e-8}"
export SC_REL_TOL="${SC_REL_TOL:-1e-8}"
# Pass-0 profile: "ansatz" = power law A z^Δ- + B z^Δ+ matched to phi(z0)=1,
# phi'(z0)=0; "homogeneous" = source-off UV solve of the KG equation.
export SC_INIT="${SC_INIT:-ansatz}"

# Install + precompile once in the master before workers load packages
# concurrently (no-op after the first run).
julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

julia run_selfconsistent_checkpointed.jl

# Clean exit (set -e means we only get here if Julia succeeded): reset the
# consecutive-failure counter.
echo 0 > "$COUNT_FILE"

# If that run finished the whole problem, the follow-up job we just queued
# would only wake up, see the flag file, and no-op - cancel it instead so it
# doesn't waste a queue slot / allocation charge.
if [ -f "$CKPT_DIR/CONVERGED" ] || [ -f "$CKPT_DIR/MAX_ITER_REACHED" ]; then
    echo "Finished this run. Cancelling the pre-queued follow-up job $NEXT_JOBID."
    scancel "$NEXT_JOBID" || true
fi
