#!/bin/bash
#SBATCH --job-name=scalar-selfconsistent
#SBATCH --account=m2616              # <-- YOUR NERSC charge account. Find it with:
                                          #     sacctmgr show associations user=$USER format=account
                                          #     or check https://iris.nersc.gov
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12               # matches SC_N_POINTS default (12) - full batch parallelism in one shot
#SBATCH --mem=16G
#SBATCH --time=04:00:00                  # <-- keep this in sync with WALL_SECONDS below
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
export SBATCH_WALL_SECONDS=$((4 * 3600))

# Hardcoded absolute path to THIS script, used for resubmission. Slurm does
# not reliably preserve $0 as the original submitted path inside a running
# job, so referencing it directly here is more robust than "sbatch $0".
SCRIPT_PATH="/global/u2/l/lcondren/COFI_neutrino/submit_selfconsistent.sbatch"

PROJECT_DIR="/global/u2/l/lcondren/COFI_neutrino"
CKPT_DIR="$PROJECT_DIR/checkpoints"
cd "$PROJECT_DIR"
mkdir -p "$CKPT_DIR" logs

# Safety cap: if something is broken (e.g. a persistent bug), stop the chain
# after this many resubmissions instead of silently burning allocation forever.
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
# kill, not just a clean exit.
NEXT_JOBID=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_PATH")
echo "Queued follow-up job $NEXT_JOBID (dependency: afterany:$SLURM_JOB_ID)"

julia --project=. run_selfconsistent_checkpointed.jl

# If that run finished the whole problem, the follow-up job we just queued
# would only wake up, see the flag file, and no-op - cancel it instead so it
# doesn't waste a queue slot / allocation charge.
if [ -f "$CKPT_DIR/CONVERGED" ] || [ -f "$CKPT_DIR/MAX_ITER_REACHED" ]; then
    echo "Finished this run. Cancelling the pre-queued follow-up job $NEXT_JOBID."
    scancel "$NEXT_JOBID" || true
fi
