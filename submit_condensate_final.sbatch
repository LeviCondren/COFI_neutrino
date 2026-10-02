#!/bin/bash
#SBATCH --job-name=condensate-final
#SBATCH --account=daniel_lab
#SBATCH --partition=free                 # no SU charge, but preemptible - see requeue notes below
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2                # condensate_final.jl is serial
#SBATCH --mem=8G
#SBATCH --time=03:55:00
#SBATCH --output=logs/condensate_final_%j.out
#SBATCH --error=logs/condensate_final_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Computes the exact condensate profile for the FINAL phi of the
# self-consistent chain (submit_selfconsistent.sbatch) via condensate_final.jl.
#
# Submit it with --dependency=afterany:<a chain job>. When it starts:
#   - chain not finished yet -> re-queue itself behind the chain's newest
#     link (recorded in chain_next_jobid.txt) and exit;
#   - chain finished -> pre-queue a copy of itself (afterany) so a preemption
#     doesn't lose the run, compute, then cancel that copy.
# ---------------------------------------------------------------------------

SCRIPT_PATH="/data/homezvol0/lcondren/COFI_neutrino/submit_condensate_final.sbatch"
PROJECT_DIR="/data/homezvol0/lcondren/COFI_neutrino"
CKPT_DIR="$PROJECT_DIR/checkpoints"
cd "$PROJECT_DIR"
mkdir -p logs

if [ -f "$CKPT_DIR/condensate_final.csv" ]; then
    echo "condensate_final.csv already exists. Nothing to do."
    exit 0
fi

if [ ! -f "$CKPT_DIR/CONVERGED" ] && [ ! -f "$CKPT_DIR/MAX_ITER_REACHED" ]; then
    NEXT=$(cat "$CKPT_DIR/chain_next_jobid.txt" 2>/dev/null || true)
    if [ -n "$NEXT" ] && squeue -h -j "$NEXT" -o %T 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING"; then
        WAIT_ID=$(sbatch --parsable --dependency=afterany:"$NEXT" "$SCRIPT_PATH")
        echo "Self-consistent chain still running (link $NEXT); re-queued as $WAIT_ID."
        exit 0
    fi
    echo "Chain has not finished and has no live link to wait on - giving up."
    exit 1
fi

# Safety cap on consecutive preemptions/crashes.
COUNT_FILE="$CKPT_DIR/condensate_final_attempts.txt"
COUNT=$( [ -f "$COUNT_FILE" ] && cat "$COUNT_FILE" || echo 0 )
if [ "$COUNT" -ge 20 ]; then
    echo "20 attempts without finishing - stopping."
    exit 1
fi
echo $((COUNT + 1)) > "$COUNT_FILE"

RETRY_ID=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" "$SCRIPT_PATH")
echo "Queued retry job $RETRY_ID in case this one is preempted."

module load julia/1.12.4
export JULIA_PROJECT="$PROJECT_DIR"
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SC_CKPT_DIR="$CKPT_DIR"
export CF_N_Z="${CF_N_Z:-1000}"          # z points in the output profile
export CF_RTOL="${CF_RTOL:-1e-8}"        # relative quadrature tolerance at every z

julia condensate_final.jl

rm -f "$COUNT_FILE"
scancel "$RETRY_ID" || true
echo "Done; cancelled retry job $RETRY_ID."
