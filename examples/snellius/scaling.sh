#!/bin/bash
#SBATCH --job-name=s0-scaling
#SBATCH --partition=gpu_h100
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=s0-scaling-%j.out

# S0 scaling experiment (CAMPAIGN.md): one srun per configuration, each a
# fresh Julia process running examples/snellius/scaling.jl. Every rank
# count runs inside this one 4-GPU allocation.
#
# Toggle (positional argument — Snellius sbatch does not propagate the
# submission environment): "dummy" runs the whole pipeline (all legs:
# plateau row, multi-rank, procgrid variant, save, sfs, device smoke,
# failure capture) at tiny sizes in ~10 min — submit that first to catch
# errors cheaply:
#   sbatch --time=00:20:00 examples/snellius/scaling.sh dummy
# Then the full matrix (~20 min measured):
#   sbatch examples/snellius/scaling.sh
# Any other argument is a file of extra config rows to run instead.

export DNS_MPIBUF=host
module load 2025 OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
export JULIA_CUDA_MEMORY_POOL=none

cd "$SLURM_SUBMIT_DIR"
export S0_RESULTS="s0-results-${SLURM_JOB_ID}.toml"
export S0_SAVEDIR="/scratch-shared/$USER/s0"
mkdir -p "$S0_SAVEDIR"

# "<ranks> <n> <nsteps> [PxQ] [save] [sfs] [device]" per row.
MODE="${1:-full}"
if [[ -f "$MODE" ]]; then
    mapfile -t CONFIGS < <(grep -v '^\s*\(#\|$\)' "$MODE")
elif [[ "$MODE" == dummy ]]; then
    CONFIGS=(
        "1 48 5"
        "2 48 5"
        "4 48 5 2x2 save sfs"
        "4 48 5 1x4"
        "4 48 5 auto device"
        "4 6 5 4x1"           # deliberate failure (empty block) → tests failure capture
    )
else
    CONFIGS=(
        # coarse plateau
        "1 48 50"  "1 96 50"  "1 192 50"
        "4 48 50"  "4 96 50"  "4 192 50"
        # size scan, 1 GPU (810 = OOM probe)
        "1 384 20" "1 576 10" "1 810 10"
        # size scan, 2 GPUs (1080 = probe)
        "2 810 10" "2 972 10" "2 1080 10"
        # size scan, 4 GPUs (1296 = probe); save/sfs riders on the anchors
        "4 810 10 2x2 save sfs" "4 1080 10" "4 1200 10 2x2 save" "4 1296 10"
        # procgrid variants at 810³
        "4 810 10 1x4" "4 810 10 4x1"
        # UCX device-path smoke
        "4 192 20 auto device"
    )
fi

for cfg in "${CONFIGS[@]}"; do
    set -- $cfg
    ranks=$1; shift
    echo "=== ranks=$ranks args: $* ==="
    if ! srun --mpi=pmix --ntasks="$ranks" --gpus-per-task=1 --exact \
        julia --project=examples examples/snellius/scaling.jl "$@"; then
        printf '\n[[run]]\nn = %s\nranks = %s\nargs = "%s"\nfailed = true\n' \
            "$1" "$ranks" "$*" >> "$S0_RESULTS"
        echo "=== config failed (recorded) ==="
    fi
done

echo "=== results ==="
cat "$S0_RESULTS"
