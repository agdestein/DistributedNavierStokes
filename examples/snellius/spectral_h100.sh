#!/bin/bash
#SBATCH --job-name=hit-h100
#SBATCH --partition=gpu_h100
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=hit-h100-%j.out

# Spectral forced-HIT with checkpoint/restart across SLURM time limits.
#
# SLURM sends USR1 to this batch shell ("B:" = the shell only, never the
# MPI ranks) 900 s before the limit; the trap touches $DNS_OUTDIR/stop.
# The solver's checkpointer polls that file (rank 0, decision broadcast),
# writes a checkpoint, and spectral_solve! returns cleanly; the script then
# resubmits itself and the next job resumes from the newest checkpoint
# (spectral_latest in examples/spectral_hit.jl). All signal/stop
# coordination goes through the shared filesystem — no signal handling in
# Julia, no signals to MPI ranks mid-collective.

# NCCL transpose transport: 5-11x faster steps than host-staged MPI
# (CAMPAIGN.md S0). Needs the NCCL_jll stub dev'd (snellius/README.md)
# and job-level GPU allocation (--gpus=4, not --gpus-per-task: NCCL P2P
# needs every device visible to every rank; ranks pick their device by
# node-local rank). Fall back with DNS_MPIBUF=host if NCCL misbehaves.
export DNS_MPIBUF=${DNS_MPIBUF:-nccl}
export DNS_OUTDIR=${DNS_OUTDIR:-output-hit}

module load 2025 OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0

# Keep in sync with ~/.bashrc so precompile caches are shared across nodes.
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
export JULIA_CUDA_MEMORY_POOL=none

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$DNS_OUTDIR"
rm -f "$DNS_OUTDIR/stop"

trap 'echo "USR1: requesting checkpoint + graceful stop"; touch "$DNS_OUTDIR/stop"' USR1

# srun must run in the background so the trap can fire; the first `wait` is
# interrupted by the signal (status > 128), the second collects srun itself.
srun --mpi=pmix julia --project=examples examples/spectral_hit.jl &
pid=$!
wait $pid
status=$?
((status > 128)) && wait $pid

# Stopped by the time-limit signal → continue in a fresh job. A natural
# finish (or a crash) does not resubmit.
if [[ -f "$DNS_OUTDIR/stop" ]]; then
    echo "resubmitting to continue from the newest checkpoint"
    sbatch "$0"
fi
