#!/bin/bash
#SBATCH --job-name=r1-dns
#SBATCH --partition=gpu_h100
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=r1-%j.out

# R1 — the ν = 1e-4 resolution-check DNS (data-campaign.md §2; driver
# examples/spectral_r1.jl). Default 972³ on 2 H100s (92.7 GB/GPU — the
# measured tight-but-verified fit; ≈ 11k SBU over ~30 h wall, so ~3 chunks
# of 12 h with automatic resubmission via the stop-file pattern — see
# spectral_h100.sh for the mechanism). For the roomier 1080³ variant:
#
#   sbatch --ntasks=4 --gpus=4 examples/snellius/spectral_r1.sh 1080
#
# The argument is positional because sbatch does not forward submission
# env to the job:
#
#   $1 = transform grid n (default 972), or "dummy" for a tiny end-to-end
#        pipeline test (64³, short windows — run this first, every time)
#
# NCCL transport needs the NCCL_jll stub dev'd and job-level --gpus (see
# snellius/README.md); DNS_MPIBUF=host is the fallback.

mode=${1:-972}
if [[ $mode == dummy ]]; then
    export DNS_DUMMY=1
    export DNS_OUTDIR=${DNS_OUTDIR:-output-r1-dummy}
else
    export DNS_N=$mode
    export DNS_OUTDIR=${DNS_OUTDIR:-output-r1}
fi
export DNS_MPIBUF=${DNS_MPIBUF:-nccl}

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
srun --mpi=pmix julia --project=examples examples/spectral_r1.jl &
pid=$!
wait $pid
status=$?
((status > 128)) && wait $pid

# Stopped by the time-limit signal → continue in a fresh job with the same
# geometry and argument. A natural finish (or a crash) does not resubmit.
if [[ -f "$DNS_OUTDIR/stop" ]]; then
    echo "resubmitting to continue from the newest checkpoint"
    sbatch --ntasks="${SLURM_NTASKS}" --gpus="${SLURM_GPUS:-$SLURM_NTASKS}" "$0" "$@"
fi
