#!/bin/bash
#SBATCH --job-name=filterbank
#SBATCH --partition=gpu_h100
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=bank-%j.out

# Offline filter bank over stored raw snapshots (data-campaign.md §3;
# driver examples/spectral_filterbank.jl). One pass = one job: the
# collector accumulates every sample in rank-0 host memory and writes
# each cell's fields.jld2 once at the end (no checkpoint/restart), and
# every pass writes its own <outdir>/dns_meta.jld2 — so give each pass
# its own outdir. Submit from the repo root; outdir and snapshot paths
# absolute:
#
#   sbatch examples/snellius/spectral_filterbank.sh <mode> <outdir> <snapshots…>
#
#   mode = full  — the M ≤ 128 bank on every snapshot: legacy Gaussian
#                  Δ/h {2.5, 3.5, 5} (round one's test columns, the
#                  resolution-check comparison) + Δ/η {18, 27, 40, 60} ×
#                  all four kernels on M ∈ {64, 128} (the window rule
#                  drops out-of-window combos)
#          m256  — Δ/η {18, 27, 40} × all four kernels on M = 256
#                  (Float32 cells); run on a decorrelated snapshot
#                  subset (the law-mode times)
#          dummy — tiny end-to-end test on a dummy run's snapshots
#                  (M = 32/42 at the dummy grid's kcut = 21)
#
# η is the mean over ALL sidecars in the snapshot directory (not only
# the snapshots passed), exported as DNS_ETA — so the full and m256
# passes pin matched-Δ/η columns at exactly the same widths.
#
# R1: cd ~/Projects/DistributedNavierStokes &&
#   sbatch examples/snellius/spectral_filterbank.sh full \
#     /projects/prjs1757/dns2/r1/bank /projects/prjs1757/dns2/r1/snapshots/r1_*.bin
#   sbatch examples/snellius/spectral_filterbank.sh m256 \
#     /projects/prjs1757/dns2/r1/bank256 \
#     /projects/prjs1757/dns2/r1/snapshots/r1_0001.bin \
#     /projects/prjs1757/dns2/r1/snapshots/r1_00{42..59}.bin
#
# 2 GPUs is R1's own geometry: the bank fits wherever the run fit
# (sfs_sample reuses the solve's device buffers; the per-cell work and
# the sample accumulation live in rank-0 host memory).

mode=$1
outdir=$2
shift 2

case $mode in
full)
    export DNS_MS=64,128
    export DNS_DETA=18,27,40,60
    export DNS_LEGACY=2.5,3.5,5.0
    ;;
m256)
    export DNS_MS=256
    export DNS_DETA=18,27,40
    export DNS_LEGACY=
    ;;
dummy)
    export DNS_MS=32
    export DNS_DETA=12
    export DNS_LEGACY=2.5,3.5,5.0
    ;;
*)
    echo "unknown mode: $mode (full|m256|dummy)" >&2
    exit 1
    ;;
esac
export DNS_KERNELS=gaussian,cutoff,tophat,helmholtz
export DNS_OUTDIR=$outdir
export DNS_MPIBUF=${DNS_MPIBUF:-nccl}

snapdir=$(dirname "$1")
DNS_ETA=$(grep -h '^eta ' "$snapdir"/*.toml | awk '{s += $3} END {printf "%.10g", s / NR}')
export DNS_ETA

module load 2025 OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0

# Keep in sync with ~/.bashrc so precompile caches are shared across nodes.
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
export JULIA_CUDA_MEMORY_POOL=none

cd "$SLURM_SUBMIT_DIR"
srun --mpi=pmix julia --project=examples examples/spectral_filterbank.jl "$@"
