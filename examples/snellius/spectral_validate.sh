#!/bin/bash
#SBATCH --job-name=spec-val
#SBATCH --partition=gpu_h100
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=spec-val-%j.out

# Multi-GPU validation: run spectral_validate.jl twice in one allocation —
# 4 ranks on 4 H100s, then 1 rank on 1 H100 — and compare the two
# rank-count-independent snapshot files byte-by-byte as complex numbers.
# Expect agreement to roundoff (~1e-14 relative); the per-rank UUID lines
# in the 4-rank block must show four distinct GPUs.

# Buffer mode as a positional argument (host is the default; "nccl"
# validates the NCCL transpose transport — Snellius sbatch does not
# propagate the submission environment, so an argument, not an env var):
#   sbatch examples/snellius/spectral_validate.sh nccl
export DNS_MPIBUF=${1:-host}
module load 2025 OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0

# Keep in sync with ~/.bashrc so precompile caches are shared across nodes.
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
export JULIA_CUDA_MEMORY_POOL=none

cd "$SLURM_SUBMIT_DIR"
rm -rf output-val1 output-val4

echo "=== 4 ranks, 4 GPUs ==="
DNS_OUTDIR=output-val4 srun --mpi=pmix --ntasks=4 \
    julia --project=examples examples/snellius/spectral_validate.jl

echo "=== 1 rank, 1 GPU ==="
DNS_OUTDIR=output-val1 srun --mpi=pmix --ntasks=1 \
    julia --project=examples examples/snellius/spectral_validate.jl

echo "=== compare ==="
julia -e '
    a = reinterpret(ComplexF64, read("output-val4/final.bin"))
    b = reinterpret(ComplexF64, read("output-val1/final.bin"))
    @assert length(a) == length(b)
    d = maximum(abs, a - b)
    m = maximum(abs, b)
    println("modes = ", length(a), "  max |coef| = ", m)
    println("4-GPU vs 1-GPU: max abs diff = ", d, "  rel = ", d / m)
    println(d / m < 1e-12 ? "VALIDATION PASSED" : "VALIDATION FAILED")
'
