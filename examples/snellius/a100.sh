#!/bin/bash
#SBATCH --job-name=dns-smoke-a100
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:15:00
#SBATCH --output=smoke-a100-%j.out

# Smoke test on 4 full A100s (one node; 18 cores per GPU = fair share).
# Full GPUs have peer access, so UCX's cuda_ipc transport stays enabled.

module load 2025 OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0

# Keep in sync with ~/.bashrc so precompile caches are shared across nodes.
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
# CUDA's stream-ordered memory pool and UCX registration don't always get
# along; the solver allocates nothing in the time loop, so this is free.
export JULIA_CUDA_MEMORY_POOL=none

cd "$SLURM_SUBMIT_DIR"
# Slurm's default here is pmi2, which Open MPI 5 does not support.
srun --mpi=pmix julia --project=examples examples/channel.jl
