#!/bin/bash
#SBATCH --job-name=dns-smoke-mig
#SBATCH --partition=gpu_mig
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:15:00
#SBATCH --output=smoke-mig-%j.out

# Smoke test on 4 MIG slices (a100_3g.20gb, 9 cores each = 1/8 node per
# task). MIG slices are separate CUDA devices *without* peer access or CUDA
# IPC between them, so disable UCX's cuda_ipc transport; device buffers
# then stage through the host, which is fine for correctness testing.
export UCX_TLS=^cuda_ipc

module load 2025 OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0

# Keep in sync with ~/.bashrc so precompile caches are shared across nodes.
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
# CUDA's stream-ordered memory pool and UCX registration don't always get
# along; the solver allocates nothing in the time loop, so this is free.
export JULIA_CUDA_MEMORY_POOL=none

cd "$SLURM_SUBMIT_DIR"
# Slurm's default here is pmi2, which Open MPI 5 does not support.
# --gpus-per-task=1 gives each rank its own CUDA_VISIBLE_DEVICES entry;
# without it all ranks would enumerate the *same* first MIG slice.
srun --mpi=pmix julia --project=examples examples/channel.jl
