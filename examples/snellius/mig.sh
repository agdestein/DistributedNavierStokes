#!/bin/bash
#SBATCH --job-name=dns-smoke-mig
#SBATCH --partition=gpu_mig
#SBATCH --ntasks=4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=smoke-mig-%j.out

# Cheapest cluster smoke test: the gpu_mig QOS caps every job at one MIG
# slice (cpu=9, gpu=1, node=1), so a multi-slice job is not possible —
# instead 4 MPI ranks share the single a100_3g.20gb slice (oversubscribed
# single-device mode). This still exercises srun/PMIx, CUDA-aware MPI, and
# the device kernels. For a real multi-device test use a100.sh or h100.sh.
#
# CUDA IPC is not supported on MIG devices: disable UCX's cuda_ipc
# transport so device buffers stage through the host.
export UCX_TLS=^cuda_ipc

module load 2025 OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0

# Keep in sync with ~/.bashrc so precompile caches are shared across nodes.
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
# CUDA's stream-ordered memory pool and UCX registration don't always get
# along; the solver allocates nothing in the time loop, so this is free.
export JULIA_CUDA_MEMORY_POOL=none

cd "$SLURM_SUBMIT_DIR"
# Slurm's default here is pmi2, which Open MPI 5 does not support.
srun --mpi=pmix julia --project=examples examples/channel.jl
