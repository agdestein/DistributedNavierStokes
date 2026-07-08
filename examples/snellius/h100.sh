#!/bin/bash
#SBATCH --job-name=dns-smoke-h100
#SBATCH --partition=gpu_h100
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=smoke-h100-%j.out

# Smoke test on 4 full H100s (one node; 16 cores per GPU = fair share).
#
# The device-buffer MPI path currently segfaults on Snellius: the UCX
# 1.18 module ignores UCX_MEMTYPE_CACHE=n (reported as an "unused
# environment variable"), so UCX misclassifies Julia's CUDA allocations
# as host memory and host-memcpys them (crash in ucp_memcpy_pack). Stage
# MPI messages through host mirrors until that is fixed; delete this to
# retest the device path.
export DNS_MPIBUF=host

module load 2025 OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0

# Keep in sync with ~/.bashrc so precompile caches are shared across nodes.
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
# CUDA's stream-ordered memory pool and UCX registration don't always get
# along; the solver allocates nothing in the time loop, so this is free.
export JULIA_CUDA_MEMORY_POOL=none

cd "$SLURM_SUBMIT_DIR"
# Slurm's default here is pmi2, which Open MPI 5 does not support.
srun --mpi=pmix julia --project=examples examples/channel.jl
