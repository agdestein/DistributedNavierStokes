#!/bin/bash
#SBATCH --job-name=dns2
#SBATCH --partition=gpu_h100
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=%x-%j.out

# The second-generation campaign runner: R2 ladder + R3 (+ r1, which this
# generalizes — data-campaign.md §2 in ProbabilisticClosure; driver
# examples/spectral_run.jl). One run id = one DNS realization; the params
# table below is the single source of truth for physics and geometry.
#
# Usage, from the repository root on a login node:
#
#   ./examples/snellius/spectral_campaign.sh <runid> [outdir]
#
# The script self-submits via sbatch with the run's --ntasks/--gpus/--time
# and re-executes inside the job. outdir defaults to
# /projects/prjs1757/dns2/<runid> (snapshots in <outdir>/snapshots;
# checkpoints, stats.csv, schedule.toml in <outdir>). A time-limit stop
# checkpoints and resubmits automatically (stop-file pattern); a natural
# finish or a crash does not resubmit. Runs are independent — submit any
# subset concurrently:
#
#   for id in r2a-s11 r2a-s12 r2a-s13 r2a-s14 r2ap-s15 \
#             r2b-s21 r2b-s22 r2b-s23 \
#             r2c-s31 r2c-s32 r2c-s33 r2d-s41 r3-s51; do
#       ./examples/snellius/spectral_campaign.sh $id
#   done
#
# Before every real submission round:
#
#   ./examples/snellius/spectral_campaign.sh dummy      # 64³ end-to-end test
#   DUMMY_GPUS=4 ./examples/snellius/spectral_campaign.sh dummy   # before r3
#
# Run ids (seed = digits after -s; law window = nlaw·t_int/2 after warm-up,
# snapshots every t_int/2; walltimes are estimates, chunked at 12 h):
#
#   r2a-s11 … -s14   384³  ν = 4e-4    1 GPU   20 t_int          ≈ 3 h
#   r2ap-s15         512³  ν = 4e-4    1 GPU   10 t_int          ≈ 5 h  res pair
#   r2b-s21 … -s23   512³  ν = 2.5e-4  1 GPU   20 t_int          ≈ 9 h
#   r2c-s31 … -s33   810³  ν = 1.5e-4  2 GPUs  12 t_int          ≈ 17 h
#   r2d-s41, -s42    972³  ν = 1e-4    2 GPUs  10 t_int          ≈ 34 h  s42 optional
#   r3-s51, -s52    1200³  ν = 8e-5    4 GPUs  5 t_int + 40-snap ≈ 31 h  s52 optional
#                                              pairing window
#   r1               972³  ν = 1e-4, seed 200  (complete 2026-08-16; kept
#                                              reproducible)
#
# Seeds avoid the legacy set {1, 2, 3, 100, 200}. NCCL transport needs the
# NCCL_jll stub dev'd and job-level --gpus (see snellius/README.md);
# DNS_MPIBUF=host is the fallback.

params() {
    local id=$1
    seed=${id##*-s}
    ndense=0
    case $id in
        dummy)       gpus=${DUMMY_GPUS:-1}; hours=1 ;;
        r1)          n=972;  visc=1e-4;   seed=200; nlaw=20; ndense=40; gpus=2; hours=12 ;;
        r2a-s1[1-4]) n=384;  visc=4e-4;   nlaw=40; gpus=1; hours=5 ;;
        r2ap-s15)    n=512;  visc=4e-4;   nlaw=20; gpus=1; hours=8 ;;
        r2b-s2[1-3]) n=512;  visc=2.5e-4; nlaw=40; gpus=1; hours=12 ;;
        r2c-s3[1-3]) n=810;  visc=1.5e-4; nlaw=24; gpus=2; hours=12 ;;
        r2d-s4[12])  n=972;  visc=1e-4;   nlaw=20; gpus=2; hours=12 ;;
        r3-s5[12])   n=1200; visc=8e-5;   nlaw=10; ndense=40; gpus=4; hours=12 ;;
        *) echo "unknown run id: '$id' (see the table in $0)" >&2; exit 2 ;;
    esac
}

id=${1:?usage: $0 <runid> [outdir] -- see the run table in this script}
outdir=$2
params "$id"

# Login node: submit this script with the run's geometry and stop.
if [[ -z $SLURM_JOB_ID ]]; then
    exec sbatch --job-name="$id" --ntasks="$gpus" --gpus="$gpus" \
        --time="${hours}:00:00" "$0" "$@"
fi

# Inside the job from here on.
if [[ $id == dummy ]]; then
    export DNS_DUMMY=1
    export DNS_OUTDIR=${outdir:-output-dummy}
else
    if [[ ${SLURM_NTASKS:-1} -ne $gpus ]]; then
        echo "$id needs $gpus ranks/GPUs but the job has ${SLURM_NTASKS:-1};" \
             "submit via '$0 $id' (or sbatch --ntasks=$gpus --gpus=$gpus)" >&2
        exit 1
    fi
    export DNS_N=$n DNS_VISC=$visc DNS_SEED=$seed
    export DNS_NLAW=$nlaw DNS_NDENSE=$ndense
    export DNS_OUTDIR=${outdir:-/projects/prjs1757/dns2/$id}
fi
export DNS_RUN=$id
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
srun --mpi=pmix julia --project=examples examples/spectral_run.jl &
pid=$!
wait $pid
status=$?
((status > 128)) && wait $pid

# Stopped by the time-limit signal → continue in a fresh job with the same
# geometry and arguments. A natural finish (or a crash) does not resubmit.
if [[ -f "$DNS_OUTDIR/stop" ]]; then
    echo "resubmitting to continue from the newest checkpoint"
    sbatch --job-name="$id" --ntasks="${SLURM_NTASKS}" \
        --gpus="${SLURM_GPUS:-$SLURM_NTASKS}" --time="${hours}:00:00" "$0" "$@"
fi
