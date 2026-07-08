# Running on Snellius

One-time setup, on a login node (works without a GPU — the CUDA runtime
version is pinned instead of detected, and Open MPI's CUDA support only
activates on nodes with a driver):

```sh
module load 2025 OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0
julia --project=examples -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
julia --project=examples -e 'using CUDA; CUDA.set_runtime_version!(v"12.8"; local_toolkit = true)'
julia --project=examples -e 'using Pkg; Pkg.resolve(); Pkg.precompile()'
```

This writes `examples/LocalPreferences.toml` (gitignored): MPI.jl uses the
system CUDA-aware Open MPI, and CUDA.jl uses the module's CUDA 12.8 toolkit
(no artifact download, guaranteed to match the MPI build). The job scripts
load the same module — keep the two in sync.

Precompile caches are shared across the differing node architectures via
`JULIA_CPU_TARGET` (set identically in `~/.bashrc` and the job scripts):

```sh
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
```

(znver2: login/thin nodes, znver4: H100 nodes, icelake-server: A100/MIG
nodes.)

Then, from the repository root:

```sh
sbatch examples/snellius/mig.sh    # 4 ranks sharing 1 MIG slice, cheapest
sbatch examples/snellius/a100.sh   # 4 full A100s, cheapest multi-device test
sbatch examples/snellius/h100.sh   # 4 full H100s
```

Note the gpu_mig QOS caps every job at one MIG slice (cpu=9, gpu=1,
node=1), so a multi-slice job is not possible there: the MIG script runs 4
MPI ranks oversubscribed on one slice, which still exercises srun/PMIx,
CUDA-aware MPI, and the device kernels. The cheapest *multi-device* test is
the A100 script (~90 SBU for 15 min).

Gotchas encoded in the scripts:

- `srun --mpi=pmix`: the cluster default is pmi2, which Open MPI 5 does not
  support (ranks would all come up as rank 0 of size 1).
- `--gpus-per-task=1` (A100/H100): gives each rank its own
  `CUDA_VISIBLE_DEVICES` entry, one rank per GPU.
- `DNS_MPIBUF=host` (all scripts, July 2026): passes `mpibuf = :host` to
  `setup` — every MPI message is staged through host mirrors. The
  device-buffer path segfaults in `ucp_memcpy_pack`: the UCX 1.18 module
  ignores `UCX_MEMTYPE_CACHE=n` (visible as an "unused environment
  variables" UCX warning), so buffers allocated by CUDA.jl — which UCX's
  allocation hooks cannot intercept — are misclassified as host memory.
  This crashes on full A100s too, not just MIG. On MIG host staging is
  needed regardless (no CUDA IPC there at all).
- `JULIA_CUDA_MEMORY_POOL=none`: avoids UCX registration issues with the
  stream-ordered pool; costs nothing since the time loop allocates nothing.

The example prints one line per rank with the device name and UUID — on
A100/H100 verify the four UUIDs are distinct — plus the CUDA-awareness and
buffer mode from rank 0.
