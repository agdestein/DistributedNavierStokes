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
load the same module — keep the two in sync. Then append the MIG
workaround (see gotchas below) and precompile once more:

```sh
printf '\n[CUDACore]\nnonblocking_synchronization = false\n' >> examples/LocalPreferences.toml
julia --project=examples -e 'using Pkg; Pkg.precompile()'
```

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
  device-buffer path is broken on this cluster: the UCX 1.18 module cannot
  classify CUDA.jl device memory (its allocation hooks can't intercept
  Julia, and it ignores `UCX_MEMTYPE_CACHE=n` — visible as an "unused
  environment variables" UCX warning), so some host-only transport always
  ends up touching a device pointer. Observed on full A100s and MIG alike:
  eager path segfaults in `ucp_memcpy_pack`; forcing rendezvous
  (`UCX_RNDV_THRESH=0`) passes a toy sendrecv but the solver then dies in
  CMA (`process_vm_readv ... Bad address`); `OMPI_MCA_pml=ob1` and the
  GCC OpenMPI builds crash too. Worth re-testing after SURF updates the
  UCX module.
- `[CUDACore] nonblocking_synchronization = false` in
  `examples/LocalPreferences.toml` (CUDACore is in the project `[extras]`
  so the preference resolves): CUDA.jl's nonblocking-synchronization
  worker thread segfaults on MIG slices after a few hundred steps.
- `JULIA_CUDA_MEMORY_POOL=none`: avoids UCX registration issues with the
  stream-ordered pool; costs nothing since the time loop allocates nothing.

The example prints one line per rank with the device name and UUID — on
A100/H100 verify the four UUIDs are distinct — plus the CUDA-awareness and
buffer mode from rank 0.
