# DistributedNavierStokes

*(working title)*

MPI-parallelized, GPU-capable solver for the incompressible Navier-Stokes
equations: symmetry-preserving staggered finite volumes, pencil
decomposition with an FFT/Thomas pressure solve, explicit low-storage RK3.
Targets DNS of channel flow (and periodic boxes) at grid sizes beyond a
single GPU. Companion to
[IncompressibleNavierStokes.jl](https://github.com/agdestein/IncompressibleNavierStokes.jl),
which remains the single-node, differentiable, methods-research code.

See [DESIGN.md](DESIGN.md) for the design document and decision log, and
[CODE_DESIGN.md](CODE_DESIGN.md) for the code-level design (abstractions,
API, communication layer).

## Status

Implemented and tested (CPU backend, multi-rank via mpiexec):

- Pencil/slab layouts, global transposes, halo exchange unified with
  wall/periodic boundary fills (any 2D processor grid, uneven blocks).
- Poisson solve: rfft(x) → fft(z) → batched Thomas in y (walls, stretched
  grids) or Fourier in y (fully periodic); projection is divergence-free to
  machine precision.
- Symmetry-preserving 2nd-order operators (convection skew-symmetric to
  machine precision, also on stretched y grids), Williamson low-storage
  RK3, CFL-adaptive stepping, constant body force.
- Decomposition invariance: identical trajectories (≤ 1e-11) across rank
  counts and processor grids.
- In-situ statistics: plane/time-averaged channel profiles (`channelstats`
  / `channelprofiles`) and shell-binned energy spectra (`spectrumstats` /
  `energyspectrum`, reusing the pencil FFTs); both decomposition-invariant.
- Single-GPU runs (`backend = CUDABackend()`): verified on an RTX 4090,
  matching CPU trajectories to machine precision. Rank-local communication
  uses device copies, so a single GPU rank needs no CUDA-aware MPI.
- Multi-rank GPU over CUDA-aware MPI: verified with 2 and 4 ranks sharing
  one RTX 4090 (system Open MPI 5 + UCX `cuda_ipc`), identical results
  across processor grids. See below for setup.

Not yet: multi-*device* validation on real hardware, 4th-order scheme,
semi-implicit y-diffusion, checkpointing, NCCL transport.
See CODE_DESIGN.md §15.

## Quickstart

```julia
# script.jl — run with: mpiexec -n 4 julia --project script.jl
using MPI, DistributedNavierStokes

s = setup(;
    n = (64, 32, 32),
    lims = ((0.0, 2π), (-1.0, 1.0), (0.0, π)),
    bc = (:periodic, :wall, :periodic),
    stretch = t -> (tanh(1.5 * (2t - 1)) / tanh(1.5) + 1) / 2,
    visc = 1 / 180,
    bodyforce = (1.0, 0.0, 0.0),
    # backend = CUDABackend(),   # after `using CUDA`
)
u = vectorfield(s)
velocityfield!(u, s; x = (x, y, z) -> 1 - y^2)
solve!(; u, setup = s, tlims = (0.0, 10.0), processors = (; log = DistributedNavierStokes.logger(; nupdate = 10)))
```

## Multi-GPU runs (CUDA-aware MPI)

`examples/channel.jl` is a complete multi-rank channel run (CPU or GPU;
picks one device per rank per node, ranks share a device when
oversubscribed). One-time setup pointing MPI.jl at a CUDA-aware system MPI:

```sh
julia --project=examples -e 'using Pkg; Pkg.resolve(); using MPIPreferences; MPIPreferences.use_system_binary()'
mpiexec -n 4 julia --project=examples examples/channel.jl
```

Requirements: an MPI built with CUDA support (`MPI.has_cuda()` must be
`true` *after* `MPI.Init()` — Open MPI 5 loads its CUDA accelerator
component at runtime). Verify with
`julia --project=examples -e 'using MPI; MPI.Init(); println(MPI.has_cuda())'`.

Cluster notes (e.g. Snellius):

- Load a CUDA-aware MPI module and rerun `MPIPreferences.use_system_binary()`
  on the cluster; launch with `srun`/`mpiexec`, one rank per GPU.
- **MIG partitions** (`gpu_mig`): each MIG slice of an A100 appears as its
  own CUDA device, so a few slices give a cheap *correctness* test of the
  multi-device path (one rank per slice). Caveats: MIG instances have no
  peer-to-peer access — CUDA-aware MPI stages transfers through host memory
  (fine for correctness, meaningless for bandwidth) — and NCCL does not
  support communication across MIG instances, so a future NCCL transport
  cannot be tested there.

## Tests

```sh
julia --project=test -e 'using Pkg; Pkg.resolve(); include("test/runtests.jl")'
```

Serial physics tests run in-process; layout/halo/Poisson/invariance tests
are launched under the MPI-provided `mpiexec` for 1, 2, and 4 ranks.
