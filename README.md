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
API, communication layer). A companion pseudo-spectral HIT solver sharing
the pencil/transpose/FFT layer is designed in
[DESIGN_SPECTRAL.md](DESIGN_SPECTRAL.md) (not yet implemented).

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
- Semi-implicit wall-normal diffusion (`ydiffusion = :implicit`):
  per-stage Crank-Nicolson in SMR delta form with incremental (lagged)
  pressure projection — second-order in time, removes the `Δy²ₘᵢₙ` limit
  (verified stable at 200× the explicit viscous limit), decomposition
  invariant, GPU-verified. Tridiagonal solves run in a y-local layout one
  transpose away from the native one (a device copy when `p₂ = 1`).

Not yet: multi-*device* validation on real hardware, 4th-order scheme,
checkpointing, NCCL transport. See CODE_DESIGN.md §15.

**Pseudo-spectral HIT solver** ([DESIGN_SPECTRAL.md](DESIGN_SPECTRAL.md)),
sharing the pencil/transpose layer — implemented and tested:

- Truncation-aware distributed rfft pipeline: the spectral state holds only
  2/3-rule retained modes (y-pencil, `(kcut+1, m, m)`, `m = 2kcut+1`); dead
  modes are never stored, packed, or communicated (per-peer transpose boxes
  split at the pos/neg frequency boundary), and the three field components
  travel batched in one `Alltoallv`.
- Wray low-storage RK3 with integrating-factor viscosity (exact diffusion,
  no viscous CFL limit); convection in divergence form, dealiased by
  construction. Third-order self-convergence verified; single-rank
  trajectories match the SymmetryCode single-GPU solver.
- Prescribed-spectrum random ICs from counter-based (splitmix64) physical
  noise — Hermitian symmetry automatic, fields identical for every rank
  count. Taylor-Green ICs; shell-clamp forcing; K41 statistics and
  shell-binned spectra via `Allreduce`.
- Tests (`test/spectralmpitests.jl`): distributed transforms ≡ serial FFTW,
  exact 2D-in-3D Taylor-Green decay, inviscid energy conservation,
  decomposition invariance of full forced trajectories (1-4 ranks, all
  processor grids, host-staged buffers). GPU: single-device runs and 2-4
  ranks sharing an RTX 4090 over CUDA-aware MPI match CPU to 1e-15.
- Snapshot/restart I/O (`spectral_save` / `spectral_load!` /
  `snapshotsaver`, with `tstops` landing steps exactly on save times): the
  truncated state — exactly the dealiased coefficients, no ghost modes —
  written in global index order via collective MPI-IO, so files are
  independent of rank count and processor grid; TOML metadata sidecar.
  Tested: byte-exact round trip, cross-decomposition load, restart mid-run
  replaying the uninterrupted forced trajectory to 1e-12.
- Checkpoint/restart across job time limits: `checkpointer` processor
  (wall-clock interval + stop-file trigger, keeps the newest two, all
  collective decisions broadcast from rank 0), `spectral_latest` for
  restart discovery, solver `nstart` and `:stop` processor convention.
  SLURM integration goes through a stop *file*, not signals to the ranks:
  the job script traps `--signal=B:USR1@900`, touches the file, and
  resubmits itself (`examples/snellius/spectral_h100.sh`);
  `examples/spectral_hit.jl` auto-resumes from the newest checkpoint.
- Filtered-field/sub-filter-stress research output (`sfswriter`): at
  prescribed times, one extra nonlinearity evaluation (reusing the RHS
  pipeline buffers), a Gatherv of only the LES-cube modes of û and σ̂ to
  rank 0, then SymmetryCode's `sfs!` ported verbatim on the small coarse
  grid (Gaussian test filter with forced-shell protection, 2/3 truncation,
  τ = filter(σ) − ū⊗ū trace-free) and its exact JLD2 artifact schemas
  (`fields.jld2`, `les_meta.jld2`, `dns_meta.jld2`) — the downstream
  training/analysis pipeline applies unchanged. Validated against an
  independent serial FFTW oracle to 1e-12 on all processor grids.
- Example: `examples/spectral_hit.jl` (full production pattern).
- Snellius multi-device validation: 4 H100s vs 1 H100 produce the same
  final field to machine precision (2e-16 relative, host-staged buffers;
  `examples/snellius/spectral_validate.sh`). The run caught a
  decomposition-dependent adaptive-CFL bound — per-component velocity
  maxima are now reduced globally before combining, so CFL-adaptive
  trajectories (not just fixed-`Δt` ones) are decomposition invariant.

- NCCL transpose transport (`spectral_setup(; mpibuf = :nccl)`, package
  extension loaded by `using CUDA, NCCL`): the packed transpose buffers
  travel over NCCL grouped send/recv on the current CUDA stream,
  bypassing MPI/UCX for the bandwidth-critical path (MPI keeps setup,
  reductions, I/O). On 4 H100s this makes steps 5-11× faster than
  host-staged MPI with near-ideal strong scaling, and matches the
  single-GPU field to 2e-16 (CAMPAIGN.md S0). Needs one distinct GPU
  per rank and job-level GPU allocation (`--gpus=4`, not
  `--gpus-per-task`); on clusters where CUDA.jl uses the local toolkit,
  dev the stub `examples/snellius/NCCL_jll` (see snellius/README.md).

Spectral not yet: 2D slices (`slicefile` schema), memory-footprint
reduction (measured ≈ 31 field-equivalents vs the ledger's ~14; 1200³
does not fit 4 H100s yet) — DESIGN_SPECTRAL.md §8-9, CAMPAIGN.md.

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

GPU buffers are passed straight to MPI when the library reports CUDA
support (`MPI.has_cuda()` is `true` *after* `MPI.Init()` — Open MPI 5
loads its CUDA accelerator component at runtime); otherwise every message
is staged through host mirrors, which works with any MPI. Override with
`setup(; mpibuf = :device)` or `:host` — the latter is also the workaround
when the MPI/UCX stack *claims* CUDA support but mishandles device buffers
(seen on Snellius, where UCX cannot intercept CUDA.jl's allocations).

Cluster notes:

- On clusters, also pin CUDA.jl to the module toolkit so precompilation
  works on GPU-less login nodes and matches the MPI build:
  `CUDA.set_runtime_version!(v"12.8"; local_toolkit = true)`.
- **Snellius**: ready-made setup instructions and job scripts (MIG, A100,
  H100) live in [examples/snellius](examples/snellius/README.md).
- **MIG partitions**: MIG slices are separate CUDA devices, but scheduler
  QOS typically caps jobs at one slice (as on Snellius), and MIG supports
  neither CUDA IPC nor peer access nor NCCL — so MIG is only good for
  cheap oversubscribed single-device smoke tests; use full GPUs for
  multi-device tests.

## Tests

```sh
julia --project=test -e 'using Pkg; Pkg.resolve(); include("test/runtests.jl")'
```

Serial physics tests run in-process; layout/halo/Poisson/invariance tests
are launched under the MPI-provided `mpiexec` for 1, 2, and 4 ranks.
