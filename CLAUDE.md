# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MPI-parallelized, GPU-capable (KernelAbstractions + CUDA-aware MPI) solver package in Julia containing **two solvers that share one pencil/transpose/FFT communication layer**:

1. **Finite-volume channel/box solver** (`src/*.jl`): symmetry-preserving staggered finite volumes, FFT/Thomas pressure solve, Williamson low-storage RK3, optional semi-implicit wall-normal diffusion.
2. **Pseudo-spectral HIT solver** (`src/spectral/*.jl`): truncated Fourier state, Wray RK3 with integrating-factor viscosity, shell forcing, snapshot/checkpoint I/O, filtered-field/SFS research output.

The design documents are the authority and are kept current: `DESIGN.md` (FV physics/decisions), `CODE_DESIGN.md` (FV code-level: layouts, transposes, API, memory ledger §13, testing §14, deferred seams §15), `DESIGN_SPECTRAL.md` (spectral solver, incl. I/O schemas §8, memory budget §9, development phases §11). **When you add or change a feature, update the relevant design doc and the README status section in the same commit.**

## Commands

Full test suite (serial tests in-process, then each MPI test file under `mpiexec` for 1, 2, 4 ranks):

```sh
julia --project=test -e 'using Pkg; Pkg.resolve(); include("test/runtests.jl")'
```

Run a single MPI test file at one rank count (the only way to run a subset — there is no finer-grained filter):

```sh
OMPI_MCA_rmaps_base_oversubscribe=true julia --project=test -e '
  using MPI
  run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --startup-file=no --project=test test/spectralmpitests.jl`)'
```

`test/serialtests.jl` alone runs in-process (fast iteration): `julia --project=test test/serialtests.jl` — or via TestEnv in a live session.

Spell check: `typos` (config in `typos.toml`).

The `examples/` directory is a **separate Julia environment** with its own Manifest and `LocalPreferences.toml` pointing MPI.jl at the CUDA-aware system Open MPI. Run examples with `--project=examples`; after adding a dependency to the main package, run `Pkg.resolve()` in *both* the test and examples envs (stale manifests are a recurring failure).

GPU tests/examples run multi-rank on the single local RTX 4090 (ranks share the device via UCX `cuda_ipc`); Snellius job scripts are in `examples/snellius/`.

## Architecture

### Shared layer (used by both solvers)

- `layout.jl`: 2D processor grid over pencil layouts; `AXES` maps each pencil orientation to (distributed dim, local dim, distributed dim); `blockrange(n, p, c)` is the single source of truth for uneven block splits. MPI Cartesian rank order is **row-major**: `coords(r) = (r ÷ p2, r % p2)`.
- `pack.jl` / `transpose.jl`: pack/unpack kernels and global transposes via one batched `Alltoallv`; the spectral variant is truncation-aware (dead modes never stored or communicated, per-peer boxes split at the pos/neg frequency boundary).
- `halo.jl`: halo exchange unified with wall/periodic boundary fills (FV only).
- Every message optionally stages through host mirrors (`setup(; mpibuf = :host)`) for non-CUDA-aware MPI stacks; `:device` passes GPU pointers directly. `MPI.has_cuda()` is only meaningful *after* `MPI.Init()`. On Snellius the UCX stack claims CUDA support but mishandles CUDA.jl allocations — use `:host` there until retested.

### Spectral solver conventions (matter for nearly every change)

- State `uh` is `(kcut+1, m, m, 3)` complex, `m = 2kcut+1`, y-pencil, **compact frequency order** (0..kcut then −kcut..−1, see `compactfreq`); normalization û = F[u]/n³. This *is* the dealiased set — there are no ghost modes. Mean mode û(0) is assumed zero.
- `spectral_solve!` drives processors: each is called as `proc(state, s)` with `state = (; uh, t, n)`; returning `:stop` ends the solve. `tstops` makes steps land exactly on sample times; `nstart` continues step numbering across restarts. Processors in `io.jl`/`les.jl` (`snapshotsaver`, `checkpointer`, `sfswriter`) follow this convention.
- Snapshot files (`io.jl`) are written with collective MPI-IO subarray views in **global index order**, so they are independent of rank count and processor grid; TOML sidecar carries metadata; writes are atomic (write-then-rename). All collective decisions (checkpoint due, stop file seen) are made on rank 0 and broadcast.
- SLURM time limits are handled via a stop *file*, never signals to ranks: the job script traps `--signal=B:USR1@900`, touches the file, `checkpointer` polls it, script resubmits (`examples/snellius/spectral_h100.sh`).
- `les.jl` ports SymmetryCode's (`~/Projects/Symmetry/SymmetryCode`) `sfs!` recipe and JLD2 artifact schemas **verbatim** — the downstream training pipeline must load these files unchanged, so do not "improve" filter kernels, shell binning, or schema keys.
- Memory discipline: both design docs keep an explicit ledger of field-sized allocations. New features must reuse existing buffers where possible and account for any new full-size field in the ledger.

### Testing philosophy

The load-bearing invariant is **decomposition invariance**: identical trajectories/files across rank counts and processor grids (byte-exact for same decomposition, ~1e-12 tolerance across decompositions). ICs use counter-based (splitmix64) noise specifically so this holds. New distributed features get tested against an independent serial FFTW oracle inside the MPI test files, across all processor grids for each rank count.

Two Julia gotchas that have caused real debugging pain here:

- Inside a `@testset`, a helper `function f(s); uh = ...` **rebinds an enclosing local** of the same name (closure capture) — use fresh names for helper locals.
- When an MPI test diverges, print diagnostics from **all** ranks (`MPI.Gather` then print on rank 0), not just rank 0 — per-rank divergences hide behind rank-0-only prints.
