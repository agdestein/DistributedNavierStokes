# Design document: distributed pseudo-spectral HIT solver

Status: August 2026 — phases 1-3 of §11 implemented (`src/spectral/`,
CPU + GPU + multi-rank tested; see README status), plus phase 4:
MPI-IO snapshots in global index order, wall-clock + stop-file
checkpointing with SLURM resubmission (`io.jl`,
examples/snellius/spectral_h100.sh; §8's signal handling goes through a
stop file touched by the job script's trap — signals never reach the Julia
ranks), and the filtered-field/SFS research output in the SymmetryCode
schemas (`les.jl`: LES-cube gather + rank-0 `sfs!` port, JLD2 as a hard
dependency — pure Julia, no system-library coupling). Remaining: 2D slices
(§8) and phases 5-6 (Snellius, optimizations).
Author: Syver Døving Agdestein (with Claude).
Companions: [DESIGN.md](DESIGN.md) (FV solver; §2 anticipated this solver and
§4/§5/§7 specify the shared infrastructure), [CODE_DESIGN.md](CODE_DESIGN.md)
(code-level conventions, which apply here unchanged).

This document records the design for the production HIT-DNS code: a
pseudo-spectral solver for the triply periodic incompressible Navier-Stokes
equations, distributed over multiple GPUs via the pencil/transpose/FFT layer
already built for the FV solver. It is the multi-GPU successor of the
single-GPU pseudo-spectral research solver in SymmetryCode (`solver.jl`),
which maxed out at 810³ on one H100. The goal is the *same experiments at
higher Reynolds number*: forced/decaying HIT, Taylor-Green, filtered fields +
sub-filter stresses for LES-closure research.

## 1. Goals and non-goals

Goals (v1):

- DNS of forced and decaying HIT at transform-grid sizes 2048³-4096³ on
  8-128 NVIDIA GPUs (Snellius A100/H100 nodes), CUDA-aware MPI.
- Fourier-Galerkin with exact dealiasing (2/3 rule), exact projection,
  integrating-factor viscous treatment (§6).
- Initial conditions: prescribed-spectrum random field, Taylor-Green vortex.
- Checkpoint/restart robust to SLURM time limits, restartable on a different
  rank count (§8).
- In-situ outputs: filtered velocities + sub-filter stresses on the LES grid
  in the **existing SymmetryCode artifact schemas**, DNS turbulence
  statistics, 2D slices for visualization (§8).
- Single-rank runs reproduce SymmetryCode trajectories on matching setups
  (the in-house oracle, §10).

Non-goals:

- Walls / inhomogeneous directions (that is the FV solver's job; the split
  is deliberate, DESIGN.md §2).
- 2D (SymmetryCode keeps the 2D role; a pencil code is inherently 3D).
- Differentiability, non-mutating API, LES closure evaluation inside the
  distributed solver. Closures train and run downstream on the gathered
  LES-grid data, unchanged.
- Backwards compatibility with the SymmetryCode solver API.

## 2. Relation to the FV solver: one repo, shared layer

**Decision: the spectral solver lives in this repository** and consumes
`layout.jl`, `transpose.jl`, `pack.jl`, and the stats plumbing directly.
Those files are solver-agnostic by design (their API never mentions
Navier-Stokes, DESIGN.md §2); this solver is their second consumer. The
package split (DESIGN.md §12) stays deferred until a third consumer exists —
splitting now buys nothing and costs a JLL/registry story.

New files, mirroring the FV set:

```
src/spectral/
  setup.jl       # spectral_setup(): grid, truncation, decomposition, plans, buffers
  fft.jl         # distributed 3D rfft pipeline (truncation-aware stages)
  operators.jl   # projection, viscous factor, wavenumbers from descriptors
  step.jl        # IF-RK3 (Wray), CFL, solve!
  forcing.jl     # shell energy clamp
  ic.jl          # random spectrum, Taylor-Green
  io.jl          # checkpoint/restart, fieldsfile/slice/stats writers
```

All FV code-design decisions carry over verbatim: MPI not NCCL (transport
pluggable), no pencil-decomposition library, KernelAbstractions + CUFFT via
AbstractFFTs, NamedTuples as namespaces, everything preallocated, functions
take `(data..., setup)`.

## 3. Formulation and dealiasing

Fourier-Galerkin in the divergence form: the nonlinear term is
`-ik_j F[v_i v_j]` with the symmetric stress (6 products), projection and
viscosity applied pointwise in spectral space — the same formulation as
SymmetryCode, which keeps the oracle comparison exact. (Rotational form has
the same transform count, 9 per RHS; no reason to switch.)

**Dealiasing decision (v1): 2/3 cubic truncation, made truncation-aware.**
The 2/3 rule is exact for the quadratic nonlinearity and is what the paper's
existing data uses; changing the rule would change the physics of the
comparison. What *is* modernized is the implementation: the single-GPU code
stores and transforms full N³ arrays in which a third of every dimension is
zeros. Here, truncated modes are never stored, packed, sent, or transformed:

- The spectral state lives on the retained-mode grid only
  (`(N/3+1) × 2N/3 × 2N/3` of the rfft array): **0.30×** the memory of a
  full spectral field.
- Transposes carry only retained modes: the x→z all-to-all shrinks to ~2/3
  of naive volume, the z→y one to ~4/9 (§5).
- FFT stages after the first run on reduced batch counts; padding with
  zeros happens locally inside the pipeline (this is Orszag's 3/2-padding
  view of the same rule — retained grid M, transform grid N = 3M/2).

The layout descriptors already support per-stage `gdims` (the rfft
half-size uses this, CODE_DESIGN.md §5), so truncation-aware stages fall
out of existing machinery.

Reserved seams, not v1 (§12):

- **Spherical truncation + phase shifts** (Rogallo 1981; the PSDNS-class
  approach): truncating on the sphere `|k| ≤ √2·N/3` with shifted-grid
  evaluation raises usable kmax by 41% for the same grid (≈ 2.8× fewer
  points at fixed resolved band). Exact dealiasing costs 2× nonlinear-term
  evaluations; the cheap variant alternates random shifts across RK
  substages so residual aliasing is decorrelated in time rather than
  removed. Both the truncation mask and the per-stage phase factor are
  pointwise kernels — the seam is a functor slot for the mask/shift, zero
  structural cost to reserve.
- **Hou-Li exponential filter** (`exp(-36 (k/kmax)³⁶)`) as an alternative
  mask for users who prefer smooth truncation. One line in the mask
  functor; 2/3 stays the default so results remain comparable.

## 4. Data layout and decomposition

- Spectral state: 3 complex components on the truncated grid, resident in
  the fully-transformed orientation (y-pencil in the FV nomenclature;
  degenerate slab cases included). Physical-space fields exist only
  transiently inside the RHS pipeline, in the x-pencil orientation.
- **Slab decomposition (`p₂ = 1`) is the expected default** at ≤ ~64 GPUs:
  with both x and y rank-local, a 3D transform is one local 2D rfft + one
  transpose + a 1D FFT — *half* the communication of the pencil path. The
  layout layer supports slabs with no special-casing; pencil remains
  available via `procgrid` for rank counts approaching N. The choice is a
  runtime parameter settled by the benchmark script, exactly as in the FV
  design (no autotuning framework).
- **Batched transposes from day one**: transpose plans accept a batch of
  fields (3 velocity components, or a batch of stress products) in a single
  `Alltoallv` — same bytes, ⅓-⅙ the message count. This mattered little for
  the FV code (transposes only in the Poisson solve) but is first-order
  here, where transposes are the runtime.
- Element type generic, default FP64 (§9).

## 5. The RHS pipeline

Per RHS evaluation (one RK stage):

```
3 × inverse distributed rfft   û (truncated, spectral pencil) → v (physical x-pencil)
    local products             v_i v_j, batch of B_w at a time
6 × forward distributed rfft   products → σ̂_ij (truncated)
    local accumulation         du_i -= ik_j σ̂_ij  (each σ̂_ij feeds ≤ 2 components)
    pointwise spectral         projection, viscous factor, forcing
```

9 distributed transforms per RHS; with Wray RK3 that is 27 per step —
27 transposes (slab) or 54 (pencil). This is the known price of spectral
(DESIGN.md §2 footnote) and is what the truncation-aware transposes and
batching amortize.

Implementation notes:

- **σ̂ is never stored as 6 fields.** Each transformed product is
  accumulated into `du` immediately (`du.x -= ik_x σ̂xx`, off-diagonal
  products feed two components), so stress scratch is `B_w` pipeline slots,
  not 6 persistent fields. The batch width `B_w ∈ {1, 3, 6}` trades memory
  for transpose latency (§9); default 3, reusing the velocity pipeline
  buffers.
- **FFT-along-dim-3 striding**: for the FV code, permuting in `unpack!` so
  the transform dimension is contiguous was a recorded fallback. Here FFTs
  dominate, so both variants are benchmarked in phase 1 and the winner
  becomes the default, not the fallback.
- Wavenumber kernels (projection, viscosity, gradients, spectrum masks) are
  pointwise in k: the SymmetryCode kernels port mechanically, with
  wavenumbers derived from the descriptor's global ranges instead of
  `Grid.n`.
- CFL: `max |v|` is fused into the product stage (the physical velocities
  are in hand every stage anyway) + `Allreduce(max)` every `n_cfl` steps.
  With the integrating factor there is no viscous limit (§6).

## 6. Time integration

**Low-storage RK3 (Wray) with integrating-factor viscosity.** Diffusion is
diagonal in k, so the analytic factor `exp(-ν k² Δt)` is applied per stage
to both low-storage registers (state and accumulator) at each stage
boundary; the RK coefficients act on the transformed variable
`ŵ = e^{ν k² t} û`. Compared to the Crank-Nicolson/Adams-Bashforth scheme
in SymmetryCode: same elementwise cost, *exact* rather than 2nd-order for
diffusion, unconditionally stable, no stored old-RHS register, no
first-step special case. The exponential is computed inline per mode — the
kernel is bandwidth-bound, so this is free even with adaptive Δt (no
factor tables to rebuild).

Honest expectation: at well-resolved DNS the convective CFL dominates (the
viscous/convective limit ratio scales with the cell Reynolds number, which
grows with Re_λ), so the big runs gain little Δt. The wins are the low-Re /
coarse-grid runs, late decay, and deleting the viscous branch from the
timestep proposal. Since it costs nothing, it is simply the default; the
TGV order-verification test (§10) confirms the stage-boundary factor
placement preserves 3rd order in the convective terms.

Forcing: low-wavenumber shell energy clamp, matching SymmetryCode
(`energy_shells` / `maintain_shell_energy!`): per-rank shell masks
precomputed from local k-ranges, local masked energy sums +
`Allreduce`, pointwise rescale. Decaying runs (TGV) are unforced.

## 7. Initial conditions

- **Taylor-Green**: evaluate locally in physical space, forward transform.
  Trivial.
- **Prescribed-spectrum random field**: do *not* port the spectral-space
  `randn!` + `symmetrize!` approach — distributed Hermitian symmetrization
  pairs modes on the kx = 0 plane that live on different ranks. Instead:
  white noise in *physical* space, forward transform (symmetry automatic),
  project, then shell-rescale (masked local sums + `Allreduce`), as in
  SymmetryCode's shell loop. Costs one extra transform at setup.
- **Decomposition-invariant noise**: the physical-space noise is generated
  by a counter-based RNG keyed on the *global* grid index (a
  hash-per-index kernel, not a stateful stream). The IC is then identical
  for every rank count and processor grid, which extends the single most
  valuable test — identical trajectories across decompositions — through
  initialization instead of starting it from a loaded field.

## 8. I/O: checkpointing and outputs

The governing observation: **only the DNS is distributed.** The LES-grid
artifacts (ū, τ at ~128³-270³) are small enough to gather to one rank. So
the downstream pipeline — training, analysis, plotting in SymmetryCode —
is kept *unchanged* by writing its exact artifact schemas.

### Filtered fields and sub-filter stress (the research payload)

At each sampling time: run one extra nonlinearity evaluation on the current
field (the stress products at the RK-stage time levels are not reusable),
spectrally truncate û and σ̂ to the LES cube, `Gatherv` the retained low-k
modes to rank 0 (≤ a few hundred MB), and compute
`τ = filter(σ) - ū ⊗ ū`, trace-free, exactly as `sfs!` does now. Rank 0
writes the existing JLD2 schemas: `fieldsfile` (ū, τ, per-snapshot Re_Δ),
`lesmetafile`, `dnsmetafile`. Both filter types (sharp cutoff, Gaussian
test filter) are diagonal in k and distribute trivially.

### Statistics and slices

- DNS statistics: energy, ε, Re_λ, kmax·η every few steps; shell spectra at
  coarser cadence. Local spectral sums + `Allreduce` (the RFFT
  double-counting logic ports unchanged; the FV `spectrumstats` shows the
  pattern). Rank-0 appended time series, `dnsmetafile`-compatible.
- 2D slices: gather the z = l/2 plane (FP32, `slicefile` schema) from the
  ranks owning it in the physical x-pencil stage.
- Hooks are FV-style rank-aware processors (CODE_DESIGN.md §12).

### Checkpoint/restart

Restart state is small and exact: û (3 truncated spectral fields) + time,
step, Δt, forcing shell reference energies, IC seed/config. Decisions:

- **MPI-IO, not parallel HDF5**: `MPI.File` collective writes with subarray
  views in *global index order*, so a checkpoint is layout- and
  rank-count-independent. One less library that must match the system MPI
  build — a lesson the Snellius UCX saga (examples/snellius/README.md)
  makes worth taking seriously. A small JSON/TOML sidecar carries metadata.
  JLD2 remains for the small rank-0 research artifacts only.
- Atomic via write-then-rename; keep the last two checkpoints.
- Triggers: every N wall-clock minutes *and* on SLURM
  `--signal=B:USR1@900` (trap → checkpoint → exit); the job script
  resubmits itself while the target time is not reached. Restarting on a
  different rank count must reproduce trajectories to FFT-reordering
  roundoff (tested, §10).

## 9. Memory and precision budget

Working numbers, FP64, transform grid N³ (retained grid 2N/3 per
direction). One N³ real field = 8·N³ bytes ≡ "1 equivalent"; a truncated
spectral field is 0.30 equivalents.

| What | Equivalents |
|---|---|
| û | 0.9 |
| RK register (low-storage) | 0.9 |
| du | 0.9 |
| physical velocities v (3, simultaneous) | 3.0 |
| product buffers (batch `B_w = 3`) | 3.0 |
| pipeline complex ping-pong (× batch) | ~4.0 |
| transpose send/recv | ~1.3 |
| **Total** | **~14 (B_w = 3), ~9.5 (B_w = 1)** |

≈ 80-110 bytes/point. Targets (H100 94 GB, ~60 usable; A100 40 GB ≈ 2.5×
the GPU count):

| Transform grid | Total | H100s | Re_λ vs 810³ single-GPU |
|---|---|---|---|
| 2048³ | ~0.7 TB | ~16 (4 nodes) | ×1.9 |
| 3072³ | ~2.3 TB | ~40-48 | ×2.4 |
| 4096³ | ~5.5 TB | ~96-128 | ×3.0 |

Every feature wanting a persistent 3D array argues against this table
(same discipline as DESIGN.md §8).

Precision: FP64 default and the validation baseline. Planned option: FP32
fields with FP64 reductions/statistics — spectral HIT tolerates FP32 fields
well, but it enters only after the FP64 test matrix is green.

## 10. Validation and testing plan

The decisive advantage over the FV port: **a bit-exact in-house oracle.**
SymmetryCode's single-GPU solver runs the same formulation on the same
grids.

1. **FFT pipeline**: distributed forward/inverse ≡ serial FFTW on gathered
   fields; truncation-aware round trip ≡ identity on retained modes; all
   processor grids incl. slabs and P = 1. CPU backend + `mpiexec`, the FV
   test ladder reused.
2. **Single-rank ≡ SymmetryCode**: identical trajectories to machine
   precision on matching setups (same IC field, ν, Δt, forcing; account
   for the FFT normalization convention, which need not be copied).
3. **Decomposition invariance**: identical trajectories (to roundoff)
   across rank counts and processor grids, *including* the
   counter-based-RNG random IC. The single most valuable CI test.
4. Discrete identities: divergence-freeness after projection (machine
   precision), inviscid energy conservation, Parseval consistency of the
   truncated storage.
5. **TGV order verification**: confirms 3rd-order convective accuracy with
   the integrating factor in place.
6. Physical validation: reproduce a current 512³ SymmetryCode forced-HIT
   run's statistics and spectra; K41 checks (kmax·η, spectral collapse) at
   scale.
7. I/O: restart on a different rank count ≡ uninterrupted run;
   fieldsfile/slicefile outputs load in the existing SymmetryCode analysis
   pipeline on a small end-to-end run.

## 11. Development order

1. **Distributed rfft pipeline** (truncation-aware stages, slab + pencil,
   batched transposes): tests §10.1; benchmark strided-vs-permuted dim-3
   FFTs and slab-vs-pencil on 2-8 real GPUs early.
2. **Spectral RHS + IF-RK3 + TGV**: oracle comparison (§10.2),
   decomposition invariance, conservation identities.
3. **Random IC + shell forcing + statistics**: reproduce the 512³
   reference run.
4. **Checkpoint/restart + research outputs**: MPI-IO restart, SLURM signal
   handling, fieldsfile/lesmetafile/slicefile writers, end-to-end schema
   test against the downstream pipeline.
5. **Snellius**: A100 first, then H100. Carry over the encoded gotchas
   (`--mpi=pmix`, `--gpus-per-task=1`, `JULIA_CPU_TARGET`, memory pool
   off, `DNS_MPIBUF=host` fallback). **Note:** host-staged MPI buffers
   hurt this solver far more than the FV one (transposes *are* the
   runtime) — re-test the UCX device path first, and consider the NCCL
   transport earlier than the FV roadmap did: NCCL bypasses UCX entirely
   and may be the practical fix on this cluster.
6. **Production**: scaling benchmark, procgrid choice, first 2048³
   campaign.

## 12. Deferred (seams reserved, not built)

- Spherical truncation + phase-shift dealiasing (§3): mask/shift functor
  slot in the truncation kernel; +41% usable kmax when wanted.
- Hou-Li smooth truncation option (§3): same functor slot.
- NCCL transport behind `transpose!` (shared seam with the FV solver;
  possibly promoted to v1.x by Snellius UCX findings, §11).
- Comm/compute overlap: pipeline one component's FFT stage against
  another's transpose (streams). The batched-transpose API is the seam.
- FP32 fields with FP64 reductions (§9).
- Passive scalar: one more transform per RHS, same machinery; slot in the
  RHS loop, as in the FV design.
- Higher-order / exponential integrators beyond IF-RK3: not planned; the
  seam is the standard `(a, b, c)` tableau in the stepper.

## 13. Open questions

- Module/namespace name for the spectral half (`Spectral` submodule vs.
  flat `spectral_*` prefixes).
- Snellius UCX device-path status (retest before choosing the phase-5
  transport; determines NCCL priority).
- Whether the LES-cube gather should optionally land on *all* ranks
  (Allgatherv) so future in-situ closure diagnostics can run distributed —
  v1 gathers to rank 0 only.
- Sampling cadence vs. checkpoint cadence coupling (reuse the extra
  nonlinearity evaluation at sampling times for a free CFL/statistics
  refresh?). Decide during phase 4.
