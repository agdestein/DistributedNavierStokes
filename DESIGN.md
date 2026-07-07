# Design document: distributed multi-GPU incompressible DNS solver

Status: draft, July 2026.
Author: Syver Døving Agdestein (with Claude).

This document records the design choices for a new MPI-parallelized,
GPU-resident DNS solver for the incompressible Navier-Stokes equations. It is
a companion to IncompressibleNavierStokes.jl (INS.jl), not a replacement: the
new code trades generality (differentiability, 2D, general BCs, non-mutating
API) for scale. Reference designs: CaNS / cuDecomp (Costa; Romero et al.),
AFiD, and the extreme-scale wall-flow solver of the CaNS group
(arXiv:2502.06296).

## 1. Goals and non-goals

Goals:

- DNS of homogeneous isotropic turbulence (HIT) and turbulent channel flow at
  grid sizes of 2000³-4000³, on 8-128 NVIDIA GPUs (CUDA-aware MPI / NCCL).
- Energy-conserving (symmetry-preserving) discretization, 2nd and 4th order
  (Verstappen & Veldman 2003).
- Pressure Poisson equation solved *exactly* (to machine precision) per step
  via FFT diagonalization + direct banded solve.
- Single-rank runs reproduce INS.jl physics on matching setups (validation).

Non-goals (deliberately discarded relative to INS.jl):

- Differentiability (ChainRules/Enzyme adjoints).
- Non-mutating operator variants.
- 2D (a pencil code is inherently 3D; INS.jl keeps the 2D role).
- General boundary conditions (inflow/outflow, ducts with walls in two
  directions, immersed boundaries).
- Backwards compatibility with the INS.jl API.

## 2. Physics and boundary-condition scope

- Each direction is either **periodic** or **no-slip walls** on both ends.
  Supported flow classes: triply periodic (HIT, forced or decaying), channel
  (periodic x, z; walls y). Plane Couette (moving wall) is a trivial
  extension of the wall ghost fill — keep the wall BC parametrized by a wall
  velocity, but don't generalize further.
- At most **one wall-normal direction** (canonically y). This is what keeps
  the Poisson problem "FFT × FFT × banded direct solve". Ducts are
  explicitly out of scope; do not half-design for them.
- Planned extension (design seam only, not v1): passive scalar / Boussinesq
  temperature. Fits the BC scope exactly (Rayleigh-Bénard = periodic x, z +
  walls y) and reuses all machinery. The scalar transport kernel slot should
  exist in the time stepper from the start.

## 3. Discretization

- Staggered Cartesian finite volumes, same convention as INS.jl:
  `u[I, α]` on the right face of volume `I` in direction `α`.
- **Symmetry-preserving 2nd order** (baseline) and **4th order**
  (Verstappen & Veldman 2003) convection/diffusion.
- Flux reconstruction is a **swappable stencil functor** internally (so the
  seam exists for, e.g., dissipative upwind or higher-order schemes later),
  but only the two symmetry-preserving schemes ship in v1. Rationale:
  alternative fluxes multiply halo widths and the test matrix, and break the
  energy-conservation identity of the code. LES with dissipative schemes is a
  separate project.
- **Halo width is a per-scheme parameter** (2nd order: 1 ghost layer;
  4th order: 2-3 layers). All ghost/halo machinery, kernel index ranges, and
  MPI exchange buffers are parametrized by this width from day one.
- **Non-uniform grid in the wall-normal direction only.** Periodic
  directions are uniform (required by the FFT Poisson solver anyway). Metrics
  are 1D arrays (`Δy`, `Δyᵤ`, stretching maps), indexed per-direction — no 3D
  metric fields. Skew-symmetry is defined w.r.t. the diagonal volume norm, as
  in the Verstappen line of work.
- **4th order is restricted to uniform directions in v1.** 4th-order
  symmetry-preserving coefficients on stretched grids are delicate
  (accuracy degradation, more involved construction); revisit later.
  Practical v1 matrix: {2nd order on any grid} and {4th order on uniform
  grids, i.e. HIT and channel with uniform y if ever useful}.

### Poisson operator consistency

For exact discrete mass conservation the projection must use
`L = M Ω⁻¹ Mᵀ` with the *same* divergence `M` as the scheme:

- 2nd order: `L` is diagonal per Fourier mode (modified wavenumbers) in
  periodic directions, **tridiagonal** in y.
- 4th order: still diagonal per Fourier mode (wider modified-wavenumber
  stencil) in periodic directions, but **penta/heptadiagonal** in y.

Consequence: the wall-normal solve is designed as a **banded direct solve**
(banded LU, bandwidth as a parameter), not hard-coded Thomas. Tridiagonal is
the fast path.

## 4. Data layout

- **Velocity as three separate fields** `(; ux, uy, uz)` (NamedTuple), not a
  4D array. Each component has its own shape, BC semantics, halo exchange,
  and FFT plan; separate arrays map cleanly onto pencil descriptors and avoid
  the awkward component dimension in kernels.
- Fields are GPU-resident for the whole run (`CuArray`); host copies only
  for I/O. Kernels written with **KernelAbstractions.jl**, so the CPU backend
  works with the same code (used for testing and small runs, and keeps an
  AMD/HIP door open).
- Element type generic, default `Float64`. See §8 for the precision story.

### Ghost/halo abstraction (BC ≡ MPI interface)

Every field carries `w` layers of ghost volumes on all sides. A ghost layer
is filled by exactly one of three fillers, chosen per direction/side at setup:

1. **MPI halo exchange** with the neighboring rank (CUDA-aware buffers or
   NCCL, see §7),
2. **intra-rank periodic copy** (direction not decomposed, or rank is its own
   neighbor),
3. **physical wall BC kernel** (no-slip / moving wall).

So "apply BCs" and "exchange halos" are the same operation: iterate
directions, run the configured filler. Implementation rules:

- Exchanges run **sequentially per direction**, and the sent slabs **include
  the already-filled halos** of previously handled directions. Then edge and
  corner ghosts come for free — no diagonal-neighbor communication, ever.
- The kernel-launch API distinguishes interior/boundary index ranges so that
  halo exchange can later be **overlapped** with interior computation.
  Overlap is *not* implemented in v1, but the seam must exist.

### Domain decomposition

- 2D **pencil decomposition** (with 1D slab as the degenerate case). The
  wall-normal direction y is kept rank-local during the banded solve stage;
  pencils rotate via global transposes (§5).
- The decomposition/layout descriptor (which global directions are local,
  local index ranges, neighbor ranks per direction) is *the* central
  abstraction; everything else (fields, FFT plans, halo fillers, I/O) is
  built against it. This is the component to prototype first (§10).

## 5. Poisson solver

**Decision: FFT diagonalization + banded direct solve, with pencil
transposes. No CG/multigrid fallback.**

Rationale:

- Exact to machine precision in one pass — the projection actually returns a
  divergence-free field; no iteration-tolerance tuning at 4000³.
- Distributed GPU multigrid (needed to make CG competitive) is *more*
  infrastructure than transposes, not less; plain CG's per-iteration global
  reductions are latency-poison at scale.
- The transpose problem is solved territory (2DECOMP&FFT, cuDecomp,
  diezDecomp); see §7.

Solve choreography per projection (channel case; HIT is FFTs only):

```
x-pencil:  FFT in x
transpose: x-pencil → z-pencil
z-pencil:  FFT in z
transpose: z-pencil → y-pencil
y-pencil:  banded solve in y (tridiagonal for 2nd order), per (kx, kz) mode
transpose + inverse FFTs back
```

≈ 4 global transposes per Poisson solve. Slab decomposition halves this and
is competitive up to O(100) GPUs; the layout layer supports both and the
choice is made by autotuning/benchmark, not baked in.

- Real-to-complex FFTs (cuFFT), batched over the non-transform directions.
- Modified wavenumbers of the *scheme's* divergence stencil (2nd or 4th
  order), exactly as INS.jl already does for `random_field`.
- Banded LU in y factorized once at setup per (kx, kz) (coefficients depend
  on the mode only through the eigenvalue — store per-mode factorizations or
  factor on the fly; decide by memory budget, §8).

## 6. Time integration

- **Low-storage RK3** (Wray/Spalart) as the default explicit integrator.
  Low-storage is mandatory at the target sizes (§8). CFL-adaptive Δt
  (convective + viscous limits).
- **Semi-implicit wall-normal diffusion (Crank-Nicolson in y) as a planned
  mode.** On stretched wall grids at high Re_τ the explicit viscous limit
  Δt ∝ Δy²_min dominates and crushes the time step. CN in y costs three
  tridiagonal Helmholtz solves per substep *in the same y-local layout the
  Poisson solve already uses* — shared machinery. v1 may ship explicit-only,
  but the time-stepper interface has the implicit-diffusion seam from the
  start (this is the second-biggest design decision after the layout layer).
- HIT forcing: spectral shell forcing (fixed-energy low-wavenumber shells)
  and/or linear (Lundgren) forcing. Channel: constant pressure gradient or
  constant flow rate (bulk-velocity correction).

## 7. Communication / decomposition technology (Julia ecosystem, July 2026)

Survey results:

| Component | State (July 2026) | Assessment |
|---|---|---|
| MPI.jl | CUDA-aware buffers (`CuArray` directly in send/recv, collectives) fully supported with a CUDA-aware system MPI; `MPI.has_cuda()` to check | Solid foundation, use it |
| NCCL.jl (JuliaGPU) | Maintained (recent commits, JLL-packaged libnccl) | Usable as an alternative transport for transposes/halos; often beats CUDA-aware MPI for all-to-all on NVLink/IB systems |
| KernelAbstractions.jl | Mature, actively maintained, the standard portable-kernel layer | Use for all kernels (same as INS.jl) |
| cuDecomp (NVIDIA, C/CUDA) | v0.6.2 (Mar 2025); autotunes processor grid + backend (MPI / NCCL / NVSHMEM); powers CaNS | **No Julia wrapper exists.** C API is small (~20 entry points); a `ccall` wrapper via Clang.jl is feasible but we'd own it, plus JLL/build story (needs system MPI) |
| diezDecomp | New (2025), Fortran+OpenACC, portable (HIP/AMD), used for CaNS on LUMI-class machines; performance ≈ cuDecomp on Leonardo | Wrong language surface for Julia; watch, don't adopt |
| PencilArrays.jl / PencilFFTs.jl | Actively released (PencilArrays v0.19.11, Jun 2026). CUDA support merged 2022 (CuArray-backed pencils, transposes over CUDA-aware MPI); used by Oceananigans' distributed-GPU work. But: GPU path is **not documented**, no autotuning, no NCCL/NVSHMEM backend, halo exchange exists but GPU maturity unclear | Fastest way to a working prototype; uncertain as the long-term extreme-scale layer |
| CaNS itself | Active; 2nd-order, uniform-in-x,z + stretched y, exactly our Poisson structure | Not reusable as a library (Fortran app), but the reference for every design decision and the validation baseline |

**Decision: own thin transpose/halo layer over MPI.jl, with the transport
pluggable (CUDA-aware MPI first, NCCL.jl second).**

- The required functionality is bounded: pencil descriptors, buffer
  pack/unpack kernels (KernelAbstractions), all-to-all (or
  neighbor sendrecv) transposes, halo exchanges. This is a small fraction of
  what cuDecomp does; what cuDecomp adds is *autotuning*, which we replace
  with a benchmark script + config option (processor grid and transport are
  runtime parameters).
- Prototype phase may use PencilArrays.jl to get moving and as a
  cross-check; decide after measuring whether to keep it underneath or keep
  only our own layer. The solver code must only ever see *our* layout
  abstraction, so this swap stays cheap.
- Wrapping cuDecomp stays the fallback if our transposes underperform at
  scale — the abstraction boundary (descriptor + transpose + halo API) is
  designed so cuDecomp could implement it.

## 8. Memory and precision budget

Working numbers, FP64, N³ grid, P GPUs (80 GB H100):

| Grid | Bytes/field | 64 GPUs | 128 GPUs |
|---|---|---|---|
| 2000³ | 64 GB | 1.0 GB/GPU | 0.5 GB/GPU |
| 3000³ | 216 GB | 3.4 GB/GPU | 1.7 GB/GPU |
| 4000³ | 512 GB | 8.0 GB/GPU | 4.0 GB/GPU |

Field-sized allocation budget (target ≤ 12 field equivalents): 3 velocity
components + 1 pressure + 3 RK low-storage registers + 1-2 transpose/FFT work
buffers + statistics accumulators. At 4000³ on 64 GPUs this is ~96 GB of
80 GB — so 4000³ needs 128 GPUs *or* FP32 fields; 3000³ on 64 GPUs is
comfortable. Every feature that wants a persistent 3D array must justify
itself against this table.

Precision:

- Default: FP64 throughout — simplest, correct, and the baseline for
  validation.
- Planned option: FP32 fields with FP64 Poisson solve and FP64 reductions
  (dot products, statistics, CFL). Element types are generic in Julia so this
  is mostly a testing/validation cost, but divergence and energy-conservation
  checks must run in the FP32 CI matrix before it's advertised.

## 9. I/O and in-situ statistics

At target sizes a single snapshot is 0.2-2 TB — "dump and post-process" does
not work. Therefore:

- **In-situ statistics accumulated online**: mean profiles, Reynolds
  stresses, budget terms (channel); energy spectra (reusing the pencil FFTs)
  and dissipation/Taylor-scale histories (HIT). Cheap 1D/0D outputs every few
  steps.
- **Checkpoint/restart** via parallel HDF5 (HDF5.jl has MPI support), one
  logical file per checkpoint, layout-independent (restart on a different
  rank count / processor grid must work — write in global index space).
- Occasional full-field snapshots for visualization: same HDF5 path,
  optionally FP32-truncated.
- Processor-style hooks per time step, in the spirit of INS.jl `processors`,
  but mutating and rank-aware.

## 10. Validation and testing plan

- **Single-rank ≡ INS.jl**: on grids fitting one GPU, cross-validate fields
  and energy histories against IncompressibleNavierStokes.jl (same scheme,
  same RK tableau) to near machine precision.
- Discrete identities as tests: divergence-freeness after projection
  (machine precision), global energy conservation with ν = 0 (inviscid,
  periodic), skew-symmetry of convection operators on random fields.
- **Decomposition invariance**: identical results (up to FFT reordering
  roundoff) for 1, 2, 4, 8 ranks and different processor grids — the single
  most valuable CI test, runnable on CPU backend (§11).
- Physical validation: Moser-Kim-Mansour / Lee-Moser channel data at
  Re_τ = 180 and 550; HIT spectra against INS.jl and literature.
- Kernel-level tests run on CPU backend via KernelAbstractions; GPU CI on
  one device; multi-rank CI with MPI on CPU.

## 11. Prototyping and development order

1. **Layout layer first, no Navier-Stokes**: pencil descriptor + pack/unpack
   + transpose + halo exchange, plus distributed FFT → banded solve → inverse
   pipeline. Verify against a serial reference; benchmark transposes on
   2-8 real GPUs (borrowed cluster time) early — this is the
   get-right-first-time component.
2. 2nd-order HIT (triply periodic, uniform): full solver loop, energy
   conservation checks, spectra.
3. Channel: wall BCs, stretched y metrics, tridiagonal stage, forcing,
   statistics.
4. 4th order (uniform), banded generalization of the y-solve.
5. Semi-implicit y-diffusion; FP32 option; overlap of halos with interior
   compute — each behind the seams reserved above.

### Simulating multiple GPUs on one physical GPU (dev machine: 1× RTX 4090)

Multi-rank *correctness* is fully testable on one GPU; multi-rank
*performance* is not (no real interconnect — treat all timings as
meaningless). Three modes, all launched as ordinary `mpiexec -n N` runs:

- **CPU backend** (KernelAbstractions): N ranks, plain MPI, no GPU involved.
  Tests all decomposition/halo/transpose logic with arbitrary rank counts
  and processor grids. This is the CI workhorse.
- **N ranks sharing the one GPU, CUDA-aware MPI transport**: every rank does
  `CUDA.device!(0)`. Intra-device MPI transfers become device-to-device
  copies (CUDA IPC). Works for correctness out of the box; enable **MPS**
  (`nvidia-cuda-mps-control -d`) so kernels from different ranks overlap
  instead of time-slicing. Watch total memory: N ranks × their fields must
  fit in 24 GB.
- **NCCL transport: NOT testable this way.** NCCL does not support multiple
  ranks of one communicator on the same device (collectives hang; NVIDIA
  issue #418). NCCL's own Q2 2026 roadmap lists MPS-based GPU sharing as
  future work. Consequence for the design: the transport must be pluggable
  (§7), CUDA-aware MPI is the default/reference transport, and NCCL is an
  optimization validated only on real multi-GPU machines.

## 12. Open questions

- Package/repo name (current name is a placeholder).
- Processor-grid configuration UX: explicit in config vs. benchmark-script
  autotune output.
- Per-mode banded factorization storage vs. on-the-fly factorization
  (memory/compute trade, §5/§8).
- Whether PencilArrays.jl survives under the layout abstraction after
  benchmarking, or is prototype-only scaffolding (§7).
- Scalar transport (Boussinesq) timing: v1.x, once channel is validated.
