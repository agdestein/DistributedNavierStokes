# Code design

Status: draft, July 2026. Companion to [DESIGN.md](DESIGN.md) (physics/scope
decisions); this document specifies the code: abstractions, data layout,
function signatures, and the implementation of the communication layer.
Code sketches are illustrative, not final.

## 1. Principles

- **NamedTuples as namespaces, no structs unless dispatch demands one.**
  A velocity field is `u = (; x, y, z)`; the setup is a nested NamedTuple;
  a transpose plan is a NamedTuple. Target: **zero own struct definitions**
  in v1 (FFT plans etc. are library structs). NamedTuples are fully typed,
  so there is no performance cost; use function barriers where bodies get
  large, as in INS.jl.
- **Functions take `(data..., setup)`**, INS.jl style. Metadata lives in the
  setup, never on the arrays. All state-mutating functions end in `!` and
  return nothing.
- **The complexity budget is spent on communication** (layouts, transposes,
  halos). Physics kernels are plain KernelAbstractions stencils; time
  integration is a 20-line loop. Keep it that way.
- **Everything preallocated.** Setup constructs all buffers, plans, and
  coefficient arrays once; the time loop allocates nothing.

## 2. Dependencies

Hard: `MPI`, `KernelAbstractions`, `AbstractFFTs`, `FFTW` (CPU transforms),
stdlib (`LinearAlgebra`, `Random`, `Printf`).

Not dependencies: `CUDA` (the user loads it and passes
`backend = CUDABackend()`; CUDA.jl provides cuFFT through the AbstractFFTs
interface and `CuArray` allocation through
`KernelAbstractions.allocate(backend, T, dims...)` — the package never names
a GPU type). `PencilArrays` (layouts written from scratch, §5). `HDF5`
(checkpointing arrives later, as a package extension).

## 3. Package layout

```
src/
  DistributedNavierStokes.jl   # module, includes, exports
  layout.jl      # block distribution, pencil descriptors, MPI cartesian setup
  transpose.jl   # transpose plans, pack/unpack kernels, Alltoallv wrapper
  halo.jl        # exchange_halo!, wall ghost fills
  setup.jl       # setup(): grid, metrics, decomposition, plans, buffers
  operators.jl   # divergence!, momentum! (symmetry-preserving stencils), gradient update
  poisson.jl     # spectral pipeline, modified wavenumbers, batched Thomas/banded solve
  timestep.jl    # low-storage RK3, CFL, solve!
  forcing.jl     # channel bulk forcing, HIT spectral forcing
  fields.jl      # field allocation, initial conditions, reductions
  stats.jl       # online profiles / spectra (processors)
test/
  ...            # see §14
```

## 4. Fields and ghost layers

A scalar field is a plain 3D array (`Array` or `CuArray`) of local size
`ldims .+ 2w`, where `w` is the ghost width (1 for 2nd order, 2 for 4th;
**one global `w` for all fields**, taken from the scheme — uniform buffers
beat per-field generality). Velocity: `u = (; x, y, z)` of three such
arrays. Allocation:

```julia
scalarfield(setup) = KernelAbstractions.zeros(setup.backend, setup.T, setup.ldims .+ 2setup.w)
vectorfield(setup) = (; x = scalarfield(setup), y = scalarfield(setup), z = scalarfield(setup))
```

Kernels are launched over interior ranges with an offset origin, exactly the
INS.jl `I0` pattern. Spectral-space buffers (§9) carry **no ghosts** — no
stencils are applied in spectral space.

## 5. Decomposition and pencil layouts

`P = p1 × p2` ranks in a 2D processor grid (slab = `p2 == 1`, which needs no
special-casing anywhere). At any moment a distributed array is in one of
three *orientations*, described by `axes::NTuple{3,Int}`: `axes[d]` is the
processor-grid axis over which global dimension `d` is distributed
(`0` = local). With walls in y (dim 2), the three orientations are

```
:x  axes = (0, 2, 1)   # x local; z over axis 1, y over axis 2
:z  axes = (1, 2, 0)   # z local; x over axis 1, y over axis 2
:y  axes = (1, 0, 2)   # y local; x over axis 1, z over axis 2
```

Adjacency is `x ↔ z ↔ y`: a transpose swaps the local dimension with one
distributed dimension *along a single processor-grid axis* (`:x↔:z` within
axis-1 rows, `:z↔:y` within axis-2 columns). This adjacency is a deliberate
choice — it puts z, not y, in the middle, so the Poisson sequence
FFT(x) → FFT(z) → solve(y) is one transpose per arrow (§9).

A layout descriptor is data, not a type:

```julia
"Balanced block c (0-based) of n points over p ranks."
blockrange(n, p, c) = (c * n ÷ p + 1):((c + 1) * n ÷ p)

function layout(gdims, axes, procgrid, coords)
    ranges = ntuple(3) do d
        a = axes[d]
        a == 0 ? (1:gdims[d]) : blockrange(gdims[d], procgrid[a], coords[a])
    end
    (; gdims, axes, ranges, ldims = map(length, ranges))
end
```

`gdims` differs per pipeline stage (the rfft in x shrinks dim 1 to
`nx ÷ 2 + 1`), so descriptors are cheap values computed at setup for every
(stage, orientation) pair.

The MPI side: one `MPI.Cart_create(COMM_WORLD, (p1, p2); periodic)` cartesian
communicator, with `periodic[a]` set from the *native* orientation's mapping
(periodic global dims → `true`, the wall dim → `false`); `MPI.Cart_sub`
yields the row/column subcommunicators used by transposes. This communicator
is also the halo-exchange topology (§7) — `MPI.PROC_NULL` at wall ends is
what unifies walls and MPI interfaces.

**Native orientation is a config option** (`native = :x` default). Explicit
runs want `:x` (4-transpose Poisson). A future semi-implicit-y mode wants
`:y` (8-transpose Poisson, but the three y-Helmholtz solves and all wall
metrics become transpose-free). Kernels never know the orientation; they see
local arrays and ranges.

## 6. Transposes

A transpose plan is precomputed at setup. The key simplification: both
distributions are block-contiguous in global index space, so what rank r
must send to peer c is just the **intersection of r's source box with c's
destination box** — three `intersect(::UnitRange, ::UnitRange)` calls per
peer, no other index arithmetic.

```julia
function plan_transpose(src, dst, comm, procgrid, coords, backend, T)
    a = swapped_axis(src, dst)                       # 1 or 2
    subcomm = MPI.Cart_sub(comm, keep = a)           # my row/column, p_a ranks
    peers = 0:(procgrid[a] - 1)
    sendranges = [localize(src, intersect_box(src.ranges, peer_dst_ranges(dst, c))) for c in peers]
    recvranges = [localize(dst, intersect_box(dst.ranges, peer_src_ranges(src, c))) for c in peers]
    sendbuf = KernelAbstractions.allocate(backend, T, sum(length, sendranges))
    recvbuf = KernelAbstractions.allocate(backend, T, sum(length, recvranges))
    (; subcomm, sendranges, recvranges, sendbuf, recvbuf,
       sendcounts = length.(sendranges), recvcounts = length.(recvranges))
end

function transpose!(dstdata, srcdata, plan, backend)
    pack!(plan.sendbuf, srcdata, plan.sendranges, backend)   # KA kernel(s)
    KernelAbstractions.synchronize(backend)                  # before MPI touches the buffer!
    MPI.Alltoallv!(MPI.VBuffer(plan.sendbuf, plan.sendcounts),
                   MPI.VBuffer(plan.recvbuf, plan.recvcounts), plan.subcomm)
    unpack!(dstdata, plan.recvbuf, plan.recvranges, backend)
end
```

Notes:

- Six plans total (`:x↔:z` and `:z↔:y`, forward and back, for the complex
  stage sizes; the real stage never transposes). Plans for different stages
  sharing shapes reuse buffers.
- `pack!`/`unpack!` are KernelAbstractions kernels copying strided boxes
  to/from contiguous buffer segments. v1 may use one kernel launch per peer
  (`p_a` launches); a single fused kernel over all segments is a recorded
  optimization, not a v1 requirement.
- **Synchronize the backend between kernel launches and MPI calls** — KA
  kernels are asynchronous; this is the classic GPU-aware-MPI correctness
  trap, so it lives inside `transpose!`, not in caller code.
- The self-peer segment goes through `Alltoallv!` like any other (MPI
  self-sends are memcpys); an explicit device-copy fast path is a recorded
  optimization.
- Transport is pluggable behind `transpose!`'s MPI call: CUDA-aware MPI is
  the default and reference; an NCCL path (via NCCL.jl `ncclAllToAll`-style
  grouped send/recv) can be added later without touching callers. Same for
  the halo exchange.

## 7. Halo exchange and boundary conditions

One function is both "apply BCs" and "exchange halos":

```julia
function exchange_halo!(φ, comp, setup)      # comp ∈ (:x, :y, :z, :p) — staggering
    (; comm, axes, w) = setup.native
    for d in 1:3                             # fixed order; sent slabs include
        a = axes[d]                          # halos filled by earlier d ⇒ corners free
        if a == 0                            # dimension local to this rank
            fill_local!(φ, d, comp, setup)   # periodic wrap copy, or wall kernel
        else
            lo, hi = MPI.Cart_shift(comm, a - 1, 1)   # PROC_NULL at wall ends
            haloswap!(φ, d, lo, hi, setup)   # pack faces, Sendrecv! both ways
            lo == MPI.PROC_NULL && fill_wall!(φ, d, :lo, comp, setup)
            hi == MPI.PROC_NULL && fill_wall!(φ, d, :hi, comp, setup)
        end
    end
end
```

- The cartesian communicator's periodicity does the bookkeeping: periodic
  dims wrap (interior ranks and boundary ranks are indistinguishable), the
  wall dim yields `MPI.PROC_NULL` neighbors exactly where a physical wall
  fill is needed. **This is the "BC ≡ MPI interface" abstraction, and it is
  ~15 lines.**
- Sent face slabs have width `w` and full extent (including ghosts) in the
  other two dimensions; exchanging dimensions sequentially then fills edge
  and corner ghosts with no diagonal communication.
- `fill_wall!` dispatches on staggering: wall-normal component (`u.y`): the
  face DOF on the wall is set to the wall velocity and ghosts are filled
  antisymmetrically; tangential components: `ghost = 2u_wall - mirror`;
  pressure: symmetric (homogeneous Neumann). Same conventions as INS.jl.
- Face buffers are packed contiguously (small, preallocated); like the
  transposes, synchronize before `Sendrecv!`.
- Velocity convenience wrapper: `exchange_halo!(u, setup)` loops the three
  components. Interior/boundary kernel splitting for comm/compute overlap is
  a reserved seam (kernels already take explicit index ranges), not v1.

## 8. FFTs

- Plans from `AbstractFFTs`: `plan_rfft(buf, 1)` (real, x), `plan_fft!(buf, 3)`
  (complex in-place, z), plus inverses. FFTW on CPU, cuFFT when the buffers
  are `CuArray`s — the package code is identical.
- Arrays keep natural `(i, j, k)` memory order in every orientation; cuFFT
  and FFTW both handle strided batched 1D transforms along dim 3. If
  profiling shows dim-3 transforms are slow, the recorded fallback is to let
  `unpack!` permute so the transform dimension is contiguous — a local
  change inside the transpose layer.
- Transforms run on ghost-free pipeline buffers. The divergence kernel
  writes its result directly into the real pipeline buffer; the returned
  pressure is copied into the ghosted `p` array and halo-exchanged before
  the gradient update.

## 9. Poisson solve

Pipeline (native `:x`, channel; the fully periodic case swaps step 5 for a
local FFT in y + pointwise division, a functor slot chosen at setup):

```julia
function poisson!(p, r, ps, setup)           # ps = setup.poisson
    mul!(ps.â, ps.plan_x, r)                 # 1. rfft x     — :x pencil, real → complex
    transpose!(ps.b̂, ps.â, ps.x2z, backend)  # 2.
    ps.plan_z! * ps.b̂                        # 3. fft z      — :z pencil, in place
    transpose!(ps.â, ps.b̂, ps.z2y, backend)  # 4.
    solve_y!(ps.â, ps, setup)                # 5. banded y   — :y pencil (work: ps.b̂ is free)
    transpose!(ps.b̂, ps.â, ps.y2z, backend)  # 6.
    ps.iplan_z! * ps.b̂                       # 7. ifft z
    transpose!(ps.â, ps.b̂, ps.z2x, backend)  # 8.
    mul!(ps.rbuf, ps.iplan_x, ps.â)          # 9. irfft x
    copy_interior!(p, ps.rbuf, setup)        #    into ghosted p
end
```

Two complex pencil-sized buffers (`â`, `b̂`) ping-pong through the whole
pipeline; the Thomas work array aliases the idle one. Four transposes.

`solve_y!`: one GPU thread per local `(kx, kz)` mode, Thomas algorithm along
the rank-local y line:

```julia
@kernel function thomas_kernel!(p̂, work, ay, by, cy, λx, λz, ny)
    i, k = @index(Global, NTuple)
    λ = λx[i] + λz[k]
    # forward elimination and back substitution over j = 1:ny,
    # diagonal by[j] - λ; work[i, :, k] holds the modified coefficients
end
```

- `λx`, `λz`: modified wavenumbers of the scheme's divergence/gradient
  stencils (2nd order: `2(cos θ - 1)/Δ²`; 4th order: the wider stencil's
  symbol), restricted to this rank's spectral ranges. Small 1D vectors.
- `ay, by, cy`: the stretched-grid wall-normal Laplacian with the staggered
  homogeneous-Neumann closure built into the end rows. 1D, length `ny`.
- Coefficients are formed on the fly per thread from 1D data — **no stored
  per-mode factorizations** (resolves the open question in DESIGN.md §12:
  memory wins; the Thomas sweep is bandwidth-bound on `p̂` anyway).
- The mean mode `(kx, kz) = (0, 0)` is singular (all-Neumann): the owning
  rank pins one row (`p̂₁ = 0`).
- 4th order replaces the kernel body with a bandwidth-`b` elimination; the
  slot is `solve_y!`, sized by `b`, with Thomas as the `b = 1` case.

## 10. Operators

Symmetry-preserving stencils as KernelAbstractions kernels, INS.jl
conventions (`u.x[I]` on the right x-face of volume `I`; divergence =
backward difference):

```julia
divergence!(r, u, setup)          # into the real Poisson buffer, interior only
momentum!(du, u, setup)           # convection + diffusion + forcing, one fused kernel
pressuregrad_update!(u, p, setup) # u ← u - Δt/ρ ∇p, staggered
```

- `momentum!` is **one fused kernel** writing all of `du.x, du.y, du.z`:
  the stencil is bandwidth-bound and the three components read the same
  neighborhoods of `u`. Metrics enter as 1D arrays (`Δy`, `Δyu` over the
  local y-range plus ghosts); uniform directions are scalars.
- Scheme order selects the stencil functor (and `w`) at setup; the kernel
  code is shared via `@inline` flux functions, not duplicated per order.
- Correctness oracle: INS.jl's operators on identical single-rank setups.

## 11. Time integration and projection

Low-storage RK3 in Williamson 2N form — two registers total (`u` and `q`),
coefficients `(A_i, B_i)` equivalent to Wray's scheme:

```julia
function step!(state, setup, Δt)
    (; u, q, p) = state
    for (A, B) in setup.rk
        foreach((:x, :y, :z)) do c
            @. q[c] = A * q[c] + Δt * du_from(momentum!)[c]   # sketch; fused in practice
            @. u[c] += B * q[c]
        end
        project!(u, p, setup)      # divergence! → poisson! → halo(p) → pressuregrad_update!
        exchange_halo!(u, setup)
    end
end
```

- One projection and one velocity halo exchange per substage; `momentum!`
  needs valid halos, which the end of the previous substage guarantees.
- `Δt` from CFL: local `maximum(abs, u.x ./ Δx) ...` (mapreduce on the
  device) + `MPI.Allreduce(max)`. Recomputed every `n_cfl` steps.
- The semi-implicit-y seam: `step!` is selected at setup
  (`step_explicit!` / later `step_imex!`); the IMEX variant inserts three
  y-Helmholtz solves reusing `solve_y!` machinery. Nothing else changes.

## 12. User-facing API

```julia
using MPI, CUDA, DistributedNavierStokes

s = setup(;
    n = (3072, 1536, 2304),            # global grid
    lims = ((0, 8π), (-1, 1), (0, 3π)),
    bc = (:periodic, :wall, :periodic),
    stretch = tanh_stretch(1.8),       # y face distribution, 1D
    visc = 1 / 5200,
    order = 2,
    procgrid = (8, 16),                # or `:auto` (near-square heuristic; see §15)
    native = :x,
    backend = CUDABackend(),
    T = Float64,
)

u = velocityfield(s) do x, y, z        # broadcast over local coordinates + noise/spectrum helpers
    ...
end

solve!(;
    u, setup = s, tlims = (0.0, 100.0),
    Δt = nothing,                      # nothing ⇒ CFL-adaptive
    processors = (;
        log  = logger(; nupdate = 10),           # rank-0 printing
        stats = channelstats(s; nupdate = 20),   # online profiles/stresses
        spec = spectrum(s; nupdate = 100),       # reuses Poisson FFT plans
    ),
)
```

- `setup` returns the one nested NamedTuple: grid + metrics, communicator
  and native-layout info, transpose plans, FFT plans, Poisson coefficients,
  RK coefficients, preallocated buffers, `backend`, `workgroupsize`.
- Processors are `(state, setup) -> nothing` closures, rank-aware, INS.jl
  spirit but mutating. Reductions inside them use the same
  mapreduce + `Allreduce` pattern.
- Scripts launch as `mpiexec -n P julia --project script.jl`; each rank
  picks its GPU by `CUDA.device!(local_rank)` (or all `device!(0)` for the
  single-GPU multi-rank testing mode, DESIGN.md §11).

## 13. Memory ledger (field-sized allocations)

| What | Count (real-field equivalents) |
|---|---|
| `u` | 3 |
| `q` (RK 2N register) | 3 |
| `p` | 1 |
| real Poisson buffer (`r`/`rbuf`, shared) | 1 |
| complex pipeline buffers `â`, `b̂` | ≈ 2 |
| transpose send/recv buffers | ≈ 2 |
| **Total** | **≈ 12** |

Matches the DESIGN.md §8 budget: 3000³ FP64 on 64 GPUs comfortable, 4000³
needs 128 GPUs. Any new persistent 3D array must argue against this table.
(Recorded optimizations if needed: alias transpose buffers with `b̂`;
FP32 fields with FP64 solve.)

## 14. Testing strategy

Layer-by-layer, mostly on the CPU backend with `mpiexec -n {1,2,4,8}`:

1. **layout/transpose**: round trip `transpose!` chains ≡ identity;
   distributed-FFT of a random field ≡ serial FFT of the gathered field;
   all processor grids incl. slabs (`p2 = 1`) and `P = 1`.
2. **halo**: ghost values ≡ serial periodic/wall fills for every staggering,
   every decomposition.
3. **poisson**: `L p = r` residual at machine precision; mean-mode handling;
   agreement with a dense/serial reference on small grids.
4. **operators**: single-rank ≡ INS.jl to machine precision; discrete
   identities (skew-symmetry of convection against random fields, inviscid
   energy conservation, divergence-freeness after projection).
5. **solver**: decomposition invariance — identical trajectories (to
   roundoff) across rank counts and processor grids; Taylor-Green order
   verification (2nd/4th); short channel + HIT regression checksums.
6. GPU CI: single device runs of 1-4 above; multi-rank-one-GPU (MPS) as a
   manual pre-cluster gate.

## 15. Deferred (seams reserved, not built)

- Comm/compute overlap (interior/boundary kernel split exists; async halo
  path does not).
- NCCL transport behind `transpose!`/`haloswap!`.
- `procgrid = :auto` beyond the near-square heuristic: a benchmark script
  timing the 4-transpose pipeline over candidate grids, output pasted into
  the config (replaces cuDecomp autotuning).
- IMEX y-diffusion (`step_imex!`), scalar transport slot in `momentum!`,
  HDF5 checkpoint extension, FP32 field mode.
