# Filtered-field / sub-filter-stress research output (DESIGN_SPECTRAL.md §8),
# in SymmetryCode's exact artifact schemas so the downstream training and
# analysis pipeline applies unchanged. Division of labor:
#
# - Distributed (all ranks): one extra nonlinearity evaluation on the
#   current state (reusing the RHS product pipeline and its buffers — no new
#   full-size fields), then a Gatherv of only the LES-cube modes
#   (|k_i| ≤ nles/2, a few hundred MB at most) of û and the six σ̂ = F[v_i v_j]
#   to rank 0.
# - Rank 0 (coarse rfft arrays, plain loops + FFTW): SymmetryCode's `sfs!`
#   ported verbatim — test filter with forced-shell protection,
#   2/3 truncation (which also zeroes the mean, as in SymmetryCode), LES-grid
#   nonlinearity, τ = filter(σ) − ū⊗ū made trace-free — plus `filter_reynolds`
#   and the shell spectrum, then the JLD2 writers (`fields.jld2`,
#   `les_meta.jld2`, `dns_meta.jld2`).
#
# The bank generalizes SymmetryCode's (Gaussian, one M) recipe along the
# free axes of the data campaign: a bank *cell* is `(; M, kernel, Δfac)` —
# coarse grid M, spectral kernel (:gaussian, :cutoff, :tophat, :helmholtz),
# width Δ = Δfac·l/M — and one gather at the largest M feeds every cell
# (smaller-M coarse arrays are index-extracted on rank 0, identical to a
# direct gather at that M). All kernels share the low-k carve-out
# (identity below shell protectshells+1) so every kernel commutes with the
# shell forcing. A single-M all-Gaussian bank keeps SymmetryCode's
# flat `delta=<Δf>/` layout (plus new metadata keys); any other bank nests cells as
# `filter=<kernel>/M=<M>/delta=<Δf>/`, so the downstream loader pointed at
# one (kernel, M) directory works unchanged. The same sampling/writing
# core runs in-situ ([`sfswriter`](@ref)) or as post-processing over
# stored raw snapshots ([`sfs_offline`](@ref)), where the kinematic-null
# variant is a [`spectral_phaserandomize!`](@ref) pass before filtering.
#
# Conventions shared with SymmetryCode: û = F[u]/n³ (both codes), coarse
# arrays are full rfft layout (nles÷2+1, nles, nles) with wrapped negative
# frequencies, vector fields are (; x, y, z), symmetric tensors
# (; xx, yy, zz, xy, yz, zx). The state is assumed mean-free (û(0) = 0 —
# true for the random IC, Taylor-Green, and their evolution): SymmetryCode's
# `nonlinearity!` zero-means its input via `twothirds!`, this pipeline does
# not, so a nonzero mean would enter the products here but not there.

"fftfreq-style integer frequency of index `i` on an `n` grid."
lesfreq(i, n) = i - 1 - ifelse(i ≤ (n + 1) >> 1, 0, n)

# --- rank-0 coarse-grid kernels (ported from SymmetryCode solver.jl/filtering.jl) ---

"2/3 truncation on a coarse rfft array: zero `|k| > n÷3` *and* the mean."
function les_twothirds!(a, n)
    kc = n ÷ 3
    for k in axes(a, 3), j in axes(a, 2), i in axes(a, 1)
        k1, k2, k3 = i - 1, lesfreq(j, n), lesfreq(k, n)
        keep = (k1 ≤ kc) & (abs(k2) ≤ kc) & (abs(k3) ≤ kc) & ((k1, k2, k3) != (0, 0, 0))
        keep || (a[i, j, k] = zero(eltype(a)))
    end
    a
end

"`sin(x)/x` continued through 0."
les_sinc(x) = ifelse(x == 0, one(x), sin(x) / x)

"""
Spectral test filter of width `Δ` on a coarse rfft array, leaving the
forced shells (`|k_int| ≤ protectshells`) untouched (the low-k carve-out,
as in SymmetryCode's `gaussianfilter!`, which hardcodes `nshell = 2`;
uniform across kernels so every kernel commutes with the shell forcing).
Kernels: `:gaussian` `exp(-Δ²k²/24)`; `:cutoff` sharp at `|k| ≤ π/Δ`;
`:tophat` the box filter `∏ᵢ sinc(kᵢΔ/2)` (negative lobes intended);
`:helmholtz` the differential filter `1/(1 + Δ²k²/24)` (matches the
Gaussian to second order, algebraically invertible).
"""
function les_filterkernel!(a, n, l, Δ, protectshells, kernel)
    kf = 2π / l
    kbound2 = (kf * (protectshells + 1))^2
    for k in axes(a, 3), j in axes(a, 2), i in axes(a, 1)
        k1, k2, k3 = kf * (i - 1), kf * lesfreq(j, n), kf * lesfreq(k, n)
        kk = k1^2 + k2^2 + k3^2
        kk < kbound2 && continue
        w =
            kernel == :gaussian ? exp(-Δ^2 * kk / 24) :
            kernel == :cutoff ? ifelse(kk ≤ (π / Δ)^2, 1.0, 0.0) :
            kernel == :tophat ?
            les_sinc(k1 * Δ / 2) * les_sinc(k2 * Δ / 2) * les_sinc(k3 * Δ / 2) :
            kernel == :helmholtz ? 1 / (1 + Δ^2 * kk / 24) :
            error("unknown filter kernel $kernel")
        a[i, j, k] *= w
    end
    a
end

"LES-grid nonlinearity: `σ2[c] = F[v_i v_j]/n³` from dealiased `ub` (6 comps)."
function les_nonlinearity!(σ2, vs, vv, tmp, ub, plan, n)
    fac = n^3
    for i = 1:3
        copyto!(tmp, ub[i])
        les_twothirds!(tmp, n)
        ldiv!(vs[i], plan, tmp)
        vs[i] .*= fac
    end
    for (c, (i, j)) in enumerate(((1, 1), (2, 2), (3, 3), (1, 2), (2, 3), (3, 1)))
        @. vv = vs[i] * vs[j]
        mul!(σ2[c], plan, vv)
        σ2[c] ./= fac
    end
    σ2
end

"Global filter Reynolds number `Re_Δ = Δ²√⟨|∇ū|²⟩/ν` (ū already dealiased)."
function les_filter_reynolds(ub, n, l, visc, Δ)
    kf = 2π / l
    a2 = 0.0
    for k in axes(ub[1], 3), j in axes(ub[1], 2), i in axes(ub[1], 1)
        kk = kf^2 * ((i - 1)^2 + lesfreq(j, n)^2 + lesfreq(k, n)^2)
        u2 = abs2(ub[1][i, j, k]) + abs2(ub[2][i, j, k]) + abs2(ub[3][i, j, k])
        w = ifelse(1 < i < size(ub[1], 1), 2, 1)
        a2 += w * kk * u2
    end
    Δ^2 * sqrt(a2) / visc
end

"Shell-binned energy spectrum over shells `1:n÷3` (SymmetryCode binning)."
function les_spectrum(ub, n)
    kc = n ÷ 3
    E = zeros(kc)
    for k in axes(ub[1], 3), j in axes(ub[1], 2), i in axes(ub[1], 1)
        s2 = (i - 1)^2 + lesfreq(j, n)^2 + lesfreq(k, n)^2
        sh = isqrt(s2)
        1 ≤ sh ≤ kc || continue
        u2 = abs2(ub[1][i, j, k]) + abs2(ub[2][i, j, k]) + abs2(ub[3][i, j, k])
        w = ifelse(1 < i < size(ub[1], 1), 2, 1)
        E[sh] += w * u2 / 2
    end
    E
end

"Subtract the trace from the diagonal of a 6-component symmetric tensor."
function les_tracefree!(σ)
    xx, yy, zz = σ[1], σ[2], σ[3]
    for idx in eachindex(xx)
        tr = (xx[idx] + yy[idx] + zz[idx]) / 3
        xx[idx] -= tr
        yy[idx] -= tr
        zz[idx] -= tr
    end
    σ
end

# --- the LES-cube gather ---

# The LES cube |k_i| ≤ nles/2 of the truncated state (y-pencil,
# gdims (kcut+1, m, m), compact frequency order). Global index segments:
# dim 1 is kx = 0..nles/2 (contiguous), dims 2-3 split into a positive
# (k = 0..nles/2-1, indices 1..nles/2) and a negative
# (k = -nles/2..-1, indices m+1-nles/2..m) segment — up to 4 boxes per rank
# (dim 2 is rank-local). Boxes are enumerated in a fixed deterministic order
# on every rank, so rank 0 reconstructs each peer's boxes without metadata.

"Coarse rfft-array index of compact global index `g` (dims 2-3)."
lescoarse(g, m, nles) = g ≤ nles ÷ 2 ? g : g - m + nles

"LES-cube boxes (global state indices) of the rank owning `ranges`."
function lesboxes(ranges, m, nles)
    b1 = intersect(ranges[1], 1:(nles÷2+1))
    seg23 = (1:(nles÷2), (m+1-nles÷2):m)
    [
        (b1, b2, b3) for
        b3 in map(sg -> intersect(ranges[3], sg), seg23), b2 in seg23 if
        !isempty(b1) && !isempty(b2) && !isempty(b3)
    ]
end

"""
Extract the M-cube of a coarse rfft array on grid `Mmax` into `dst` (grid
`M ≤ Mmax`, both with wrapped negative frequencies). Identical to gathering
the M-cube directly.
"""
function les_extract!(dst, src, M, Mmax)
    for k = 1:M, j = 1:M, i = 1:(M÷2+1)
        js = j ≤ M ÷ 2 ? j : j - M + Mmax
        ks = k ≤ M ÷ 2 ? k : k - M + Mmax
        dst[i, j, k] = src[i, js, ks]
    end
    dst
end

const LES_KERNELS = (:gaussian, :cutoff, :tophat, :helmholtz)

"""
Normalize a bank-cell spec: an iterable of NamedTuples with `M` (coarse
grid), `kernel` (see [`les_filterkernel!`](@ref)), `Δfac` (`Δ = Δfac·l/M`),
and optionally `Δη` (the Δ/η label the width was pinned from, recorded in
the metadata). The legacy form `nles` + `filters` means single-M Gaussian
cells (SymmetryCode-compatible).
"""
function lescells(; nles = nothing, filters = nothing, cells = nothing)
    if cells === nothing
        (nles === nothing || filters === nothing) &&
            error("pass `cells`, or the legacy `nles` + `filters`")
        cells = [(; M = nles, kernel = :gaussian, Δfac = f) for f in filters]
    end
    [
        (;
            M = Int(c.M), kernel = Symbol(c.kernel), Δfac = Float64(c.Δfac),
            Δη = get(c, :Δη, nothing),
        ) for c in cells
    ]
end

"""
    etacells(; deta, eta, l, Ms, kernels = (:gaussian,), window = (2.0, 5.0))

Bank cells with widths pinned in `Δ/η` (`deta` list) at measured Kolmogorov
length `eta`: for each width × kernel, every coarse grid in `Ms` whose
resulting `Δ/h = Δη·η·M/l` lands inside `window` carries the column (the
choose-M-per-column design; widths rounded to 4 significant digits so the
directory label equals the value used).
"""
etacells(; deta, eta, l, Ms, kernels = (:gaussian,), window = (2.0, 5.0)) = [
    (; M, kernel = Symbol(kernel), Δfac = round(de * eta * M / l; sigdigits = 4), Δη = Float64(de))
    for de in deta for kernel in kernels for M in sort(collect(Ms)) if
    window[1] - 1e-9 ≤ de * eta * M / l ≤ window[2] + 1e-9
]

"""
Gather plan for the LES cube + rank-0 per-M coarse-grid workspaces. Pass
`cells` (see [`lescells`](@ref)) or legacy `nles` + `filters`; the gather
runs once at the largest M. `outtype(M)` sets the element type of the
stored cell outputs (e.g. `M -> M ≥ 256 ? Float32 : Float64`); raw states
are unaffected.
"""
function lessampler(s; nles = nothing, filters = nothing, cells = nothing,
    protectshells = 2, outtype = _ -> Float64)
    (; kcut, m, T, topo, lspec) = s
    cells = lescells(; nles, filters, cells)
    for c in cells
        c.M % 2 == 0 || error("coarse grid M must be even (got $(c.M))")
        c.kernel in LES_KERNELS || error("kernel must be one of $LES_KERNELS")
        c.Δfac > 0 || error("Δfac must be positive")
    end
    nles = maximum(c -> c.M, cells)
    nles ÷ 2 ≤ kcut || error("LES cube needs M/2 ≤ kcut (M = $nles, kcut = $kcut)")
    myboxes = lesboxes(lspec.ranges, m, nles)
    # Per-rank counts and (on rank 0) global boxes, from the deterministic
    # layout. MPI cartesian ranks are row-major: rank = c1·p2 + c2.
    allboxes = map(0:(prod(topo.procgrid)-1)) do r
        coords = (r ÷ topo.procgrid[2], r % topo.procgrid[2])
        lesboxes(layout((kcut + 1, m, m), AXES.y, topo.procgrid, coords).ranges, m, nles)
    end
    counts = map(bs -> sum(b -> prod(length, b), bs; init = 0), allboxes)
    sendbuf = Vector{Complex{T}}(undef, counts[topo.rank+1])
    root = topo.rank == 0
    coarse3(M) = ntuple(_ -> Array{Complex{T},3}(undef, M ÷ 2 + 1, M, M), 3)
    coarse6(M) = (coarse3(M)..., coarse3(M)...)
    phys3(M) = ntuple(_ -> Array{T,3}(undef, M, M, M), 3)
    ub = root ? coarse3(nles) : nothing        # gathered û (unfiltered)
    σb = root ? coarse6(nles) : nothing        # gathered σ̂ (6)
    ws = root ? Dict(
        M => (;
            ub = M == nles ? ub : coarse3(M),  # M-cube of û (extracted)
            σb = M == nles ? σb : coarse6(M),
            ubf = coarse3(M),                  # per-cell ū workspace
            σ2 = coarse6(M),                   # LES nonlinearity
            τ = coarse6(M),
            vs = phys3(M),
            vv = Array{T,3}(undef, M, M, M),
            tmp = Array{Complex{T},3}(undef, M ÷ 2 + 1, M, M),
            plan = plan_rfft(Array{T,3}(undef, M, M, M)),
        ) for M in unique(c.M for c in cells)
    ) : nothing
    (;
        nles, cells, protectshells, outtype, myboxes, allboxes, counts,
        sendbuf,
        recvbuf = root ? Vector{Complex{T}}(undef, sum(counts)) : nothing,
        ub, σb, ws,
    )
end

"""
Gather component `c` of the state-shaped array `a` into the rank-0 coarse
rfft array `dst` (a no-op on other ranks). Collective.
"""
function les_gather!(dst, a, c, sam, s)
    (; m, topo, lspec) = s
    off = 0
    for (b1, b2, b3) in sam.myboxes
        loc = map((b, r) -> b .- (first(r) - 1), (b1, b2, b3), lspec.ranges)
        n = prod(length, (b1, b2, b3))
        copyto!(view(sam.sendbuf, (off+1):(off+n)), vec(Array(view(a, loc..., c))))
        off += n
    end
    MPI.Gatherv!(
        sam.sendbuf,
        topo.rank == 0 ? MPI.VBuffer(sam.recvbuf, sam.counts) : nothing,
        topo.cart,
    )
    if topo.rank == 0
        off = 0
        for boxes in sam.allboxes, (b1, b2, b3) in boxes
            dims = map(length, (b1, b2, b3))
            n = prod(dims)
            dst[b1, lescoarse.(b2, m, sam.nles), lescoarse.(b3, m, sam.nles)] .=
                reshape(view(sam.recvbuf, (off+1):(off+n)), dims)
            off += n
        end
    end
    dst
end

"""
Sample the SFS data for the current state: distributed products + gather,
then the rank-0 `sfs!` port per bank cell. Returns, on rank 0, a vector over
`sam.cells` of `(; ubar, τ, redelta, spectrum)` (freshly allocated copies,
element type `sam.outtype(M)`); `nothing` on other ranks. Collective;
clobbers the RHS pipeline buffers.
"""
function sfs_sample!(sam, uh, s)
    (; l, visc, T) = s
    f = s.fft
    root = s.topo.rank == 0
    # ū source: the LES cube of û itself (sharp cutoff = plain truncation)
    for c = 1:3
        les_gather!(root ? sam.ub[c] : nothing, uh, c, sam, s)
    end
    # σ̂ = F[v_i v_j] on the DNS grid, LES cube only (same two product
    # batches as spectral_rhs!)
    spec_to_phys!(f.v, uh, s)
    @. f.rbuf = f.v * f.v
    phys_to_spec!(f.σh, f.rbuf, s)
    for c = 1:3
        les_gather!(root ? sam.σb[c] : nothing, f.σh, c, sam, s)
    end
    @views begin
        @. f.rbuf[:, :, :, 1] = f.v[:, :, :, 1] * f.v[:, :, :, 2]   # xy
        @. f.rbuf[:, :, :, 2] = f.v[:, :, :, 2] * f.v[:, :, :, 3]   # yz
        @. f.rbuf[:, :, :, 3] = f.v[:, :, :, 3] * f.v[:, :, :, 1]   # zx
    end
    phys_to_spec!(f.σh, f.rbuf, s)
    for c = 1:3
        les_gather!(root ? sam.σb[3+c] : nothing, f.σh, c, sam, s)
    end
    root || return nothing
    (; nles, protectshells, ws) = sam
    # smaller-M cubes, extracted once per unique M
    for (M, w) in ws
        M == nles && continue
        for c = 1:3
            les_extract!(w.ub[c], sam.ub[c], M, nles)
        end
        for c = 1:6
            les_extract!(w.σb[c], sam.σb[c], M, nles)
        end
    end
    map(sam.cells) do cell
        (; M, kernel, Δfac) = cell
        w = ws[M]
        Δ = Float64(Δfac * l / M)
        # ū = filter(cutoff(û)), dealiased; σbar1 = filter(cutoff(σ̂))
        for c = 1:3
            copyto!(w.ubf[c], w.ub[c])
            les_filterkernel!(w.ubf[c], M, l, Δ, protectshells, kernel)
            les_twothirds!(w.ubf[c], M)
        end
        for c = 1:6
            copyto!(w.τ[c], w.σb[c])   # τ starts as σbar1
            les_filterkernel!(w.τ[c], M, l, Δ, protectshells, kernel)
        end
        # τ = σbar1 - F[ū_i ū_j], dealiased, trace-free
        les_nonlinearity!(w.σ2, w.vs, w.vv, w.tmp, w.ubf, w.plan, M)
        for c = 1:6
            w.τ[c] .-= w.σ2[c]
            les_twothirds!(w.τ[c], M)
        end
        les_tracefree!(w.τ)
        conv(a) = Complex{sam.outtype(M)}.(a)
        (;
            ubar = (; x = conv(w.ubf[1]), y = conv(w.ubf[2]), z = conv(w.ubf[3])),
            τ = (;
                xx = conv(w.τ[1]), yy = conv(w.τ[2]), zz = conv(w.τ[3]),
                xy = conv(w.τ[4]), yz = conv(w.τ[5]), zx = conv(w.τ[6]),
            ),
            redelta = les_filter_reynolds(w.ubf, M, l, Float64(visc), Δ),
            spectrum = les_spectrum(w.ubf, M),
        )
    end
end

"Write a multi-key JLD2 archive atomically (write sibling temp, then rename)."
function jldsave_atomic(file; kwargs...)
    tmp = tempname(dirname(file); cleanup = false)
    try
        JLD2.jldsave(tmp; kwargs...)
        mv(tmp, file; force = true)
    catch
        isfile(tmp) && rm(tmp; force = true)
        rethrow()
    end
    nothing
end

"""
Directory of a bank cell under `dir`: a single-M all-Gaussian bank keeps
SymmetryCode's flat `delta=<Δf>/`; any other bank nests as
`filter=<kernel>/M=<M>/delta=<Δf>/` (the downstream loader pointed at one
`filter=…/M=…/` directory sees the flat layout it expects).
"""
celldir(dir, cell, legacy) =
    legacy ? joinpath(dir, "delta=$(cell.Δfac)") :
    joinpath(dir, "filter=$(cell.kernel)", "M=$(cell.M)", "delta=$(cell.Δfac)")

"""
Sample accumulator + writer shared by the in-situ [`sfswriter`](@ref) and
the offline [`sfs_offline`](@ref): `record!(t, uh)` samples the bank
(collective), `finish!()` writes the artifact archives (rank 0, atomic).
"""
function sfscollector(sam, s, dir)
    (; cells) = sam
    tstart = time()
    tsamples = Float64[]
    spectra_dns = Vector{Float64}[]
    statistics_dns = []
    inputs = [NamedTuple[] for _ in cells]
    outputs = [NamedTuple[] for _ in cells]
    redelta = [Float64[] for _ in cells]
    spectra_les = [Vector{Float64}[] for _ in cells]
    record! = (t, uh) -> begin
        st = spectral_stats(uh, s)
        sp = spectral_spectrum(uh, s)
        samples = sfs_sample!(sam, uh, s)
        if s.topo.rank == 0
            push!(tsamples, t)
            push!(spectra_dns, Float64.(sp.E[2:end]))   # shells 1:kcut
            push!(statistics_dns, st)
            for (k, smp) in enumerate(samples)
                push!(inputs[k], smp.ubar)
                push!(outputs[k], smp.τ)
                push!(redelta[k], smp.redelta)
                push!(spectra_les[k], smp.spectrum)
            end
        end
        nothing
    end
    finish! = () -> begin
        s.topo.rank == 0 && !isempty(tsamples) || return nothing
        jldsave_atomic(
            joinpath(mkpath(dir), "dns_meta.jld2");
            times = tsamples, spectra_dns,
            statistics_dns = [statistics_dns...],
            t_int = statistics_dns[1].t_int, walltime = time() - tstart,
        )
        legacy = all(c -> c.kernel == :gaussian && c.M == sam.nles && c.Δη === nothing, cells)
        for (k, cell) in enumerate(cells)
            ddir = mkpath(celldir(dir, cell, legacy))
            jldsave_atomic(
                joinpath(ddir, "fields.jld2");
                inputs = [inputs[k]...], outputs = [outputs[k]...],
                redelta = redelta[k],
                Δ = Float64(cell.Δfac * s.l / cell.M), Δ_factor = cell.Δfac,
                visc = Float64(s.visc),
                kernel = String(cell.kernel), M = cell.M, delta_eta = cell.Δη,
            )
            jldsave_atomic(
                joinpath(ddir, "les_meta.jld2");
                spectra_les = spectra_les[k],
                redelta_mean = sum(redelta[k]) / length(redelta[k]),
            )
        end
        nothing
    end
    (; record!, finish!)
end

"""
    sfswriter(s; dir, times, nles, filters, cells, protectshells = 2,
              outtype = _ -> Float64)

Processor for [`spectral_solve!`](@ref) that samples filtered velocities and
sub-filter stresses at each entry of `times` (pass the same `times` as
`tstops` so steps land on them exactly) and, after the last sample, writes
SymmetryCode's artifact schemas under `dir` (rank 0, atomic):

- `dns_meta.jld2`: `times`, `spectra_dns`, `statistics_dns`, `t_int`,
  `walltime` (bank-independent DNS metadata).
- per bank cell (see [`lescells`](@ref) for `cells` vs the legacy
  `nles` + `filters`; directory layout: [`celldir`](@ref)):
  `…/fields.jld2` with `inputs` (ū, spectral, `(; x, y, z)` per snapshot),
  `outputs` (τ, trace-free, `(; xx, yy, zz, xy, yz, zx)`), `redelta`, `Δ`,
  `Δ_factor`, `visc`, plus `kernel`, `M`, `delta_eta`;
  `…/les_meta.jld2` with `spectra_les`, `redelta_mean`.

Every cell needs `M/2 ≤ kcut`; `Δ = Δfac·l/M`. The state must be mean-free
(see the module header note).

Samples accumulate in memory and the artifacts are written only after the
*last* time — this processor does not survive checkpoint restarts. For
runs long enough to need the checkpointer, save raw snapshots
([`snapshotsaver`](@ref)) and build the bank offline
([`sfs_offline`](@ref)) instead.
"""
function sfswriter(s; dir, times, nles = nothing, filters = nothing, cells = nothing,
    protectshells = 2, outtype = _ -> Float64)
    sam = lessampler(s; nles, filters, cells, protectshells, outtype)
    col = sfscollector(sam, s, dir)
    times = sort!(Float64[t for t in times])
    inext = Ref(1)
    (state, _) -> begin
        while inext[] ≤ length(times) &&
            state.t ≥ times[inext[]] - 1e-10 * (abs(times[inext[]]) + 1)
            col.record!(state.t, state.uh)
            inext[] += 1
            inext[] > length(times) && col.finish!()
        end
        nothing
    end
end

"""
    sfs_offline(prefixes, s; dir, nles, filters, cells, protectshells = 2,
                outtype = _ -> Float64, phaseseed = nothing)

Offline filter bank: run the [`sfswriter`](@ref) sampling + artifact
writing over stored raw snapshots (`prefixes` as saved by
[`spectral_save`](@ref), in the intended time order) instead of a live
solve — the bank is a regenerable derivative of the raw store, so new
kernels, widths, or coarse grids never require a rerun. Collective over
`s`'s communicator (σ̂ needs the full DNS-grid nonlinearity); `s` may use
a leaner transform grid than the original run as long as `kcut` and `l`
match the files ([`spectral_load!`](@ref)'s rule). `phaseseed ≠ nothing`
applies [`spectral_phaserandomize!`](@ref) to each snapshot before
filtering — the kinematic-null variant of the bank.
"""
function sfs_offline(prefixes, s; dir, nles = nothing, filters = nothing,
    cells = nothing, protectshells = 2, outtype = _ -> Float64, phaseseed = nothing)
    sam = lessampler(s; nles, filters, cells, protectshells, outtype)
    col = sfscollector(sam, s, dir)
    uh = specvelocity(s)
    for prefix in prefixes
        md = spectral_load!(uh, prefix, s)
        phaseseed === nothing || spectral_phaserandomize!(uh, s; seed = phaseseed)
        col.record!(md.time, uh)
    end
    col.finish!()
    nothing
end
