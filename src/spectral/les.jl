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
#   ported verbatim — Gaussian test filter with forced-shell protection,
#   2/3 truncation (which also zeroes the mean, as in SymmetryCode), LES-grid
#   nonlinearity, τ = filter(σ) − ū⊗ū made trace-free — plus `filter_reynolds`
#   and the shell spectrum, then the JLD2 writers (`fields.jld2`,
#   `les_meta.jld2`, `dns_meta.jld2`).
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

"""
Gaussian test filter `exp(-Δ²k²/24)` on a coarse rfft array, leaving the
forced shells (`|k_int| ≤ protectshells`) untouched, as in SymmetryCode's
`gaussianfilter!` (which hardcodes `nshell = 2`).
"""
function les_gaussian!(a, n, l, Δ, protectshells)
    kf = 2π / l
    kbound2 = (kf * (protectshells + 1))^2
    for k in axes(a, 3), j in axes(a, 2), i in axes(a, 1)
        kk = kf^2 * ((i - 1)^2 + lesfreq(j, n)^2 + lesfreq(k, n)^2)
        w = ifelse(kk < kbound2, 1.0, exp(-Δ^2 * kk / 24))
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
Gather plan for the LES cube + rank-0 coarse-grid workspace. `filters` are
the filter-width factors `Δf` (`Δ = Δf·l/nles`).
"""
function lessampler(s; nles, filters, protectshells = 2)
    (; kcut, m, T, topo, lspec) = s
    nles % 2 == 0 || error("nles must be even")
    nles ÷ 2 ≤ kcut || error("LES cube needs nles/2 ≤ kcut (nles = $nles, kcut = $kcut)")
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
    coarse() = ntuple(_ -> Array{Complex{T},3}(undef, nles ÷ 2 + 1, nles, nles), 3)
    phys() = ntuple(_ -> Array{T,3}(undef, nles, nles, nles), 3)
    (;
        nles, filters, protectshells, myboxes, allboxes, counts,
        sendbuf,
        recvbuf = root ? Vector{Complex{T}}(undef, sum(counts)) : nothing,
        ub = root ? coarse() : nothing,        # gathered û (unfiltered)
        σb = root ? (coarse()..., coarse()...) : nothing,  # gathered σ̂ (6)
        ubf = root ? coarse() : nothing,       # per-filter ū workspace
        σ2 = root ? (coarse()..., coarse()...) : nothing,  # LES nonlinearity
        τ = root ? (coarse()..., coarse()...) : nothing,
        vs = root ? phys() : nothing,
        vv = root ? Array{T,3}(undef, nles, nles, nles) : nothing,
        tmp = root ? Array{Complex{T},3}(undef, nles ÷ 2 + 1, nles, nles) : nothing,
        plan = root ? plan_rfft(Array{T,3}(undef, nles, nles, nles)) : nothing,
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
then the rank-0 `sfs!` port per filter. Returns, on rank 0, a vector over
`sam.filters` of `(; ubar, τ, redelta, spectrum)` (freshly allocated copies);
`nothing` on other ranks. Collective; clobbers the RHS pipeline buffers.
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
    (; nles, protectshells) = sam
    map(sam.filters) do Δf
        Δ = Float64(Δf * l / nles)
        # ū = gaussian(cutoff(û)), dealiased; σbar1 = gaussian(cutoff(σ̂))
        for c = 1:3
            copyto!(sam.ubf[c], sam.ub[c])
            les_gaussian!(sam.ubf[c], nles, l, Δ, protectshells)
            les_twothirds!(sam.ubf[c], nles)
        end
        for c = 1:6
            copyto!(sam.τ[c], sam.σb[c])   # τ starts as σbar1
            les_gaussian!(sam.τ[c], nles, l, Δ, protectshells)
        end
        # τ = σbar1 - F[ū_i ū_j], dealiased, trace-free
        les_nonlinearity!(sam.σ2, sam.vs, sam.vv, sam.tmp, sam.ubf, sam.plan, nles)
        for c = 1:6
            sam.τ[c] .-= sam.σ2[c]
            les_twothirds!(sam.τ[c], nles)
        end
        les_tracefree!(sam.τ)
        (;
            ubar = (; x = copy(sam.ubf[1]), y = copy(sam.ubf[2]), z = copy(sam.ubf[3])),
            τ = (;
                xx = copy(sam.τ[1]), yy = copy(sam.τ[2]), zz = copy(sam.τ[3]),
                xy = copy(sam.τ[4]), yz = copy(sam.τ[5]), zx = copy(sam.τ[6]),
            ),
            redelta = les_filter_reynolds(sam.ubf, nles, l, Float64(visc), Δ),
            spectrum = les_spectrum(sam.ubf, nles),
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
    sfswriter(s; dir, nles, filters, times, protectshells = 2)

Processor for [`spectral_solve!`](@ref) that samples filtered velocities and
sub-filter stresses at each entry of `times` (pass the same `times` as
`tstops` so steps land on them exactly) and, after the last sample, writes
SymmetryCode's artifact schemas under `dir` (rank 0, atomic):

- `dns_meta.jld2`: `times`, `spectra_dns`, `statistics_dns`, `t_int`,
  `walltime` (Δ-independent DNS metadata).
- `delta=<Δf>/fields.jld2` per filter factor: `inputs` (ū, spectral,
  `(; x, y, z)` per snapshot), `outputs` (τ, trace-free,
  `(; xx, yy, zz, xy, yz, zx)`), `redelta`, `Δ`, `Δ_factor`, `visc`.
- `delta=<Δf>/les_meta.jld2`: `spectra_les`, `redelta_mean`.

`nles` is the LES transform grid (needs `nles/2 ≤ kcut`); `Δ = Δf·l/nles`.
The state must be mean-free (see the module header note).
"""
function sfswriter(s; dir, nles, filters, times, protectshells = 2)
    sam = lessampler(s; nles, filters, protectshells)
    times = sort!(Float64[t for t in times])
    inext = Ref(1)
    tstart = time()
    tsamples = Float64[]
    spectra_dns = Vector{Float64}[]
    statistics_dns = []
    inputs = [NamedTuple[] for _ in filters]
    outputs = [NamedTuple[] for _ in filters]
    redelta = [Float64[] for _ in filters]
    spectra_les = [Vector{Float64}[] for _ in filters]
    (state, s2) -> begin
        while inext[] ≤ length(times) &&
            state.t ≥ times[inext[]] - 1e-10 * (abs(times[inext[]]) + 1)
            st = spectral_stats(state.uh, s2)
            sp = spectral_spectrum(state.uh, s2)
            samples = sfs_sample!(sam, state.uh, s2)
            if s2.topo.rank == 0
                push!(tsamples, state.t)
                push!(spectra_dns, Float64.(sp.E[2:end]))   # shells 1:kcut
                push!(statistics_dns, st)
                for (k, smp) in enumerate(samples)
                    push!(inputs[k], smp.ubar)
                    push!(outputs[k], smp.τ)
                    push!(redelta[k], smp.redelta)
                    push!(spectra_les[k], smp.spectrum)
                end
            end
            inext[] += 1
            if inext[] > length(times) && s2.topo.rank == 0
                jldsave_atomic(
                    joinpath(mkpath(dir), "dns_meta.jld2");
                    times = tsamples, spectra_dns,
                    statistics_dns = [statistics_dns...],
                    t_int = statistics_dns[1].t_int, walltime = time() - tstart,
                )
                for (k, Δf) in enumerate(filters)
                    ddir = mkpath(joinpath(dir, "delta=$(Δf)"))
                    jldsave_atomic(
                        joinpath(ddir, "fields.jld2");
                        inputs = [inputs[k]...], outputs = [outputs[k]...],
                        redelta = redelta[k],
                        Δ = Float64(Δf * s2.l / nles), Δ_factor = Δf,
                        visc = Float64(s2.visc),
                    )
                    jldsave_atomic(
                        joinpath(ddir, "les_meta.jld2");
                        spectra_les = spectra_les[k],
                        redelta_mean = sum(redelta[k]) / length(redelta[k]),
                    )
                end
            end
        end
        nothing
    end
end
