# In-situ statistics: online plane/time-averaged channel profiles and
# shell-binned energy spectra. Accumulators are plain NamedTuples holding
# host vectors and a `sample` closure to pass as a processor; a finalizer
# reduces over ranks and returns global results (on every rank).

"""
    cs = channelstats(setup; nupdate = 1)

Accumulator for channel-flow statistics: plane- (x, z) and time-averaged
first and second moments. Pass `cs.sample` as a processor to `solve!`, then
call [`channelprofiles`](@ref). Cell-centered quantities (`U`, `W`, `uu`,
`ww`) live at `y`; `V`, `vv`, and the shear stress `uv` live at y-faces.
Holds one scratch field for the interpolated products.
"""
function channelstats(setup; nupdate = 1)
    (; T, ranges) = setup
    nc = length(ranges.p[2])   # local cell rows
    nf = length(ranges.y[2])   # local active face rows
    acc = (;
        nsamples = Ref(0),
        Uc = zeros(T, nc),
        Wc = zeros(T, nc),
        uuc = zeros(T, nc),
        wwc = zeros(T, nc),
        Vf = zeros(T, nf),
        vvf = zeros(T, nf),
        Uf = zeros(T, nf),
        uvf = zeros(T, nf),
        scratch = scalarfield(setup),
    )
    sample =
        (state, s) -> begin
            state.n % nupdate == 0 && sample_channelstats!(acc, state.u, s)
            nothing
        end
    (; acc..., sample)
end

function sample_channelstats!(acc, u, s)
    rp = s.ranges.p
    ry = s.ranges.y
    plane(A, rng) = vec(Array(sum(view(A, rng...); dims = (1, 3))))
    plane2(A, rng) = vec(Array(sum(abs2, view(A, rng...); dims = (1, 3))))
    acc.Uc .+= plane(u.x, rp)   # ranges.x ≡ ranges.p (periodic x)
    acc.Wc .+= plane(u.z, rp)
    acc.uuc .+= plane2(u.x, rp)
    acc.wwc .+= plane2(u.z, rp)
    acc.Vf .+= plane(u.y, ry)
    acc.vvf .+= plane2(u.y, ry)
    # u interpolated to the y-face points (xf, yf, zc), then the product uv
    r1, r2, r3 = ry
    @views acc.scratch[r1, r2, r3] .= (u.x[r1, r2, r3] .+ u.x[r1, r2 .+ 1, r3]) ./ 2
    acc.Uf .+= plane(acc.scratch, ry)
    @views acc.scratch[r1, r2, r3] .*= (u.y[r1, r2, r3] .+ u.y[r1 .+ 1, r2, r3]) ./ 2
    acc.uvf .+= plane(acc.scratch, ry)
    acc.nsamples[] += 1
    nothing
end

"""
Global channel profiles from a [`channelstats`](@ref) accumulator, on every
rank: `(; y, yf, U, V, W, uu, vv, ww, uv, nsamples)` with the second
moments central (fluctuations): `uv = ⟨uv⟩ - ⟨u⟩⟨v⟩` etc.
"""
function channelprofiles(cs, setup)
    (; topo, T, bc, n) = setup
    nsamples = cs.nsamples[]
    norm = max(nsamples, 1) * n[1] * n[3]
    # total plane sums: reduce over the z-chunks (processor-grid axis 1)
    tot(v) = MPI.Allreduce(v, +, topo.subcomms[1]) ./ norm
    Uc, Wc, uuc, wwc = tot(cs.Uc), tot(cs.Wc), tot(cs.uuc), tot(cs.wwc)
    Vf, vvf, Uf, uvf = tot(cs.Vf), tot(cs.vvf), tot(cs.Uf), tot(cs.uvf)
    # assemble global y profiles (processor-grid axis 2)
    p2 = topo.procgrid[2]
    ccounts = [length(blockrange(n[2], p2, c)) for c = 0:(p2-1)]
    fcounts = map(0:(p2-1)) do c
        b = blockrange(n[2], p2, c)
        length(first(b):(bc[2]==:wall ? min(last(b), n[2]-1) : last(b)))
    end
    gather(v, counts) = (
        out = zeros(T, sum(counts));
        MPI.Allgatherv!(v, MPI.VBuffer(out, counts), topo.subcomms[2]);
        out
    )
    yfg = setup.grid.yfg
    (;
        y = setup.grid.ycg,
        yf = bc[2] == :wall ? yfg[2:n[2]] : yfg[2:(n[2]+1)],
        U = gather(Uc, ccounts),
        V = gather(Vf, fcounts),
        W = gather(Wc, ccounts),
        uu = gather(uuc .- Uc .^ 2, ccounts),
        vv = gather(vvf .- Vf .^ 2, fcounts),
        ww = gather(wwc .- Wc .^ 2, ccounts),
        uv = gather(uvf .- Uf .* Vf, fcounts),
        nsamples,
    )
end

"""
    sp = spectrumstats(setup; nupdate = 1)

Accumulator for the shell-binned kinetic energy spectrum of a fully
periodic box (integer-wavenumber shells; assumes equal domain extents).
Pass `sp.sample` as a processor, then call [`energyspectrum`](@ref).
Reuses the Poisson pipeline's FFT plans and buffers.
"""
function spectrumstats(setup; nupdate = 1)
    setup.bc == (:periodic, :periodic, :periodic) ||
        error("spectra require a fully periodic box")
    nbins = ceil(Int, sqrt(sum(abs2, setup.n .÷ 2))) + 1
    acc = (; nsamples = Ref(0), E = zeros(setup.T, nbins))
    sample = (state, s) -> begin
        state.n % nupdate == 0 && sample_spectrum!(acc, state.u, s)
        nothing
    end
    (; acc..., sample)
end

function sample_spectrum!(acc, u, s)
    (; backend, lay, w) = s
    ps = s.poisson
    src = map(l -> (w+1):(w+l), lay.ldims)
    dst = map(l -> 1:l, lay.ldims)
    for c in (:x, :y, :z)
        copybox!(ps.r, dst, u[c], src, backend)   # strip ghosts
        mul!(ps.âx, ps.plan_x, ps.r)
        transpose!(ps.âz, ps.âx, ps.xz, backend)
        ps.plan_z! * ps.âz
        transpose!(ps.ây, ps.âz, ps.zy, backend)
        ps.plan_y! * ps.ây
        KernelAbstractions.synchronize(backend)
        binshells!(acc.E, Array(ps.ây), ps.lyc, s.n)
    end
    acc.nsamples[] += 1
    nothing
end

"""
Add ½|û|²/N² to the integer-wavenumber shells. The rfft in x stores only
`kx = 0..nx÷2`; the missing conjugate modes `0 < kx < nx/2` count twice.
"""
function binshells!(E, Â, lyc, n)
    N2 = float(prod(n))^2
    for (k, gk) in enumerate(lyc.ranges[3]),
        (j, gj) in enumerate(lyc.ranges[2]),
        (i, gi) in enumerate(lyc.ranges[1])

        kx = gi - 1
        ky = gj - 1 > n[2] ÷ 2 ? gj - 1 - n[2] : gj - 1
        kz = gk - 1 > n[3] ÷ 2 ? gk - 1 - n[3] : gk - 1
        κ = round(Int, sqrt(kx^2 + ky^2 + kz^2))
        wgt = 0 < kx < n[1] / 2 ? 2 : 1
        E[κ+1] += wgt * abs2(Â[i, j, k]) / 2 / N2
    end
    E
end

"""
Time-averaged energy spectrum from a [`spectrumstats`](@ref) accumulator,
reduced over all ranks: `(; κ, E)` with `κ` the integer shell wavenumbers.
`sum(E)` equals the volume-averaged kinetic energy `½⟨|u|²⟩` (Parseval).
"""
function energyspectrum(sp, setup)
    E = MPI.Allreduce(sp.E, +, setup.topo.cart) ./ max(sp.nsamples[], 1)
    (; κ = 0:(length(E)-1), E)
end
