# Single-process physics tests (COMM_SELF): discrete identities and
# accuracy. Included by runtests.jl.

tanhstretch(β) = t -> (tanh(β * (2t - 1)) / tanh(β) + 1) / 2

channelsetup(; n = (16, 12, 8), kwargs...) = setup(;
    n,
    lims = ((0.0, 2π), (-1.0, 1.0), (0.0, π)),
    bc = (:periodic, :wall, :periodic),
    stretch = tanhstretch(1.5),
    comm = MPI.COMM_SELF,
    kwargs...,
)

"Random values on all active DOFs, halos exchanged."
function randomfield!(u, s; seed = 42)
    Random.seed!(seed)
    for c in (:x, :y, :z)
        u[c][s.ranges[c]...] .= randn(s.T, length.(s.ranges[c])...)
    end
    exchange_halo!(u, s)
    u
end

"Volume-weighted inner products (⟨u, q⟩, ⟨u, u⟩) over active DOFs."
function weighted_dot(u, q, s)
    Δy = Array(s.grid.Δy)
    Δyu = Array(s.grid.Δyu)
    d = zero(s.T)
    n2 = zero(s.T)
    for c in (:x, :y, :z)
        r = s.ranges[c]
        Δ2 = c == :y ? Δyu : Δy
        for k in r[3], j in r[2], i in r[1]
            vol = s.grid.Δx * Δ2[j] * s.grid.Δz
            d += vol * u[c][i, j, k] * q[c][i, j, k]
            n2 += vol * u[c][i, j, k]^2
        end
    end
    d, n2
end

@testset "projection and skew-symmetry ($case)" for (case, s) in (
    ("periodic", setup(; n = (16, 12, 8), visc = 0.0, comm = MPI.COMM_SELF)),
    ("channel", channelsetup(; visc = 0.0)),
)
    u = randomfield!(vectorfield(s), s)
    p = scalarfield(s)
    project!(u, p, s)
    @test maxdiv(u, s) < 1e-12

    # Pure convection (visc = 0, no force): ⟨u, C(u)⟩ = 0 for div-free u,
    # also on the stretched wall-bounded grid.
    q = vectorfield(s)
    DNS.momentum!(q, u, 0.0, 1.0, s)
    KernelAbstractions.synchronize(s.backend)
    d, n2 = weighted_dot(u, q, s)
    @test abs(d) < 1e-12 * n2
end

@testset "inviscid energy conservation" begin
    s = setup(; n = (16, 16, 16), visc = 0.0, comm = MPI.COMM_SELF)
    u = randomfield!(vectorfield(s), s)
    p = scalarfield(s)
    project!(u, p, s)
    e0 = DNS.kinetic_energy(u, s)
    solve!(; u, setup = s, tlims = (0.0, 0.1), Δt = 0.002)
    e1 = DNS.kinetic_energy(u, s)
    # Space is energy-conserving; the drift is the RK3 O(Δt³)-per-step error.
    @test abs(e1 - e0) / e0 < 1e-7
    @test maxdiv(u, s) < 1e-12
end

@testset "Taylor-Green accuracy" begin
    ν = 0.01
    tend = 0.5
    err = map((16, 32)) do n
        s = setup(;
            n = (n, n, 4),
            lims = ((0.0, 2π), (0.0, 2π), (0.0, 2π)),
            visc = ν,
            comm = MPI.COMM_SELF,
        )
        u = vectorfield(s)
        velocityfield!(
            u,
            s;
            x = (x, y, z) -> sin(x) * cos(y),
            y = (x, y, z) -> -cos(x) * sin(y),
        )
        solve!(; u, setup = s, tlims = (0.0, tend), Δt = 1e-3)
        decay = exp(-2 * ν * tend)
        r = s.ranges.x
        maximum(
            abs(u.x[i, j, k] - sin(s.grid.xf[i-s.w]) * cos(s.grid.yc[j-s.w]) * decay)
            for k in r[3], j in r[2], i in r[1]
        )
    end
    @test err[2] < 1e-3
    @test 3 < err[1] / err[2] < 5   # second order in space
end

@testset "energy spectrum" begin
    s = setup(; n = (16, 16, 16), visc = 0.0, comm = MPI.COMM_SELF)
    u = randomfield!(vectorfield(s), s)
    p = scalarfield(s)
    project!(u, p, s)
    sp = spectrumstats(s)
    sp.sample((; u, p, t = 0.0, n = 0), s)
    (; E) = energyspectrum(sp, s)
    V = prod(l -> l[2] - l[1], s.lims)
    @test sum(E) ≈ DNS.kinetic_energy(u, s) / V rtol = 1e-12   # Parseval

    utg = vectorfield(s)
    velocityfield!(
        utg,
        s;
        x = (x, y, z) -> sin(x) * cos(y),
        y = (x, y, z) -> -cos(x) * sin(y),
    )
    sptg = spectrumstats(s)
    sptg.sample((; u = utg, p, t = 0.0, n = 0), s)
    Etg = energyspectrum(sptg, s).E
    @test Etg[2] / sum(Etg) ≈ 1 rtol = 1e-12   # Taylor-Green: all energy in shell κ = 1
end

@testset "channel statistics (exact laminar)" begin
    s = channelsetup()
    u = vectorfield(s)
    velocityfield!(u, s; x = (x, y, z) -> 1 - y^2)
    cs = channelstats(s)
    cs.sample((; u, p = nothing, t = 0.0, n = 0), s)
    prof = channelprofiles(cs, s)
    @test prof.U ≈ 1 .- prof.y .^ 2 atol = 1e-13
    @test maximum(abs, prof.uu) < 1e-13
    @test maximum(abs, prof.uv) < 1e-13
    @test length(prof.U) == s.n[2]
    @test length(prof.uv) == s.n[2] - 1
end

@testset "channel flow sanity" begin
    s = channelsetup(; visc = 0.05, bodyforce = (1.0, 0.0, 0.0))
    u = vectorfield(s)
    st = solve!(; u, setup = s, tlims = (0.0, 1.0))
    @test st.t ≈ 1.0
    @test maxdiv(u, s) < 1e-12
    @test 0 < DNS.kinetic_energy(u, s) < 100
    # x-velocity should be symmetric about the centerline and positive
    ux = Array(u.x)
    r = s.ranges.x
    profile = [sum(ux[i, j, k] for k in r[3], i in r[1]) for j in r[2]]
    @test all(>(0), profile)
    @test profile ≈ reverse(profile) rtol = 1e-8
end
