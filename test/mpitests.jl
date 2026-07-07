# Multi-rank tests, launched under mpiexec by runtests.jl. Every rank runs
# the same asserts; a failure on any rank fails the whole mpiexec run.

using DistributedNavierStokes
using KernelAbstractions
using MPI
using Random
using Test

const DNS = DistributedNavierStokes

MPI.Init()
comm = MPI.COMM_WORLD
nranks = MPI.Comm_size(comm)
backend = CPU()

"All 2D processor grids for `n` ranks."
procgrids(n) = [(a, n ÷ a) for a = 1:n if n % a == 0]

tanhstretch(β) = t -> (tanh(β * (2t - 1)) / tanh(β) + 1) / 2

casesetup(bcy; kwargs...) = setup(;
    n = (12, 10, 8),
    lims = ((0.0, 2π), bcy == :wall ? (-1.0, 1.0) : (0.0, 2π), (0.0, π)),
    bc = (:periodic, bcy, :periodic),
    stretch = bcy == :wall ? tanhstretch(1.3) : identity,
    kwargs...,
)

# ---------------------------------------------------------------- layouts

# Deterministic global field: every rank can evaluate any global entry.
gval(i, j, k) = 1.0i + 100.0j + 10000.0k

function fillglobal!(A, l)
    for k in axes(A, 3), j in axes(A, 2), i in axes(A, 1)
        A[i, j, k] = gval(l.ranges[1][i], l.ranges[2][j], l.ranges[3][k])
    end
    A
end

matchesglobal(A, l) = all(
    A[i, j, k] == gval(l.ranges[1][i], l.ranges[2][j], l.ranges[3][k]) for
    k in axes(A, 3), j in axes(A, 2), i in axes(A, 1)
)

@testset "topology n=$nranks pg=$pg" for pg in procgrids(nranks)
    topo = DNS.topology(comm, pg, (true, false))
    # Subcommunicator rank must equal the grid coordinate along that axis
    # (transpose buffer segments rely on this ordering).
    for a = 1:2
        @test MPI.Comm_rank(topo.subcomms[a]) == topo.coords[a]
    end
end

@testset "transpose n=$nranks pg=$pg gdims=$gdims" for pg in procgrids(nranks),
    gdims in [(8, 8, 8), (6, 5, 7), (13, 4, 9)]

    topo = DNS.topology(comm, pg, (true, false))
    lays = map(ax -> DNS.layout(gdims, ax, topo.procgrid, topo.coords), DNS.AXES)
    xz = DNS.plan_transpose(lays.x, lays.z, topo, backend, Float64)
    zy = DNS.plan_transpose(lays.z, lays.y, topo, backend, Float64)

    # Forward: data lands where the destination layout says it should.
    ax = fillglobal!(zeros(lays.x.ldims), lays.x)
    az = zeros(lays.z.ldims)
    ay = zeros(lays.y.ldims)
    DNS.transpose!(az, ax, xz, backend)
    @test matchesglobal(az, lays.z)
    DNS.transpose!(ay, az, zy, backend)
    @test matchesglobal(ay, lays.y)

    # Round trip with random data is exact.
    Random.seed!(topo.rank)
    r = rand(lays.x.ldims...)
    DNS.transpose!(az, r, xz, backend)
    DNS.transpose!(ay, az, zy, backend)
    DNS.transpose!(az, ay, DNS.reverse_plan(zy), backend)
    b = zeros(lays.x.ldims)
    DNS.transpose!(b, az, DNS.reverse_plan(xz), backend)
    @test b == r
end

# ------------------------------------------------------------------ halos

"""
Expected field value at *global* indices (possibly outside the domain):
periodic wrap in x/z (and y if periodic), wall reflection rules in y
matching `fill_wall!` (normal: zero wall faces, antisymmetric; tangential:
antisymmetric about the wall plane; pressure: symmetric).
"""
function expectedvalue(stag, g, gi, gj, gk, n, bc)
    ny = n[2]
    if bc[2] == :wall
        if stag == :y
            (gj == 0 || gj == ny) && return 0.0
            gj < 0 && return -expectedvalue(stag, g, gi, -gj, gk, n, bc)
            gj > ny && return -expectedvalue(stag, g, gi, 2ny - gj, gk, n, bc)
        else
            sgn = stag == :p ? 1.0 : -1.0
            gj < 1 && return sgn * expectedvalue(stag, g, gi, 1 - gj, gk, n, bc)
            gj > ny && return sgn * expectedvalue(stag, g, gi, 2ny + 1 - gj, gk, n, bc)
        end
    end
    g(mod1(gi, n[1]), mod1(gj, n[2]), mod1(gk, n[3]))
end

@testset "halo n=$nranks pg=$pg bc=$bcy stag=$stag" for pg in procgrids(nranks),
    bcy in (:periodic, :wall),
    stag in (:x, :y, :z, :p)

    s = casesetup(bcy; procgrid = pg)
    g(i, j, k) = sin(0.7i) + cos(0.3j) * sin(0.2k)
    φ = scalarfield(s)
    rng = stag == :p ? s.ranges.p : s.ranges[stag]
    for k in rng[3], j in rng[2], i in rng[1]
        φ[i, j, k] =
            g(s.lay.ranges[1][i-s.w], s.lay.ranges[2][j-s.w], s.lay.ranges[3][k-s.w])
    end
    exchange_halo!(φ, stag, s)
    ok = true
    for k in axes(φ, 3), j in axes(φ, 2), i in axes(φ, 1)
        gi = first(s.lay.ranges[1]) - 1 - s.w + i
        gj = first(s.lay.ranges[2]) - 1 - s.w + j
        gk = first(s.lay.ranges[3]) - 1 - s.w + k
        ok &= φ[i, j, k] == expectedvalue(stag, g, gi, gj, gk, s.n, s.bc)
    end
    @test ok
end

# ---------------------------------------------------------------- poisson

@testset "poisson n=$nranks pg=$pg bc=$bcy" for pg in procgrids(nranks),
    bcy in (:periodic, :wall)

    s = casesetup(bcy; procgrid = pg)
    pex = scalarfield(s)
    for k in s.ranges.p[3], j in s.ranges.p[2], i in s.ranges.p[1]
        pex[i, j, k] =
            sin(0.5 * s.lay.ranges[1][i-s.w]) +
            cos(0.3 * s.lay.ranges[2][j-s.w]) * sin(0.2 * s.lay.ranges[3][k-s.w])
    end
    exchange_halo!(pex, :p, s)
    u = vectorfield(s)
    DNS.pressuregrad_update!(u, pex, s)   # u = -∇pex (zero on wall faces)
    exchange_halo!(u, s)
    DNS.divergence!(s.poisson.r, u, s)    # r = -L pex
    p2 = scalarfield(s)
    DNS.poisson!(p2, s)                   # p2 = -pex + constant
    δ = Array(DNS.interior(p2, s.w)) .+ Array(DNS.interior(pex, s.w))
    c = MPI.Allreduce(sum(δ), +, comm) / prod(s.n)
    dev = MPI.Allreduce(maximum(abs, δ .- c), max, comm)
    @test dev < 1e-8
end

# -------------------------------------------- decomposition invariance

function runcase(bcy, comm, procgrid)
    s = casesetup(bcy; comm, procgrid, visc = 0.02, bodyforce = (0.5, 0.0, 0.0))
    u = vectorfield(s)
    velocityfield!(
        u,
        s;
        x = (x, y, z) -> sin(x) * cos(y) * cos(z) + 0.2 * cos(2z),
        y = (x, y, z) -> (1 - y^2) * sin(2x),
        z = (x, y, z) -> cos(x) * sin(z) + 0.1 * sin(2y),
    )
    st = solve!(; u, setup = s, tlims = (0.0, 0.05), Δt = 0.01)
    u, s, st
end

@testset "decomposition invariance n=$nranks pg=$pg bc=$bcy" for pg in procgrids(nranks),
    bcy in (:periodic, :wall)

    upar, spar, _ = runcase(bcy, comm, pg)
    user, sser, _ = runcase(bcy, MPI.COMM_SELF, (1, 1))   # full grid on every rank
    for c in (:x, :y, :z)
        mine = Array(DNS.interior(upar[c], spar.w))
        ref = view(DNS.interior(user[c], sser.w), spar.lay.ranges...)
        @test maximum(abs, mine .- ref) < 1e-11
    end
    @test maxdiv(upar, spar) < 1e-12
end
