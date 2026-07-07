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
procgrids(n) = [(a, n ÷ a) for a in 1:n if n % a == 0]

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
    for a in 1:2
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
