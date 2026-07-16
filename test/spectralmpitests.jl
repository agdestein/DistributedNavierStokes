# Multi-rank tests of the pseudo-spectral solver, launched under mpiexec by
# runtests.jl. Every rank runs the same asserts; a failure on any rank fails
# the whole mpiexec run.

using DistributedNavierStokes
using FFTW
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

# Deterministic global velocity field, evaluable on any rank: a smooth
# function of the physical coordinates (so all rank counts see identical
# initial data).
fx(x, y, z) = sin(x) * cos(y) * cos(z) + 0.3 * cos(2y) * sin(z)
fy(x, y, z) = -cos(x) * sin(y) * cos(z) + 0.2 * sin(2z) * cos(x)
fz(x, y, z) = 0.1 * sin(x + 2z) * cos(2y) + 0.25 * cos(3x)

@testset "spectral pipeline n=$nranks pg=$pg N=$N" for pg in procgrids(nranks),
    N in (12, 18)

    s = spectral_setup(; n = N, comm, procgrid = pg)
    (; kcut, m, lphys, lspec) = s

    # Global reference field and its truncated spectrum via serial FFTW.
    coord(gi) = 2π / N * (gi - 0.5)
    u = [
        f(coord(i), coord(j), coord(k)) for i = 1:N, j = 1:N, k = 1:N,
        f in (fx, fy, fz)
    ]
    F = rfft(u, 1:3) ./ N^3
    fine(j) = j ≤ kcut + 1 ? j : j + (N - m)

    # Forward: local state block matches the reference truncation.
    uh = specvelocity(s)
    spectral_velocityfield!(uh, s; x = fx, y = fy, z = fz, project = false)
    uhh = Array(uh)
    @test maximum(
        abs(uhh[i, j, k, c] - F[lspec.ranges[1][i], fine(j), fine(lspec.ranges[3][k]), c])
        for c = 1:3, k in axes(uhh, 3), j = 1:m, i in axes(uhh, 1)
    ) < 1e-13

    # Backward: physical block matches the serially dealiased field.
    mask = [
        (i - 1 ≤ kcut && min(j - 1, N - j + 1) ≤ kcut && min(k - 1, N - k + 1) ≤ kcut)
        for i = 1:(N÷2+1), j = 1:N, k = 1:N
    ]
    udeal = irfft(rfft(u, 1:3) .* mask, N, 1:3)
    v = similar(s.fft.v)
    spec_to_phys!(v, uh, s)
    vh = Array(v)
    @test maximum(
        abs(vh[i, j, k, c] - udeal[lphys.ranges[1][i], lphys.ranges[2][j], lphys.ranges[3][k], c])
        for c = 1:3, k in axes(vh, 3), j in axes(vh, 2), i in axes(vh, 1)
    ) < 1e-13

    # Projection kills the divergence.
    spectral_project!(uh, s)
    @test spectral_maxdiv(uh, s) < 1e-14
end

@testset "spectral TGV2D exact decay n=$nranks pg=$pg" for pg in procgrids(nranks)
    visc = 0.01
    s = spectral_setup(; n = 12, visc, comm, procgrid = pg)
    uh = specvelocity(s)
    spectral_velocityfield!(
        uh, s;
        x = (x, y, z) -> sin(x) * cos(y),
        y = (x, y, z) -> -cos(x) * sin(y),
        z = Returns(0.0),
    )
    spectral_solve!(; uh, setup = s, tlims = (0.0, 1.0), Δt = 0.05)
    spec_to_phys!(s.fft.v, uh, s)
    d = exp(-2 * visc * 1.0)
    coord(gi) = 2π / 12 * (gi - 0.5)
    vh = Array(s.fft.v)
    err = maximum(
        abs(
            vh[i, j, k, c] -
            d * (c == 1 ? 1 : -1) *
            (c == 1 ? sin(coord(s.lphys.ranges[1][i])) * cos(coord(s.lphys.ranges[2][j])) :
             c == 2 ? cos(coord(s.lphys.ranges[1][i])) * sin(coord(s.lphys.ranges[2][j])) : 0.0),
        ) for c = 1:3, k in axes(vh, 3), j in axes(vh, 2), i in axes(vh, 1)
    )
    @test err < 1e-12
end

@testset "spectral decomposition invariance n=$nranks pg=$pg" for pg in procgrids(nranks)
    function runcase(comm, procgrid)
        s = spectral_setup(; n = 18, visc = 0.005, comm, procgrid)
        uh = specvelocity(s)
        spectral_velocityfield!(uh, s; x = fx, y = fy, z = fz)
        spectral_solve!(; uh, setup = s, tlims = (0.0, 0.1), Δt = 0.01)
        spec_to_phys!(s.fft.v, uh, s)
        s, uh, Array(s.fft.v)
    end
    spar, uhpar, vpar = runcase(comm, pg)
    sser, uhser, vser = runcase(MPI.COMM_SELF, (1, 1))
    ref = vser[spar.lphys.ranges[1], spar.lphys.ranges[2], spar.lphys.ranges[3], :]
    @test maximum(abs, vpar .- ref) < 1e-12
    @test spectral_maxdiv(uhpar, spar) < 1e-13
    # energy and dissipation are decomposition-invariant
    @test spectral_energy(uhpar, spar) ≈ spectral_energy(uhser, sser)
    @test spectral_dissipation(uhpar, spar) ≈ spectral_dissipation(uhser, sser)
end

@testset "spectral inviscid conservation n=$nranks" begin
    s = spectral_setup(; n = 18, visc = 0.0, comm)
    uh = specvelocity(s)
    taylorgreen!(uh, s)
    e0 = spectral_energy(uh, s)
    spectral_solve!(; uh, setup = s, tlims = (0.0, 0.1), Δt = 0.002)
    @test abs(spectral_energy(uh, s) - e0) / e0 < 1e-10
    @test spectral_maxdiv(uh, s) < 1e-13
end

@testset "spectral host-staged buffers n=$nranks" begin
    s = spectral_setup(; n = 12, comm, mpibuf = :host)
    uh = specvelocity(s)
    spectral_velocityfield!(uh, s; x = fx, y = fy, z = fz)
    sref = spectral_setup(; n = 12, comm, mpibuf = :auto)
    uhref = specvelocity(sref)
    spectral_velocityfield!(uhref, sref; x = fx, y = fy, z = fz)
    @test Array(uh) == Array(uhref)
end
