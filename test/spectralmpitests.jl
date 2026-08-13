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

@testset "spectral random IC + forcing n=$nranks pg=$pg" for pg in procgrids(nranks)
    function runforced(comm, procgrid)
        s = spectral_setup(; n = 18, visc = 5e-3, comm, procgrid)
        uh = specvelocity(s)
        spectral_randomfield!(uh, s; totalenergy = 0.4, kpeak = 2, seed = 7)
        f = shellforcing(uh, s; shells = 1:2)
        spectral_solve!(; uh, setup = s, tlims = (0.0, 0.1), Δt = 0.01, forcing = f)
        spec_to_phys!(s.fft.v, uh, s)
        s, uh, Array(s.fft.v)
    end
    spar, uhpar, vpar = runforced(comm, pg)
    sser, uhser, vser = runforced(MPI.COMM_SELF, (1, 1))
    # The counter-based IC makes the whole forced trajectory
    # decomposition-invariant, not just the operators.
    ref = vser[spar.lphys.ranges[1], spar.lphys.ranges[2], spar.lphys.ranges[3], :]
    @test maximum(abs, vpar .- ref) < 1e-12
    # IC diagnostics
    s0 = spectral_setup(; n = 18, visc = 5e-3, comm, procgrid = pg)
    uh0 = specvelocity(s0)
    spectral_randomfield!(uh0, s0; totalenergy = 0.4, kpeak = 2, seed = 7)
    @test spectral_energy(uh0, s0) ≈ 0.4
    @test spectral_maxdiv(uh0, s0) < 1e-14
    sp = spectral_spectrum(uh0, s0)
    kdiag = floor(Int, sqrt(3) * s0.kcut)
    prof = [k^4 * exp(-2 * (k / 2)^2) for k = 0:kdiag]
    target = 0.4 .* prof ./ sum(prof)
    @test maximum(abs, sp.E .- target[1:(s0.kcut+1)]) < 1e-13
    # forced shells hold their reference energy through the run
    @test shell_energies(uhpar, spar, 1:2) ≈ shell_energies(uh0, s0, 1:2)
end

@testset "spectral snapshot I/O n=$nranks pg=$pg" for pg in procgrids(nranks)
    rank = MPI.Comm_rank(comm)
    dir = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm)
    s = spectral_setup(; n = 18, comm, procgrid = pg)
    uh = specvelocity(s)
    spectral_velocityfield!(uh, s; x = fx, y = fy, z = fz)
    eref = shell_energies(uh, s, 1:2)
    spectral_save(joinpath(dir, "field"), uh, s; time = 1.5, step = 7, meta = (; eref))

    # Byte-exact round trip on the same decomposition, metadata included.
    uh2 = specvelocity(s)
    md = spectral_load!(uh2, joinpath(dir, "field"), s)
    @test Array(uh2) == Array(uh)
    @test md.time == 1.5 && md.step == 7
    @test md.meta["eref"] == eref

    # The file is written in global index order: a serial (1-rank) reader on
    # every rank reconstructs the full field, matching a serially generated
    # one to pipeline roundoff.
    sser = spectral_setup(; n = 18, comm = MPI.COMM_SELF, procgrid = (1, 1))
    uhser = specvelocity(sser)
    spectral_velocityfield!(uhser, sser; x = fx, y = fy, z = fz)
    uhread = specvelocity(sser)
    spectral_load!(uhread, joinpath(dir, "field"), sser)
    @test maximum(abs, Array(uhread) .- Array(uhser)) < 1e-13

    MPI.Barrier(comm)
    rank == 0 && rm(dir; recursive = true)
end

@testset "spectral snapshot restart n=$nranks pg=$pg" for pg in procgrids(nranks)
    rank = MPI.Comm_rank(comm)
    dir = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm)
    tsave = [0.05, 0.1]

    # Uninterrupted forced run, snapshots at 0.05 and 0.1 (tstops makes the
    # steps land on them exactly).
    s = spectral_setup(; n = 18, visc = 5e-3, comm, procgrid = pg)
    uh = specvelocity(s)
    spectral_randomfield!(uh, s; totalenergy = 0.4, kpeak = 2, seed = 7)
    eref = shell_energies(uh, s, 1:2)
    spectral_solve!(;
        uh, setup = s, tlims = (0.0, 0.1), Δt = 0.008,
        forcing = shellforcing(uh, s; shells = 1:2),
        tstops = tsave,
        processors = (; snap = snapshotsaver(joinpath(dir, "full"); times = tsave, meta = (; eref))),
    )

    # Restart from the mid snapshot (forcing reference from the sidecar)
    # and replay to 0.1: the trajectory continues as if uninterrupted.
    s2 = spectral_setup(; n = 18, visc = 5e-3, comm, procgrid = pg)
    uh2 = specvelocity(s2)
    md = spectral_load!(uh2, joinpath(dir, "full_0001"), s2)
    @test md.time == 0.05
    spectral_solve!(;
        uh = uh2, setup = s2, tlims = (md.time, 0.1), Δt = 0.008,
        forcing = shellforcing(uh2, s2; shells = 1:2, eref = md.meta["eref"]),
        processors = (; snap = snapshotsaver(joinpath(dir, "restart"); times = [0.1])),
    )
    a = reinterpret(ComplexF64, read(joinpath(dir, "full_0002.bin")))
    b = reinterpret(ComplexF64, read(joinpath(dir, "restart_0001.bin")))
    @test maximum(abs, a .- b) < 1e-12

    MPI.Barrier(comm)
    rank == 0 && rm(dir; recursive = true)
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
