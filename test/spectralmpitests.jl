# Multi-rank tests of the pseudo-spectral solver, launched under mpiexec by
# runtests.jl. Every rank runs the same asserts; a failure on any rank fails
# the whole mpiexec run.

using DistributedNavierStokes
using FFTW
using JLD2: load
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
        # cfl (not Δt): also pins down the adaptive time step, whose CFL
        # bound must reduce per-component maxima globally to be
        # decomposition-invariant (caught on Snellius with 4 H100s vs 1).
        state = spectral_solve!(; uh, setup = s, tlims = (0.0, 0.1), cfl = 0.4, forcing = f)
        spec_to_phys!(s.fft.v, uh, s)
        s, uh, Array(s.fft.v), state.n
    end
    spar, uhpar, vpar, npar = runforced(comm, pg)
    sser, uhser, vser, nser = runforced(MPI.COMM_SELF, (1, 1))
    # The counter-based IC makes the whole forced trajectory
    # decomposition-invariant, not just the operators.
    @test npar == nser
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

@testset "spectral checkpointing n=$nranks pg=$pg" for pg in procgrids(nranks)
    rank = MPI.Comm_rank(comm)
    dir = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm)

    # Uninterrupted forced run with a checkpoint every step (interval 0);
    # only the newest two survive the cleanup.
    runsetup() = spectral_setup(; n = 18, visc = 5e-3, comm, procgrid = pg)
    # NB: the helper's locals must not share names with this testset's
    # variables — assignment in a closure rebinds an enclosing local of the
    # same name (this silently replaced the reference field once).
    function initial(sl)
        ul = specvelocity(sl)
        spectral_randomfield!(ul, sl; totalenergy = 0.4, kpeak = 2, seed = 7)
        ul, shell_energies(ul, sl, 1:2)
    end
    s = runsetup()
    uh, eref = initial(s)
    spectral_solve!(;
        uh, setup = s, tlims = (0.0, 0.08), Δt = 0.008,
        forcing = shellforcing(uh, s; shells = 1:2, eref),
        processors = (; ck = checkpointer(joinpath(dir, "ck"); interval = 0.0, meta = (; eref))),
    )
    l = DNS.checkpointlist(joinpath(dir, "ck"))
    @test length(l) == 2
    @test spectral_latest(joinpath(dir, "ck"), s) == last(l)

    # Stop-file run: a processor touches the stop file after step 4; the
    # checkpointer saves and ends the solve.
    stopfile = joinpath(dir, "stop")
    s2 = runsetup()
    uh2, eref2 = initial(s2)
    st = spectral_solve!(;
        uh = uh2, setup = s2, tlims = (0.0, 0.08), Δt = 0.008,
        forcing = shellforcing(uh2, s2; shells = 1:2, eref = eref2),
        processors = (;
            toucher = (state, s) -> (s.topo.rank == 0 && state.n == 4 && touch(stopfile); nothing),
            ck = checkpointer(joinpath(dir, "ck2"); interval = Inf, stopfile, meta = (; eref = eref2)),
        ),
    )
    @test st.n == 4 && st.t < 0.08

    # Resume from the stop checkpoint and finish: matches the uninterrupted
    # run's final state.
    s3 = runsetup()
    uh3 = specvelocity(s3)
    latest = spectral_latest(joinpath(dir, "ck2"), s3)
    md = spectral_load!(uh3, latest, s3)
    @test md.step == 4 && md.time ≈ st.t
    spectral_solve!(;
        uh = uh3, setup = s3, tlims = (md.time, 0.08), Δt = 0.008,
        forcing = shellforcing(uh3, s3; shells = 1:2, eref = md.meta["eref"]),
        nstart = md.step,
        processors = (; ck = checkpointer(joinpath(dir, "ck2"); interval = 0.0, meta = (; eref = md.meta["eref"]))),
    )
    @test maximum(abs, Array(uh3) .- Array(uh)) < 1e-12
    # resumed checkpoint numbering continued from the loaded step
    @test length(DNS.checkpointlist(joinpath(dir, "ck2"))) == 2

    MPI.Barrier(comm)
    rank == 0 && rm(dir; recursive = true)
end

@testset "spectral SFS output n=$nranks pg=$pg" for pg in procgrids(nranks)
    rank = MPI.Comm_rank(comm)
    dir = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm)
    N, nles = 24, 12
    visc = 4e-3
    filters = [1.0, 2.0]
    times = [0.0, 0.03]

    s = spectral_setup(; n = N, visc, comm, procgrid = pg)
    uh = specvelocity(s)
    spectral_randomfield!(uh, s; totalenergy = 0.3, kpeak = 2, seed = 11)
    spectral_solve!(;
        uh, setup = s, tlims = (0.0, 0.03), Δt = 0.005, tstops = times,
        processors = (; sfs = sfswriter(s; dir = joinpath(dir, "out"), nles, filters, times)),
    )
    MPI.Barrier(comm)

    # Independent serial oracle: SymmetryCode's sfs! recipe with plain FFTW
    # on the full fine rfft array, reconstructed from a serial (1-rank) run
    # of the same trajectory (the counter-based IC makes it identical).
    sser = spectral_setup(; n = N, visc, comm = MPI.COMM_SELF, procgrid = (1, 1))
    uhser = specvelocity(sser)
    spectral_randomfield!(uhser, sser; totalenergy = 0.3, kpeak = 2, seed = 11)
    freq(g, n) = g - 1 - (g ≤ (n + 1) ÷ 2 ? 0 : n)                    # fftfreq int
    function oracle(uhloc, sfilters)
        (; kcut, m) = sser
        uhh = Array(uhloc)
        # full fine rfft array of û (retained modes at fine positions)
        F = zeros(ComplexF64, N ÷ 2 + 1, N, N, 3)
        finepos(g) = (k = DNS.compactfreq(g, m, kcut); k ≥ 0 ? k + 1 : N + k + 1)
        for c = 1:3, k = 1:m, j = 1:m, i = 1:(kcut+1)
            F[i, finepos(j), finepos(k), c] = uhh[i, j, k, c]
        end
        # physical dealiased velocities and full product spectra
        v = [brfft(F[:, :, :, c], N) for c = 1:3]
        prods = ((1, 1), (2, 2), (3, 3), (1, 2), (2, 3), (3, 1))
        σf = [rfft(v[i] .* v[j]) ./ N^3 for (i, j) in prods]
        # sharp cutoff fine → coarse (SymmetryCode's cutoff!)
        finei(g) = freq(g, nles) ≥ 0 ? freq(g, nles) + 1 : N + freq(g, nles) + 1
        cutf(a) = [a[i, finei(j), finei(k)] for i = 1:(nles÷2+1), j = 1:nles, k = 1:nles]
        function gauss(a, Δ)
            b = copy(a)
            for k = 1:nles, j = 1:nles, i = 1:(nles÷2+1)
                kk = (i - 1)^2 + freq(j, nles)^2 + freq(k, nles)^2
                kk < 9 || (b[i, j, k] *= exp(-Δ^2 * kk / 24))   # forced shells protected
            end
            b
        end
        function deal(a)
            b = copy(a)
            for k = 1:nles, j = 1:nles, i = 1:(nles÷2+1)
                k1, k2, k3 = i - 1, freq(j, nles), freq(k, nles)
                keep = k1 ≤ nles ÷ 3 && abs(k2) ≤ nles ÷ 3 && abs(k3) ≤ nles ÷ 3 &&
                       (k1, k2, k3) != (0, 0, 0)
                keep || (b[i, j, k] = 0)
            end
            b
        end
        map(sfilters) do Δf
            Δ = Δf * 2π / nles
            ub = [deal(gauss(cutf(F[:, :, :, c]), Δ)) for c = 1:3]
            σ1 = [gauss(cutf(σ), Δ) for σ in σf]
            vb = [brfft(deal(u), nles) for u in ub]
            σ2 = [rfft(vb[i] .* vb[j]) ./ nles^3 for (i, j) in prods]
            τ = [deal(σ1[c] - σ2[c]) for c = 1:6]
            tr = (τ[1] .+ τ[2] .+ τ[3]) ./ 3
            for c = 1:3
                τ[c] = τ[c] .- tr
            end
            (; ub, τ)
        end
    end

    err(a, b) = maximum(abs, a .- b)
    for (j, t) in enumerate(times)
        t == 0.0 || spectral_solve!(; uh = uhser, setup = sser, tlims = (0.0, t), Δt = 0.005)
        ref = oracle(uhser, filters)
        for (k, Δf) in enumerate(filters)
            inp, out, redelta, Δ = load(
                joinpath(dir, "out", "delta=$(Δf)", "fields.jld2"),
                "inputs", "outputs", "redelta", "Δ",
            )
            @test Δ ≈ Δf * 2π / nles
            @test maximum(
                err(getproperty(inp[j], c), ref[k].ub[ci]) for
                (ci, c) in enumerate((:x, :y, :z))
            ) < 1e-12
            @test maximum(
                err(getproperty(out[j], c), ref[k].τ[ci]) for
                (ci, c) in enumerate((:xx, :yy, :zz, :xy, :yz, :zx))
            ) < 1e-12
            @test length(redelta) == 2 && all(>(0), redelta)
        end
    end
    # DNS-side metadata: times land exactly, spectra over shells 1:kcut,
    # statistics match the serial state at the final time.
    mtimes, spectra_dns, statistics_dns = load(
        joinpath(dir, "out", "dns_meta.jld2"), "times", "spectra_dns", "statistics_dns",
    )
    @test mtimes ≈ times atol = 1e-10
    @test length(spectra_dns[1]) == s.kcut
    @test statistics_dns[2].e ≈ spectral_energy(uhser, sser) rtol = 1e-10
    sples, redmean = load(
        joinpath(dir, "out", "delta=2.0", "les_meta.jld2"), "spectra_les", "redelta_mean",
    )
    @test length(sples) == 2 && length(sples[1]) == nles ÷ 3
    @test redmean > 0

    MPI.Barrier(comm)
    rank == 0 && rm(dir; recursive = true)
end

@testset "spectral filter bank offline n=$nranks pg=$pg" for pg in procgrids(nranks)
    rank = MPI.Comm_rank(comm)
    dir = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm)
    N = 24
    visc = 4e-3
    times = [0.0, 0.02]
    cells = [
        (; M = 12, kernel = :gaussian, Δfac = 2.0),
        (; M = 12, kernel = :tophat, Δfac = 3.0),
        (; M = 8, kernel = :cutoff, Δfac = 2.0),
        (; M = 8, kernel = :helmholtz, Δfac = 2.5, Δη = 27.0),
    ]
    s = spectral_setup(; n = N, visc, comm, procgrid = pg)
    uh = specvelocity(s)
    spectral_randomfield!(uh, s; totalenergy = 0.3, kpeak = 2, seed = 13)
    spectral_solve!(;
        uh, setup = s, tlims = (0.0, 0.02), Δt = 0.005, tstops = times,
        processors = (; snap = snapshotsaver(joinpath(dir, "snap"); times)),
    )
    MPI.Barrier(comm)
    prefixes = [joinpath(dir, "snap_0001"), joinpath(dir, "snap_0002")]
    # multi-kernel/multi-M bank + the legacy spec through the offline path
    sfs_offline(prefixes, s; dir = joinpath(dir, "bank"), cells)
    sfs_offline(prefixes, s; dir = joinpath(dir, "legacy"), nles = 12, filters = [2.0])
    MPI.Barrier(comm)

    # Independent serial oracle (fresh FFTW port of the recipe, per cell).
    sser = spectral_setup(; n = N, visc, comm = MPI.COMM_SELF, procgrid = (1, 1))
    uhser = specvelocity(sser)
    spectral_randomfield!(uhser, sser; totalenergy = 0.3, kpeak = 2, seed = 13)
    freqq(g, n) = g - 1 - (g ≤ (n + 1) ÷ 2 ? 0 : n)
    sincf(x) = x == 0 ? 1.0 : sin(x) / x
    prods = ((1, 1), (2, 2), (3, 3), (1, 2), (2, 3), (3, 1))
    function bankoracle(uhloc, cell)
        (; kcut, m) = sser
        (; M, kernel, Δfac) = cell
        Δ = Δfac * 2π / M
        uhh = Array(uhloc)
        F = zeros(ComplexF64, N ÷ 2 + 1, N, N, 3)
        finepos(g) = (k = DNS.compactfreq(g, m, kcut); k ≥ 0 ? k + 1 : N + k + 1)
        for c = 1:3, k = 1:m, j = 1:m, i = 1:(kcut+1)
            F[i, finepos(j), finepos(k), c] = uhh[i, j, k, c]
        end
        v = [brfft(F[:, :, :, c], N) for c = 1:3]
        σf = [rfft(v[i] .* v[j]) ./ N^3 for (i, j) in prods]
        finei(g) = freqq(g, M) ≥ 0 ? freqq(g, M) + 1 : N + freqq(g, M) + 1
        cutf(a) = [a[i, finei(j), finei(k)] for i = 1:(M÷2+1), j = 1:M, k = 1:M]
        function filt(a)
            b = copy(a)
            for k = 1:M, j = 1:M, i = 1:(M÷2+1)
                k1, k2, k3 = i - 1, freqq(j, M), freqq(k, M)
                kk = k1^2 + k2^2 + k3^2
                kk < 9 && continue   # low-k carve-out (protectshells = 2)
                b[i, j, k] *=
                    kernel == :gaussian ? exp(-Δ^2 * kk / 24) :
                    kernel == :cutoff ? (kk ≤ (π / Δ)^2 ? 1.0 : 0.0) :
                    kernel == :tophat ?
                    sincf(k1 * Δ / 2) * sincf(k2 * Δ / 2) * sincf(k3 * Δ / 2) :
                    1 / (1 + Δ^2 * kk / 24)
            end
            b
        end
        function deal(a)
            b = copy(a)
            for k = 1:M, j = 1:M, i = 1:(M÷2+1)
                k1, k2, k3 = i - 1, freqq(j, M), freqq(k, M)
                keep = k1 ≤ M ÷ 3 && abs(k2) ≤ M ÷ 3 && abs(k3) ≤ M ÷ 3 &&
                       (k1, k2, k3) != (0, 0, 0)
                keep || (b[i, j, k] = 0)
            end
            b
        end
        ub = [deal(filt(cutf(F[:, :, :, c]))) for c = 1:3]
        σ1 = [filt(cutf(σ)) for σ in σf]
        vb = [brfft(deal(u), M) for u in ub]
        σ2 = [rfft(vb[i] .* vb[j]) ./ M^3 for (i, j) in prods]
        τ = [deal(σ1[c] - σ2[c]) for c = 1:6]
        tr = (τ[1] .+ τ[2] .+ τ[3]) ./ 3
        for c = 1:3
            τ[c] = τ[c] .- tr
        end
        (; ub, τ)
    end

    err(a, b) = maximum(abs, a .- b)
    for (j, t) in enumerate(times)
        t == 0.0 ||
            spectral_solve!(; uh = uhser, setup = sser, tlims = (0.0, t), Δt = 0.005)
        for cell in cells
            ref = bankoracle(uhser, cell)
            cdir = joinpath(dir, "bank", "filter=$(cell.kernel)", "M=$(cell.M)",
                "delta=$(cell.Δfac)")
            inp, out, kern, M, Δ, deta = load(
                joinpath(cdir, "fields.jld2"),
                "inputs", "outputs", "kernel", "M", "Δ", "delta_eta",
            )
            @test kern == String(cell.kernel) && M == cell.M
            @test Δ ≈ cell.Δfac * 2π / cell.M
            @test deta === get(cell, :Δη, nothing)
            @test maximum(
                err(getproperty(inp[j], c), ref.ub[ci]) for
                (ci, c) in enumerate((:x, :y, :z))
            ) < 1e-12
            @test maximum(
                err(getproperty(out[j], c), ref.τ[ci]) for
                (ci, c) in enumerate((:xx, :yy, :zz, :xy, :yz, :zx))
            ) < 1e-12
        end
        # legacy spec through the offline path: flat layout, Gaussian
        refg = bankoracle(uhser, (; M = 12, kernel = :gaussian, Δfac = 2.0))
        inp = load(joinpath(dir, "legacy", "delta=2.0", "fields.jld2"), "inputs")
        @test maximum(
            err(getproperty(inp[j], c), refg.ub[ci]) for
            (ci, c) in enumerate((:x, :y, :z))
        ) < 1e-12
    end
    mtimes = load(joinpath(dir, "bank", "dns_meta.jld2"), "times")
    @test mtimes ≈ times atol = 1e-10
    @test isfile(joinpath(dir, "bank", "filter=cutoff", "M=8", "delta=2.0", "les_meta.jld2"))

    MPI.Barrier(comm)
    rank == 0 && rm(dir; recursive = true)
end

@testset "spectral phase randomization n=$nranks pg=$pg" for pg in procgrids(nranks)
    s = spectral_setup(; n = 24, comm, procgrid = pg)
    uh = specvelocity(s)
    spectral_randomfield!(uh, s; totalenergy = 0.4, kpeak = 3, seed = 5)
    e0 = spectral_energy(uh, s)
    sp0 = spectral_spectrum(uh, s)
    uh0 = copy(uh)
    spectral_phaserandomize!(uh, s; seed = 42)
    # preserves modal energies and incompressibility, changes the field
    @test spectral_energy(uh, s) ≈ e0 rtol = 1e-13
    @test spectral_spectrum(uh, s).E ≈ sp0.E rtol = 1e-12
    @test spectral_maxdiv(uh, s) < 1e-12
    @test maximum(abs, Array(uh) .- Array(uh0)) > 1e-3
    # Hermitian symmetry survives: the physical round trip is exact
    spec_to_phys!(s.fft.v, uh, s)
    uh2 = specvelocity(s)
    phys_to_spec!(uh2, s.fft.v, s)
    @test maximum(abs, Array(uh2) .- Array(uh)) < 1e-13
    # counter-based: decomposition-invariant
    sser = spectral_setup(; n = 24, comm = MPI.COMM_SELF, procgrid = (1, 1))
    uhser = specvelocity(sser)
    spectral_randomfield!(uhser, sser; totalenergy = 0.4, kpeak = 3, seed = 5)
    spectral_phaserandomize!(uhser, sser; seed = 42)
    (; lspec) = s
    ref = Array(uhser)[lspec.ranges[1], lspec.ranges[2], lspec.ranges[3], :]
    @test maximum(abs, Array(uh) .- ref) < 1e-14
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
