# R1 — the ν = 1e-4 resolution-check DNS (data-campaign.md §2): round-one
# physics (l = 2π, E₀ = 0.2, k^(-5/3) init, shell clamp on shells 1-2) on a
# grid that lifts k_max·η from 1.09 to ≥ 1.3. Three phases in one driver,
# checkpoint/restartable at any point (submit via snellius/spectral_r1.sh):
#
# 1. Warm-up to t = DNS_TWARM (default 25 ≈ 3 nominal t_int — the
#    campaign's hygiene fix over round one's 0.65 t_int), stationarity
#    drift recorded in stats.csv throughout.
# 2. Schedule: at warm-up end the *measured* t_int fixes the sampling —
#    law mode (snapshots every t_int/2 over 10 t_int) plus a test-style
#    dense window (40 snapshots over the first t_int) — written once to
#    schedule.toml so every restart replays identical times.
# 3. Production: raw truncated-f64 snapshots at the scheduled times (P1:
#    the filter bank is post-processing — examples/spectral_filterbank.jl
#    on the snapshot prefixes; no in-situ sfswriter, it cannot span
#    restarts).
#
# Environment knobs (set by the job script; sbatch does not forward
# submission env, so they are baked there): DNS_N (972 default; 1080 needs
# 4 GPUs), DNS_VISC, DNS_SEED, DNS_TWARM, DNS_OUTDIR, DNS_MPIBUF,
# DNS_DUMMY=1 (tiny grid + short windows, full pipeline — run it before
# every real submission).

using CUDA
using NCCL
using KernelAbstractions
using MPI
using DistributedNavierStokes
const DNS = DistributedNavierStokes

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
if CUDA.functional()
    CUDA.device!(rank % length(CUDA.devices()))
    backend = CUDABackend()
else
    backend = CPU()
end

dummy = get(ENV, "DNS_DUMMY", "0") == "1"
n = parse(Int, get(ENV, "DNS_N", dummy ? "64" : "972"))
visc = parse(Float64, get(ENV, "DNS_VISC", dummy ? "5e-3" : "1e-4"))
seed = parse(Int, get(ENV, "DNS_SEED", "200"))
t_warm = parse(Float64, get(ENV, "DNS_TWARM", dummy ? "2.0" : "25.0"))
nlaw = dummy ? 2 : 20     # law samples after the first: window = nlaw·t_int/2
ndense = dummy ? 5 : 40   # test-style samples over the first t_int
totalenergy = 0.2
cfl = 0.85  # fraction of the RK3 stability boundary (see spectral_solve!)
outdir = get(ENV, "DNS_OUTDIR", "output-r1")
snapdir = joinpath(outdir, "snapshots")
rank == 0 && (mkpath(outdir); mkpath(snapdir))
MPI.Barrier(comm)
ckpt = joinpath(outdir, "ckpt")
stopfile = joinpath(outdir, "stop")
tol = 1e-10

s = spectral_setup(;
    n,
    l = 2π,
    visc,
    backend,
    mpibuf = let m = get(ENV, "DNS_MPIBUF", "auto")
        m == "host" ? :host : m == "nccl" ? :nccl : :auto
    end,
)
rank == 0 && println(
    "R1: n = $n, kcut = $(s.kcut), visc = $visc, seed = $seed, " *
    "procgrid = $(s.topo.procgrid), mpibuf = $(s.mpimode)$(dummy ? " [DUMMY]" : "")",
)

uh = specvelocity(s)
latest = spectral_latest(ckpt, s)
if latest === nothing
    spectral_randomfield!(uh, s; totalenergy, seed, profile = k -> k > 0 ? float(k)^(-5 / 3) : 0.0)
    eref = shell_energies(uh, s, 1:2)
    t0, nstart = 0.0, 0
else
    md = spectral_load!(uh, latest, s)
    eref = collect(Float64, md.meta["eref"])
    t0, nstart = md.time, md.step
    rank == 0 && println("restarting from $latest (t = $t0, step = $nstart)")
end
forcing = shellforcing(uh, s; shells = 1:2, eref)

log = (state, s) -> begin
    state.n % (dummy ? 20 : 50) == 0 || return
    st = spectral_stats(state.uh, s)
    s.topo.rank == 0 && println(
        "n = $(lpad(state.n, 6))  t = $(round(state.t; sigdigits = 5))  " *
        "E = $(round(st.e; sigdigits = 5))  ε = $(round(st.diss; sigdigits = 4))  " *
        "Re_λ = $(round(st.Re_tay; sigdigits = 4))  kmaxη = $(round(st.kmax_eta; sigdigits = 3))",
    )
end
stats = statswriter(; file = joinpath(outdir, "stats.csv"), nupdate = 10)
checkpoints = checkpointer(ckpt; interval = 1800.0, stopfile, meta = (; eref))

# Phase 1: warm-up (drift record only, no snapshots).
if t0 < t_warm - tol
    state = spectral_solve!(;
        uh, setup = s, tlims = (t0, t_warm), cfl, forcing, nstart,
        processors = (; log, stats, ckpt = checkpoints),
    )
    t0, nstart = state.t, state.n
    if t0 < t_warm - tol
        rank == 0 && println("stopped during warm-up at t = $t0 (checkpointed; resubmit to continue)")
        exit()
    end
end

# Phase 2: the sampling schedule, from the warm-up-end *measured* t_int
# (identical on all ranks — spectral_stats reduces globally), persisted so
# restarts replay the exact same times and file numbering.
schedfile = joinpath(outdir, "schedule.toml")
sched = MPI.bcast(
    rank == 0 ? (isfile(schedfile) ? DNS.TOML.parsefile(schedfile) : nothing) : nothing,
    comm,
)
if sched === nothing
    tint = clamp(spectral_stats(uh, s).t_int, 2.0, 15.0)
    law = t_warm .+ (0:nlaw) .* (tint / 2)
    dense = t_warm .+ (1:ndense) .* (tint / ndense)
    times = sort!(unique(round.(vcat(law, dense); digits = 6)))
    sched = Dict("t_int" => tint, "t_warm" => t_warm, "times" => times)
    if rank == 0
        tmp = schedfile * ".tmp"
        open(io -> DNS.TOML.print(io, sched), tmp, "w")
        mv(tmp, schedfile; force = true)
        println("schedule: t_int = $tint, $(length(times)) snapshots, t ∈ [$(times[1]), $(times[end])]")
    end
end
times = Vector{Float64}(sched["times"])

# Phase 3: production — raw snapshots at the scheduled times. `tstart = t0`
# skips the entries an earlier job already saved.
snaps = snapshotsaver(
    joinpath(snapdir, "r1");
    times,
    meta = (; run = "R1", seed, eref),
    tstart = t0,
)
state = spectral_solve!(;
    uh, setup = s, tlims = (t0, times[end]), cfl, forcing, nstart, tstops = times,
    processors = (; log, stats, snaps, ckpt = checkpoints),
)

if state.t < times[end] - tol
    rank == 0 && println("stopped at t = $(state.t) (checkpointed; resubmit to continue)")
else
    st = spectral_stats(uh, s)
    rank == 0 && println(
        "R1 complete: t = $(state.t), $(length(times)) snapshots under $snapdir\n" *
        "final stats: Re_λ = $(st.Re_tay), kmaxη = $(st.kmax_eta), t_int = $(st.t_int), η = $(st.l_kol)",
    )
end
