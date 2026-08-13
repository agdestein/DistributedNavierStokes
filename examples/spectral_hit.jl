# Forced homogeneous isotropic turbulence with the pseudo-spectral solver,
# with checkpoint/restart: if a checkpoint exists under $DNS_OUTDIR the run
# resumes from it, and touching $DNS_OUTDIR/stop makes it checkpoint and
# exit gracefully (the SLURM time-limit pattern, see
# snellius/spectral_h100.sh).
#
# Run distributed (each rank picks its own GPU):
#   mpiexec -n 4 julia --project=examples examples/spectral_hit.jl
# or single-process (CPU: swap the backend):
#   julia --project=examples examples/spectral_hit.jl

using CUDA
using KernelAbstractions
using MPI
using DistributedNavierStokes

MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if CUDA.functional()
    # One rank per GPU under SLURM/mpiexec with --gpus-per-task=1; all ranks
    # see device 0 then. Otherwise round-robin over visible devices.
    CUDA.device!(rank % length(CUDA.devices()))
    backend = CUDABackend()
else
    backend = CPU()
end

outdir = get(ENV, "DNS_OUTDIR", "output")
rank == 0 && mkpath(outdir)
MPI.Barrier(MPI.COMM_WORLD)
ckpt = joinpath(outdir, "ckpt")
stopfile = joinpath(outdir, "stop")
tend = 5.0

s = spectral_setup(;
    n = 128,                  # transform grid (kcut = n ÷ 3)
    l = 2π,
    visc = 1e-3,
    backend,
    mpibuf = get(ENV, "DNS_MPIBUF", "auto") == "host" ? :host : :auto,
)
rank == 0 && println("n = $(s.n), kcut = $(s.kcut), procgrid = $(s.topo.procgrid)")

uh = specvelocity(s)
latest = spectral_latest(ckpt, s)
if latest === nothing
    spectral_randomfield!(uh, s; totalenergy = 0.5, kpeak = 4, seed = 0)
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
    state.n % 20 == 0 || return
    st = spectral_stats(state.uh, s)
    s.topo.rank == 0 && println(
        "n = $(lpad(state.n, 5))  t = $(round(state.t; sigdigits = 4))  " *
        "E = $(round(st.e; sigdigits = 5))  ε = $(round(st.diss; sigdigits = 4))  " *
        "Re_λ = $(round(st.Re_tay; sigdigits = 4))  kmaxη = $(round(st.kmax_eta; sigdigits = 3))",
    )
end

state = spectral_solve!(;
    uh,
    setup = s,
    tlims = (t0, tend),
    cfl = 0.4,
    forcing,
    nstart,
    processors = (;
        log,
        ckpt = checkpointer(ckpt; interval = 600.0, stopfile, meta = (; eref)),
    ),
)

if state.t < tend
    rank == 0 && println("stopped at t = $(state.t) (checkpointed; resubmit to continue)")
else
    sp = spectral_spectrum(uh, s)
    if rank == 0
        println("final spectrum (κ, E):")
        foreach((κ, E) -> println("  ", lpad(κ, 3), "  ", E), sp.κ, sp.E)
    end
end
