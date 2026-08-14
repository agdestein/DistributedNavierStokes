# Multi-GPU validation rollout: short forced HIT from the deterministic
# counter-based IC, saving the final spectral state to $DNS_OUTDIR/final.
# The snapshot format is independent of rank count and processor grid, so
# running this once with 1 rank and once with 4 produces two files that
# must agree to roundoff (~1e-14 relative) — the decomposition-invariance
# check on real multi-device hardware (spectral_validate.sh).

using CUDA
using KernelAbstractions
using MPI
using DistributedNavierStokes

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

backend = if CUDA.functional()
    # one rank per device on each node; ranks share a device when
    # oversubscribed (single-GPU testing mode)
    nodecomm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    CUDA.device!(MPI.Comm_rank(nodecomm) % length(CUDA.devices()))
    dev = CUDA.device()
    println("rank $rank: $(CUDA.name(dev)) ($(CUDA.uuid(dev)))")
    CUDABackend()
else
    CPU()
end

outdir = get(ENV, "DNS_OUTDIR", "output-validate")
rank == 0 && mkpath(outdir)
MPI.Barrier(comm)

s = spectral_setup(;
    n = 256,
    l = 2π,
    visc = 1e-3,
    backend,
    mpibuf = get(ENV, "DNS_MPIBUF", "auto") == "host" ? :host : :auto,
)
rank == 0 && println(
    "n = $(s.n), kcut = $(s.kcut), procgrid = $(s.topo.procgrid), " *
    "MPI buffers = $(s.stagehost ? "host-staged" : "device")",
)

uh = specvelocity(s)
spectral_randomfield!(uh, s; totalenergy = 0.5, kpeak = 4, seed = 0)
forcing = shellforcing(uh, s; shells = 1:2)

log = (state, s) -> begin
    st = spectral_stats(state.uh, s)
    s.topo.rank == 0 && println(
        "n = $(lpad(state.n, 4))  t = $(round(state.t; sigdigits = 4))  " *
        "E = $(round(st.e; sigdigits = 8))  ε = $(round(st.diss; sigdigits = 4))",
    )
end

walltime = @elapsed state = spectral_solve!(;
    uh,
    setup = s,
    tlims = (0.0, 0.05),
    cfl = 0.4,
    forcing,
    processors = (; log),
)

spectral_save(joinpath(outdir, "final"), uh, s; time = state.t, step = state.n)
rank == 0 && println(
    "saved $(joinpath(outdir, "final")).bin after $(state.n) steps, " *
    "t = $(state.t), wall = $(round(walltime; sigdigits = 3)) s (incl. compilation)",
)
