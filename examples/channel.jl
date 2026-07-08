# Turbulent-ish channel flow demo, CPU or GPU, any number of MPI ranks.
#
# One-time setup (points MPI.jl at the system CUDA-aware MPI; skip for
# CPU-only runs, which also work with the default MPI_jll):
#
#     julia --project=examples -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
#
# Run:
#
#     mpiexec -n 4 julia --project=examples examples/channel.jl

using MPI
using CUDA
using KernelAbstractions
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

s = setup(;
    n = (64, 48, 32),
    lims = ((0.0, 2π), (-1.0, 1.0), (0.0, π)),
    bc = (:periodic, :wall, :periodic),
    stretch = t -> (tanh(1.5 * (2t - 1)) / tanh(1.5) + 1) / 2,
    visc = 1 / 180,
    bodyforce = (1.0, 0.0, 0.0),
    mpibuf = Symbol(get(ENV, "DNS_MPIBUF", "auto")),
    backend,
)
rank == 0 && println(
    "ranks = $(MPI.Comm_size(comm)), procgrid = $(s.topo.procgrid), " *
    "backend = $backend, CUDA-aware MPI = $(MPI.has_cuda()), " *
    "MPI buffers = $(s.stagehost ? "host-staged" : "device")",
)

u = vectorfield(s)
velocityfield!(
    u,
    s;
    x = (x, y, z) -> (1 - y^2) * (1 + 0.2 * sin(2x) * cos(3z) + 0.1 * cos(x) * sin(2z)),
    y = (x, y, z) -> 0.1 * (1 - y^2) * sin(3x) * sin(2z),
)

cs = channelstats(s; nupdate = 5)
st = solve!(;
    u,
    setup = s,
    tlims = (0.0, 2.0),
    processors = (; cs.sample, log = DistributedNavierStokes.logger(; nupdate = 25)),
)

prof = channelprofiles(cs, s)
div = maxdiv(u, s)
if rank == 0
    println("done: t = $(st.t), n = $(st.n) steps, $(prof.nsamples) samples, maxdiv = $div")
    println("U profile (centerline $(prof.U[end÷2])):")
    for (y, U) in zip(prof.y, prof.U)
        println("  y = $(round(y; digits = 3)):  U = $(round(U; digits = 4))")
    end
end
