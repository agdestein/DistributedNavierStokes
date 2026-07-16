# Forced homogeneous isotropic turbulence with the pseudo-spectral solver.
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

s = spectral_setup(;
    n = 128,                  # transform grid (kcut = n ÷ 3)
    l = 2π,
    visc = 1e-3,
    backend,
    mpibuf = get(ENV, "DNS_MPIBUF", "auto") == "host" ? :host : :auto,
)
rank == 0 && println("n = $(s.n), kcut = $(s.kcut), procgrid = $(s.topo.procgrid)")

uh = specvelocity(s)
spectral_randomfield!(uh, s; totalenergy = 0.5, kpeak = 4, seed = 0)
forcing = shellforcing(uh, s; shells = 1:2)

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
    tlims = (0.0, 5.0),
    cfl = 0.4,
    forcing,
    processors = (; log),
)

sp = spectral_spectrum(uh, s)
if rank == 0
    println("final spectrum (κ, E):")
    foreach((κ, E) -> println("  ", lpad(κ, 3), "  ", E), sp.κ, sp.E)
end
