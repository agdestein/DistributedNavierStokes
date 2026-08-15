# S0 scaling benchmark — one configuration per process (CAMPAIGN.md).
#
#   julia --project=examples examples/snellius/scaling.jl <n> <nsteps> [PxQ|auto] [save] [sfs] [device] [nccl]
#
# Runs a synthetic forced-HIT configuration at transform grid n³, times
# `nsteps` full solver steps individually after two throw-away warm-up
# steps (the first carries compilation — its wall time is recorded
# separately), plus a transform-only round-trip microbenchmark, per-rank
# device memory, and optionally one snapshot save (`save`) and one SFS
# filter sample (`sfs`). `device` runs with mpibuf = :device instead of
# the default from $DNS_MPIBUF. Rank 0 appends one TOML `[[run]]` block
# per invocation to $S0_RESULTS (default "s0-results.toml").

using CUDA
using NCCL
using KernelAbstractions
using MPI
using DistributedNavierStokes
const DNS = DistributedNavierStokes

twall = time()
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

n = parse(Int, ARGS[1])
nsteps = parse(Int, ARGS[2])
procgrid = nothing
dosave, dosfs, dodevice, donccl = false, false, false, false
for a in ARGS[3:end]
    if a == "save"
        global dosave = true
    elseif a == "sfs"
        global dosfs = true
    elseif a == "device"
        global dodevice = true
    elseif a == "nccl"
        global donccl = true
    elseif a != "auto"
        global procgrid = Tuple(parse.(Int, split(a, "x")))
    end
end
mpibuf =
    donccl ? :nccl :
    dodevice ? :device : get(ENV, "DNS_MPIBUF", "auto") == "host" ? :host : :auto

backend = if CUDA.functional()
    nodecomm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    CUDA.device!(MPI.Comm_rank(nodecomm) % length(CUDA.devices()))
    dev = CUDA.device()
    println("rank $rank: $(CUDA.name(dev)) ($(CUDA.uuid(dev)))")
    CUDABackend()
else
    CPU()
end

s = spectral_setup(; n, l = 2π, visc = 1e-3, backend, mpibuf, procgrid)
(; topo, T) = s
cart = topo.cart
rank == 0 && println(
    "n = $n, kcut = $(s.kcut), procgrid = $(topo.procgrid), " *
    "MPI buffers = $(s.mpimode)",
)

# Round-one-like state: k^-5/3 spectrum, E₀ = 0.2, shell-clamp forcing.
uh = specvelocity(s)
spectral_randomfield!(
    uh, s;
    totalenergy = 0.2, seed = 0,
    profile = k -> k == 0 ? 0.0 : float(k)^(-5 / 3),
)
forcing = shellforcing(uh, s; shells = 1:2)
cache = (; ustart = specvelocity(s), du = specvelocity(s))
globalvmax(vloc) = MPI.Allreduce(vloc, max, cart)

# CFL-proposed dt at this field (recorded for steps-per-time-unit
# pricing), then held fixed so every timed step does identical work.
spec_to_phys!(s.fft.v, uh, s)
vmax = globalvmax(DNS.vsummax(s.fft.v))
dt = T(0.85) * T(√3) / (T(2π) * s.kcut / s.l) / vmax

# A full production step, as spectral_solve! executes it.
function prodstep!()
    vloc = DNS.spectral_step!(uh, s, dt, cache)
    forcing(uh, s)
    globalvmax(vloc)
end
sync() = (KernelAbstractions.synchronize(backend); MPI.Barrier(cart))

# Warm-up: step 1 carries compilation (recorded), step 2 confirms steady.
MPI.Barrier(cart)
tcompile = @elapsed (prodstep!(); sync())
prodstep!()
sync()

steptimes = map(1:nsteps) do _
    t0 = time_ns()
    prodstep!()
    sync()
    (time_ns() - t0) / 1e9
end

transtimes = map(1:nsteps) do _
    t0 = time_ns()
    spec_to_phys!(s.fft.v, uh, s)
    phys_to_spec!(uh, s.fft.v, s)
    sync()
    (time_ns() - t0) / 1e9
end

memgb = if CUDA.functional()
    (CUDA.total_memory() - CUDA.free_memory()) / 2^30
else
    Sys.maxrss() / 2^30
end
memgb = MPI.Allreduce(memgb, max, cart)

savetime = savegb = 0.0
if dosave
    prefix = joinpath(get(ENV, "S0_SAVEDIR", "."), "s0save-n$n-r$(MPI.Comm_size(comm))")
    savetime = @elapsed spectral_save(prefix, uh, s; time = 0.0, step = 0)
    MPI.Barrier(cart)
    savegb = filesize(prefix * ".bin") / 2^30
    rank == 0 && foreach(rm, (prefix * ".bin", prefix * ".toml"))
end

sfstime = 0.0
if dosfs
    nles = min(128, s.kcut - s.kcut % 2)
    sam = DNS.lessampler(s; nles, filters = [2.0])
    DNS.sfs_sample!(sam, uh, s)   # warm-up (compilation, FFTW plans)
    sync()
    sfstime = @elapsed (DNS.sfs_sample!(sam, uh, s); sync())
end

if rank == 0
    stats(x) = (minimum(x), sort(x)[(length(x)+1)÷2], sum(x) / length(x))
    smin, smed, smean = stats(steptimes)
    tmin, tmed, _ = stats(transtimes)
    open(get(ENV, "S0_RESULTS", "s0-results.toml"), "a") do io
        println(io, "\n[[run]]")
        println(io, "n = $n")
        println(io, "ranks = $(MPI.Comm_size(comm))")
        println(io, "procgrid = [$(join(topo.procgrid, ", "))]")
        println(io, "mpibuf = \"$(s.mpimode)\"")
        println(io, "gpu = $(CUDA.functional())")
        println(io, "nsteps = $nsteps")
        println(io, "step_min = $smin")
        println(io, "step_median = $smed")
        println(io, "step_mean = $smean")
        println(io, "transform_min = $tmin")
        println(io, "transform_median = $tmed")
        println(io, "compile_s = $tcompile")
        println(io, "dt_cfl = $dt")
        println(io, "mem_gb_max = $memgb")
        dosave && println(io, "save_s = $savetime")
        dosave && println(io, "save_gb = $savegb")
        dosave && println(io, "save_gbps = $(savegb / savetime)")
        dosfs && println(io, "sfs_s = $sfstime")
        println(io, "total_wall_s = $(time() - twall)")
    end
    println(
        "n = $n ranks = $(MPI.Comm_size(comm)): step $(round(smed; sigdigits = 4)) s " *
        "(min $(round(smin; sigdigits = 4))), transform $(round(tmed; sigdigits = 4)) s, " *
        "compile $(round(tcompile; sigdigits = 4)) s, mem $(round(memgb; sigdigits = 4)) GB",
    )
end
