# Offline filter bank over stored raw DNS snapshots (data-campaign P1: the
# bank is a regenerable derivative of the raw store). Grid parameters come
# from the first snapshot's TOML sidecar; the transform grid is the leanest
# alias-free one (n = 3·kcut, rounded even), independent of the grid the
# run used. Edit the BANK block, then:
#
#   srun/mpiexec -n 4 julia --project=examples examples/spectral_filterbank.jl out/snap_*.bin
#
# (prefixes with or without .bin/.toml; processed in sorted order).
# Environment: DNS_OUTDIR (default "bank"), DNS_MPIBUF (host|nccl|auto),
# DNS_PHASESEED (set an integer for the kinematic-null pass).

using CUDA
using NCCL
using KernelAbstractions
using MPI
using DistributedNavierStokes
const DNS = DistributedNavierStokes

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

prefixes = sort!(unique!([replace(a, r"\.(bin|toml)$" => "") for a in ARGS]))
isempty(prefixes) && error("pass snapshot prefixes (e.g. out/snap_*.bin)")

backend = if CUDA.functional()
    nodecomm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    CUDA.device!(MPI.Comm_rank(nodecomm) % length(CUDA.devices()))
    CUDABackend()
else
    CPU()
end

md = DNS.TOML.parsefile(prefixes[1] * ".toml")
kcut, l, visc = md["kcut"], md["l"], md["visc"]
n = 3kcut + isodd(3kcut)   # leanest alias-free even transform grid
s = spectral_setup(;
    n, kcut, l, visc, backend,
    mpibuf = let m = get(ENV, "DNS_MPIBUF", "auto")
        m == "host" ? :host : m == "nccl" ? :nccl : :auto
    end,
)
rank == 0 && println(
    "$(length(prefixes)) snapshots, kcut = $kcut (bank grid n = $n, run grid n = $(md["n"])), " *
    "visc = $visc, procgrid = $(s.topo.procgrid), MPI buffers = $(s.mpimode)",
)

# ---- BANK (edit me) ------------------------------------------------------
# Explicit cells (Δ = Δfac·l/M), or pin widths in Δ/η via etacells once η
# is known (e.g. l_kol from a pilot's statistics_dns):
#   cells = etacells(; deta = [18, 27, 40], eta = 0.012, l,
#                    Ms = [64, 128, 256], kernels = (:gaussian, :cutoff, :tophat))
cells = [
    (; M, kernel, Δfac) for M in (min(128, 2kcut),) for
    kernel in (:gaussian, :cutoff, :tophat) for Δfac in (2.0, 3.0, 4.0)
]
outtype = M -> M ≥ 256 ? Float32 : Float64
phaseseed = let p = get(ENV, "DNS_PHASESEED", "")
    isempty(p) ? nothing : parse(Int, p)
end
# --------------------------------------------------------------------------

outdir = get(ENV, "DNS_OUTDIR", "bank")
walltime = @elapsed sfs_offline(prefixes, s; dir = outdir, cells, outtype, phaseseed)
rank == 0 && println(
    "bank of $(length(cells)) cells × $(length(prefixes)) snapshots → $outdir " *
    "($(round(walltime; sigdigits = 3)) s)" *
    (phaseseed === nothing ? "" : " [kinematic null, phaseseed = $phaseseed]"),
)
