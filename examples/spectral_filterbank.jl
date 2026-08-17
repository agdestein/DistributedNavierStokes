# Offline filter bank over stored raw DNS snapshots (data-campaign P1: the
# bank is a regenerable derivative of the raw store). Grid parameters come
# from the first snapshot's TOML sidecar; the transform grid is the leanest
# alias-free one (n = 3·kcut, rounded even), independent of the grid the
# run used. Configure the bank via the DNS_* environment (see the BANK
# block), then:
#
#   srun/mpiexec -n 4 julia --project=examples examples/spectral_filterbank.jl out/snap_*.bin
#
# (prefixes with or without .bin/.toml; processed in sorted order).
# Environment: DNS_MS, DNS_DETA, DNS_KERNELS, DNS_LEGACY, DNS_ETA (the
# bank — defaults in the BANK block), DNS_OUTDIR (default "bank"),
# DNS_MPIBUF (host|nccl|auto), DNS_PHASESEED (set an integer for the
# kinematic-null pass).

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

# ---- BANK (env-configured; ProbabilisticClosure data-campaign.md §3) -----
# Δ/η-pinned columns (DNS_DETA × DNS_KERNELS; `etacells`'s window rule
# picks which of DNS_MS carries each width), plus optional legacy Gaussian
# Δ/h columns at M = min(128, 2·kcut) (DNS_LEGACY — SymmetryCode's
# round-one cells). η defaults to the sidecar mean over the snapshots
# passed; set DNS_ETA to pin it (e.g. one value shared by an M ≤ 128 pass
# and an M = 256 subset pass, so matched-Δ/η columns match exactly).
splitlist(str) = [String(x) for x in split(str, ",") if !isempty(x)]
Ms = filter(M -> M ÷ 2 ≤ kcut, parse.(Int, splitlist(get(ENV, "DNS_MS", "64,128"))))
deta = parse.(Float64, splitlist(get(ENV, "DNS_DETA", "18,27,40,60")))
kernels = Symbol.(splitlist(get(ENV, "DNS_KERNELS", "gaussian,cutoff,tophat,helmholtz")))
legacy = parse.(Float64, splitlist(get(ENV, "DNS_LEGACY", "")))
etas = [Float64(DNS.TOML.parsefile(p * ".toml")["meta"]["eta"]) for p in prefixes]
eta = something(tryparse(Float64, get(ENV, "DNS_ETA", "")), sum(etas) / length(etas))
cells = [
    [(; M = min(128, 2kcut), kernel = :gaussian, Δfac) for Δfac in legacy]
    isempty(deta) || isempty(Ms) ? [] : etacells(; deta, eta, l, Ms, kernels)
]
isempty(cells) && error("empty bank (η = $eta): check DNS_MS/DNS_DETA/DNS_LEGACY")
outtype = M -> M ≥ 256 ? Float32 : Float64
phaseseed = let p = get(ENV, "DNS_PHASESEED", "")
    isempty(p) ? nothing : parse(Int, p)
end
# --------------------------------------------------------------------------

if rank == 0
    println(
        "η = $eta (sidecar mean $(sum(etas) / length(etas)), " *
        "range $(minimum(etas))–$(maximum(etas)))",
    )
    for c in cells
        Δη = get(c, :Δη, nothing)
        println(
            "  cell: $(c.kernel) M = $(c.M) Δfac = $(c.Δfac)" *
            (Δη === nothing ? " (legacy)" : " (Δ/η = $Δη)"),
        )
    end
end

outdir = get(ENV, "DNS_OUTDIR", "bank")
walltime = @elapsed sfs_offline(prefixes, s; dir = outdir, cells, outtype, phaseseed)
rank == 0 && println(
    "bank of $(length(cells)) cells × $(length(prefixes)) snapshots → $outdir " *
    "($(round(walltime; sigdigits = 3)) s)" *
    (phaseseed === nothing ? "" : " [kinematic null, phaseseed = $phaseseed]"),
)
