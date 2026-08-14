# Import a SymmetryCode DNS state into a spectral snapshot — the V0
# twin-validation recipe: both solvers then evolve the *same* field, so the
# comparison is exact instead of statistical (CAMPAIGN.md item 6).
#
#   julia --project=examples examples/symmetrycode_import.jl warmed.jld2 out/snap [time]
#
# The JLD2 file holds `u` = (; x, y, z) full-rfft spectral velocities with
# û = F[u]/n³ (SymmetryCode's warmed-DNS format; `times` supplies the
# default snapshot time). Box size and viscosity are not stored there —
# pass DNS_L / DNS_VISC (they go into the sidecar for restarts). Serial;
# the written snapshot restarts on any rank count or processor grid.

using MPI
using DistributedNavierStokes
const DNS = DistributedNavierStokes

MPI.Init()
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run the import serially")
file, prefix = ARGS[1], ARGS[2]

d = DNS.JLD2.load(file)
u = d["u"]
t = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) :
    haskey(d, "times") && !isempty(d["times"]) ? Float64(last(d["times"])) : 0.0
nf = size(u.x, 2)
l = parse(Float64, get(ENV, "DNS_L", string(2π)))
visc = parse(Float64, get(ENV, "DNS_VISC", "1e-3"))

s = spectral_setup(; n = nf, l, visc, comm = MPI.COMM_SELF)
uh = specvelocity(s)
spectral_from_rfft!(uh, u, s)
spectral_save(prefix, uh, s; time = t, meta = (; source = basename(file)))
println("imported $file (n = $nf, kcut = $(s.kcut)) → $prefix at t = $t")
