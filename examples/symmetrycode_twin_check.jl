# Spectral-solver side of the twin comparison (V0 dry run): import the
# warmed SymmetryCode field written by `symmetrycode_twin_dns.jl`, replay
# each fixed-Δt rollout window with the distributed solver (same forcing,
# same Δt sequence), and compare final fields in the energy norm. The two
# solvers share formulation, tableau, and truncation and differ only in the
# viscous treatment (integrating factor vs explicit stress), so the
# difference must converge away at ~3rd order in Δt — a plateau or O(1)
# offset would mean a real mismatch (forcing, normalization, indexing).
#
#   julia --project=examples examples/symmetrycode_twin_check.jl <outdir>
#
# Serial (the import requires one rank); CPU; seconds to run.

using MPI
using Printf
using DistributedNavierStokes
const DNS = DistributedNavierStokes

MPI.Init()
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run serially")
outdir = ARGS[1]

warmed = DNS.JLD2.load(joinpath(outdir, "warmed.jld2"))
nf = size(warmed["u"].x, 2)
s = spectral_setup(;
    n = nf,
    l = warmed["l"],
    visc = warmed["visc"],
    comm = MPI.COMM_SELF,
)
uh0 = specvelocity(s)
spectral_from_rfft!(uh0, warmed["u"], s)

# Import consistency: the statistics of the imported state must reproduce
# SymmetryCode's (identical definitions on the identical field).
stref = warmed["statistics"][end]
st0 = spectral_stats(uh0, s)
println("imported warmed field, n = $nf, kcut = $(s.kcut), visc = $(s.visc)")
for k in (:e, :diss, :Re_tay, :kmax_eta)
    a, b = getfield(st0, k), getfield(stref, k)
    @printf("  %-8s  spectral %-15.8g symcode %-15.8g rel %.2e\n", k, a, b, abs(a - b) / abs(b))
end

# Physical (rfft-double-counted) energy norm on the truncated state.
wnorm(a) = sqrt(sum(c -> sum(s.w1r .* abs2.(view(a, :, :, :, c))), 1:3))

rollouts = filter(startswith("rollout_"), readdir(outdir))
results = map(rollouts) do file
    r = DNS.JLD2.load(joinpath(outdir, file))
    Δt, T = r["Δt"], r["T"]
    uh = copy(uh0)
    forcing = shellforcing(uh, s; shells = 1:2)
    state = spectral_solve!(; uh, setup = s, tlims = (0.0, T), Δt, forcing)
    state.n == r["nsteps"] || @warn "step-count mismatch" state.n r["nsteps"]
    uhref = specvelocity(s)
    spectral_from_rfft!(uhref, r["u"], s)
    (; Δt, nsteps = r["nsteps"], reldiff = wnorm(uh .- uhref) / wnorm(uhref))
end
sort!(results; by = r -> -r.Δt)

println("\ncross-solver field difference after T = 1 (IF vs explicit viscosity):")
println("  Δt          nsteps   rel L2 diff   observed order")
for (i, r) in enumerate(results)
    order = i == 1 ? "" :
            @sprintf("%.2f", log2(results[i-1].reldiff / r.reldiff))
    @printf("  %-10.6g  %-7d  %-12.4e  %s\n", r.Δt, r.nsteps, r.reldiff, order)
end
