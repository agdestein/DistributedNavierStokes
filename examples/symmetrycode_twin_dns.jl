# SymmetryCode side of the twin comparison (V0 dry run, CAMPAIGN.md item 6):
# warm up a small forced HIT case with the original single-GPU solver, save
# the warmed field in dnsfile format, then roll it out at several *fixed* Δt
# with shell forcing, saving each final field. The spectral solver replays
# the same windows from the imported warmed field
# (`symmetrycode_twin_check.jl`); with matched Δt the two solvers differ
# only in the viscous treatment (integrating factor vs explicit), so the
# cross-solver field difference must shrink at ~3rd order in Δt.
#
#   julia --project=$HOME/Projects/Symmetry/SymmetryCode \
#       examples/symmetrycode_twin_dns.jl <outdir> [n] [visc]
#
# Parameters are `case_snellius` scaled down (l = 2π, totalenergy = 0.2,
# warm-up to t = 5, cfl = 0.35); default n = 64, visc = 5e-3 keeps
# kmax_eta ≈ 1.5 at that n. CPU, serial, seconds to run.

import SymmetryCode as S
using JLD2
using KernelAbstractions: CPU
using Random: Xoshiro

outdir = mkpath(ARGS[1])
n = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 64
visc = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : 5e-3
l, totalenergy, warmup_tstop, cfl, seed = 2π, 0.2, 5.0, 0.35, 1
dts = [1 / 64, 1 / 128, 1 / 256]   # powers of two: nsteps * Δt is exact
T = 1.0                            # rollout window

g = S.Grid{3}(; l, n, backend = CPU())
cache = S.getcache(g)
sc = S.statscache(g)
rng = Xoshiro(seed)
u = S.randomfield(S.linear_profile_3D, g; rng, totalenergy)
shells = S.energy_shells(g, [1, 2], u)

# Warm-up, as in `create_dns` (adaptive Δt; the statistics call empties the
# dynamically inert ghost modes each step, so the saved field is truncated).
t = 0.0
times = [t]
statistics = [S.turbulence_statistics(u, visc, g, sc)]
while t < warmup_tstop
    Δt = cfl * S.propose_timestep(u, g, visc, cache)
    Δt = min(Δt, warmup_tstop - t)
    global t += Δt
    S.wray3!(S.convectiondiffusion!, u, Δt, g, cache; visc)
    S.maintain_shell_energy!(u, shells)
    push!(times, t)
    push!(statistics, S.turbulence_statistics(u, visc, g, sc))
end
st = statistics[end]
println("warm-up done: $(length(times) - 1) steps to t = $t")
println("  e = $(st.e), diss = $(st.diss), Re_tay = $(st.Re_tay), kmax_eta = $(st.kmax_eta)")
jldsave(joinpath(outdir, "warmed.jld2"); u, times, statistics, l, visc)

# Fixed-Δt rollouts, each restarted from the warmed field. The shell
# references are recomputed from the warmed field — the spectral side does
# the same on the imported state, so both clamp to identical targets.
u0 = map(copy, u)
for Δt in dts
    foreach(copyto!, u, u0)
    rshells = S.energy_shells(g, [1, 2], u)
    nsteps = Int(round(T / Δt))
    for _ = 1:nsteps
        S.wray3!(S.convectiondiffusion!, u, Δt, g, cache; visc)
        S.maintain_shell_energy!(u, rshells)
    end
    rst = S.turbulence_statistics(u, visc, g, sc)   # also empties ghost modes
    println("rollout Δt = $Δt: $nsteps steps, e = $(rst.e), diss = $(rst.diss)")
    jldsave(joinpath(outdir, "rollout_$nsteps.jld2"); u, Δt, nsteps, T, statistics = rst)
end
println("saved to $outdir")
