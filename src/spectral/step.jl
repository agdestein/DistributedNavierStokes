# Time integration: Wray's low-storage RK3 with integrating-factor
# viscosity. Diffusion is diagonal in spectral space, so the analytic
# factor exp(-νk²Δt) advances it exactly: writing the RK scheme in the
# transformed variable w = e^{νk²(t-tₙ)}û and mapping back to û gives the
# plain Wray update with both registers damped by the per-stage factor
# φᵢ = exp(-νk²(c_{i+1}-c_i)Δt):
#
#   du = N(ûᵢ)                       (convection only)
#   û      ← φᵢ (ustart + aᵢ Δt du)
#   ustart ← φᵢ (ustart + bᵢ Δt du)
#
# With ν = 0 this is exactly Wray's scheme; with N = 0 the φᵢ telescope to
# the exact viscous decay e^{-νk²Δt}. There is no viscous time-step limit.

"Wray RK3: `a`/`b` as in SymmetryCode's `wray3!`, `dc[i] = c[i+1] - c[i]`."
const wray3 = (;
    a = (8 / 15, 5 / 12, 3 / 4),
    b = (1 / 4, 0.0, 0.0),
    dc = (8 / 15, 2 / 15, 1 / 3),
)

@kernel function ifstage_kernel!(
    uh,
    ustart,
    @Const(du),
    @Const(kx),
    @Const(ky),
    @Const(kz),
    a,
    b,
    νdcΔt,
    Δt,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        kk = kx[i]^2 + ky[j]^2 + kz[k]^2
        φ = exp(-νdcΔt * kk)
        for c = 1:3
            st = ustart[i, j, k, c]
            d = du[i, j, k, c]
            uh[i, j, k, c] = φ * (st + a * Δt * d)
            ustart[i, j, k, c] = φ * (st + b * Δt * d)
        end
    end
end

"""
One IF-RK3 step of size `Δt`. `cache = (; ustart, du)` holds the two
state-shaped registers. Returns the rank-local maximum-velocity estimate
from the first stage (an upper bound `√(Σ_c max v_c²) ≥ max|v|`), for the
caller's CFL control of the *next* step.
"""
function spectral_step!(uh, s, Δt, cache)
    (; backend, T) = s
    (; ustart, du) = cache
    copyto!(ustart, uh)
    vmax = zero(T)
    for i = 1:3
        spectral_rhs!(du, uh, s)
        i == 1 &&
            (vmax = sqrt(sum(c -> maximum(abs2, view(s.fft.v, :, :, :, c)), 1:3)))
        ifstage_kernel!(backend)(
            uh,
            ustart,
            du,
            s.kx,
            s.ky,
            s.kz,
            T(wray3.a[i]),
            T(wray3.b[i]),
            s.visc * T(wray3.dc[i]) * Δt,
            Δt;
            ndrange = size(uh)[1:3],
        )
        spectral_project!(uh, s)
    end
    vmax
end

"""
    spectral_solve!(; uh, setup, tlims, Δt = nothing, cfl = 0.3,
                    forcing = nothing, tstops = (), nstart = 0,
                    processors = (;))

Integrate the spectral state `uh` from `tlims[1]` to `tlims[2]`. The initial
field is projected. `Δt = nothing` means CFL-adaptive stepping (convective
limit only — viscosity is exact); the velocity maximum is lagged by one
step. `forcing` is a mutating hook `forcing(uh, setup)` applied after every
step (e.g. [`shellforcing`](@ref)). Steps are shortened to land exactly on
each time in `tstops` (e.g. snapshot times, see [`snapshotsaver`](@ref)).
Each processor is called as `proc(state, setup)` after every step (and once
initially) with `state = (; uh, t, n)`; step numbering starts at `nstart`
(pass the checkpoint's `step` on restart). A processor returning `:stop`
ends the solve gracefully after the current step (see
[`checkpointer`](@ref)). Returns the final state.
"""
function spectral_solve!(;
    uh,
    setup,
    tlims,
    Δt = nothing,
    cfl = 0.3,
    forcing = nothing,
    tstops = (),
    nstart = 0,
    processors = (;),
)
    s = setup
    (; T, topo) = s
    cache = (; ustart = specvelocity(s), du = specvelocity(s))
    spectral_project!(uh, s)
    h = s.l / s.n
    ϵ = eps(T)
    vmax = if isnothing(Δt)
        spec_to_phys!(s.fft.v, uh, s)
        v = sqrt(sum(c -> maximum(abs2, view(s.fft.v, :, :, :, c)), 1:3))
        MPI.Allreduce(v, max, topo.cart)
    else
        zero(T)
    end
    t, tend = float.(tlims)
    tol = 1e-10 * (abs(tend) + 1)
    stops = sort!(Float64[x for x in tstops if t + tol < x < tend - tol])
    istop = 1
    n = nstart
    stop = false
    call!(proc) = proc(state, s) === :stop && (stop = true)
    state = (; uh, t, n)
    foreach(call!, processors)
    while !stop && t < tend - tol
        while istop ≤ length(stops) && stops[istop] ≤ t + tol
            istop += 1
        end
        tnext = istop ≤ length(stops) ? stops[istop] : tend
        Δtn = min(something(Δt, T(cfl) * h / (vmax + ϵ)), tnext - t)
        vloc = spectral_step!(uh, s, T(Δtn), cache)
        isnothing(forcing) || forcing(uh, s)
        isnothing(Δt) && (vmax = MPI.Allreduce(vloc, max, topo.cart))
        t += Δtn
        n += 1
        state = (; uh, t, n)
        foreach(call!, processors)
    end
    state
end
