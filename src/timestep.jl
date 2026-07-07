# Explicit time integration: Williamson-style low-storage RK3 (two
# registers: the velocity `u` and one accumulator `q`), with a pressure
# projection after every stage.

"""
Low-storage RK3 coefficients (Williamson): `q ← A q + Δt F(u); u ← u + B q`.
`α` are the effective stage time fractions (`B q = α Δt F` for constant
`F`), used by the per-stage Crank-Nicolson diffusion.
"""
const rk3 = (;
    A = (0.0, -5 / 9, -153 / 128),
    B = (1 / 3, 15 / 16, 8 / 15),
    α = (1 / 3, 5 / 12, 1 / 4),
)

"Interior (ghost-free) view of a ghosted field."
interior(φ, w) = view(φ, ntuple(d -> (w+1):(size(φ, d)-w), 3)...)

"""
Make `u` divergence-free: solve the Poisson equation for the pseudo-pressure
and subtract its gradient. `u` must have valid halos on entry; on exit both
`u` and `ψ` have valid halos. `ψ` absorbs the time-step scaling (it is
`B Δt` times the physical pressure when called inside a Runge-Kutta stage).
"""
function project!(u, ψ, setup)
    divergence!(setup.poisson.r, u, setup)
    poisson!(ψ, setup)
    exchange_halo!(ψ, :p, setup)
    pressuregrad_update!(u, ψ, setup)
    exchange_halo!(u, setup)
end

"""
One RK3 step of size `Δt`. `u` (and `p` in semi-implicit mode) must have
valid halos (maintained). In semi-implicit mode `p` is the lagged pressure
(units of pressure per stage time): its gradient is part of the explicit
right-hand side and each stage's projection solves only for the increment
`δ`, which keeps the pressure-splitting error of the Crank-Nicolson
y-diffusion at second order. In explicit mode `δ` may alias `p`.
"""
function step!(u, p, q, δ, setup, Δt)
    (; ranges) = setup
    for (A, B, α) in zip(setup.rk.A, setup.rk.B, setup.rk.α)
        momentum!(q, u, A, Δt, setup)
        if setup.imexy
            imex_update!(u, q, p, B, α, Δt, setup)
            exchange_halo!(u, setup)
            project!(u, δ, setup)
            rp = ranges.p
            vp = view(p, rp...)
            @. vp += $view(δ, rp...) / (α * Δt)
            exchange_halo!(p, :p, setup)
        else
            for (c, φ) in pairs(u)
                r = ranges[c]
                v = view(φ, r...)
                @. v += B * $view(q[c], r...)
            end
            exchange_halo!(u, setup)
            project!(u, p, setup)
        end
    end
end

"""
Largest stable time step: CFL-limited convection (per direction, with the
smallest y cell) and explicit diffusion, reduced over all ranks.
"""
function cfl_timestep(u, setup; cfl)
    (; grid, visc, topo, w, T) = setup
    ϵ = eps(T)
    conv = min(
        grid.Δx / (maximum(abs, interior(u.x, w)) + ϵ),
        grid.Δymin / (maximum(abs, interior(u.y, w)) + ϵ),
        grid.Δz / (maximum(abs, interior(u.z, w)) + ϵ),
    )
    invΔy2 = setup.imexy ? zero(T) : 1 / grid.Δymin^2   # y-diffusion is implicit
    diff = 1 / (2 * visc * (1 / grid.Δx^2 + invΔy2 + 1 / grid.Δz^2) + ϵ)
    MPI.Allreduce(min(cfl * conv, cfl * diff), min, topo.cart)
end

"""
    solve!(; u, setup, tlims, Δt = nothing, cfl = 0.5, processors = (;))

Integrate the incompressible Navier-Stokes equations from `tlims[1]` to
`tlims[2]`, starting from (and mutating) the velocity field `u`. The initial
field is projected. `Δt = nothing` means CFL-adaptive stepping. Each
processor in the `processors` NamedTuple is called as `proc(state, setup)`
after every step (and once initially) with `state = (; u, p, t, n)`.
Returns the final state.
"""
function solve!(; u, setup, tlims, Δt = nothing, cfl = 0.5, processors = (;))
    p = scalarfield(setup)
    q = vectorfield(setup)
    δ = setup.imexy ? scalarfield(setup) : p   # projection increment
    exchange_halo!(u, setup)
    project!(u, δ, setup)
    t, tend = float.(tlims)
    n = 0
    state = (; u, p, t, n)
    foreach(proc -> proc(state, setup), processors)
    while t < tend - 1e-10 * (abs(tend) + 1)
        Δtn = min(something(Δt, cfl_timestep(u, setup; cfl)), tend - t)
        step!(u, p, q, δ, setup, Δtn)
        t += Δtn
        n += 1
        state = (; u, p, t, n)
        foreach(proc -> proc(state, setup), processors)
    end
    state
end
