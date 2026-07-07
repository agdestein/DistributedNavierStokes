# Field allocation, initial conditions, and rank-reduced diagnostics.

"Ghosted scalar field (zeros) in the native layout."
scalarfield(setup) =
    KernelAbstractions.zeros(setup.backend, setup.T, setup.lay.ldims .+ 2 * setup.w)

"Ghosted velocity field `(; x, y, z)` (zeros)."
vectorfield(setup) =
    (; x = scalarfield(setup), y = scalarfield(setup), z = scalarfield(setup))

function fillcomponent!(φ, f, setup, coords, rng)
    isnothing(f) && return
    (; T, w) = setup
    h = zeros(T, size(φ))
    for k in rng[3], j in rng[2], i in rng[1]
        h[i, j, k] = f(coords[1][i-w], coords[2][j-w], coords[3][k-w])
    end
    copyto!(φ, h)
end

"""
    velocityfield!(u, setup; x = nothing, y = nothing, z = nothing)

Set velocity components from functions of the physical coordinates,
evaluated at each component's staggered location (active DOFs only — wall
faces stay zero). Halos are exchanged afterwards.
"""
function velocityfield!(u, setup; x = nothing, y = nothing, z = nothing)
    g = setup.grid
    fillcomponent!(u.x, x, setup, (g.xf, g.yc, g.zc), setup.ranges.x)
    fillcomponent!(u.y, y, setup, (g.xc, g.yf, g.zc), setup.ranges.y)
    fillcomponent!(u.z, z, setup, (g.xc, g.yc, g.zf), setup.ranges.z)
    exchange_halo!(u, setup)
    u
end

"Maximum absolute divergence over all ranks."
function maxdiv(u, setup)
    divergence!(setup.poisson.r, u, setup)
    KernelAbstractions.synchronize(setup.backend)
    MPI.Allreduce(maximum(abs, setup.poisson.r), max, setup.topo.cart)
end

"Volume-weighted total kinetic energy `½ ∫ |u|² dV` over all ranks."
function kinetic_energy(u, setup)
    (; grid, w, topo, ranges) = setup
    e = zero(setup.T)
    # host reduction; diagnostics only
    Δyh = Array(grid.Δy)
    Δyuh = Array(grid.Δyu)
    for (c, φ) in pairs(u)
        h = Array(φ)
        Δ2 = c == :y ? Δyuh : Δyh
        r = ranges[c]
        for k in r[3], j in r[2], i in r[1]
            e += h[i, j, k]^2 * grid.Δx * Δ2[j] * grid.Δz / 2
        end
    end
    MPI.Allreduce(e, +, topo.cart)
end

"Processor printing time-step info on rank 0 every `nupdate` steps."
logger(; nupdate = 1) =
    (state, setup) -> begin
        state.n % nupdate == 0 || return
        setup.topo.rank == 0 &&
            println(@sprintf("n = %6d  t = %10.6g", state.n, state.t))
    end
