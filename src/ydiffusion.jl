# Semi-implicit wall-normal diffusion: per-RK-stage Crank-Nicolson in the
# Spalart-Moser-Rogers arrangement with incremental (lagged) pressure,
#
#     v  = uⁿ + B q - α Δt ∇π           (explicit increment + old pressure)
#     (I - θ Dy) u* = v + θ Dy uⁿ,      θ = ½ α Δt ν,
#
# which is Crank-Nicolson on (uⁿ + u*). Two structural points, both
# load-bearing: the CN average must involve the *old* stage velocity uⁿ
# (averaging the explicitly-updated field instead drops the coupling to
# first order), and the lagged pressure must enter *outside* the RK
# register with weight exactly α Δt (routing it through `q` overweights
# the pressure increments by B/α per stage and destabilizes the
# incremental projection). The stage projection afterwards solves only
# for the pressure increment, keeping the splitting error of the implicit
# treatment at second order.
#
# Dy is exactly the discrete y-diffusion stencil of operators.jl with the
# wall conditions folded into the matrix rows (so explicit and implicit
# formulations share one operator). The tridiagonal solves need y-local
# lines: fields are transposed to the y-local layout `axes = (2, 0, 1)`,
# which is adjacent to the native layout (an x ↔ y swap along processor
# axis 2 only) and degenerates to a device copy when p₂ = 1.

"""
Tridiagonal rows of the y-diffusion operator (without the ν factor), walls
folded in. Tangential components (cell centers): mirror ghost
`u₀ = -u₁` folds into the diagonal. Normal component (faces): wall values
are zero (Dirichlet); the wall-face row `ny` is zeroed so that stored zero
boundary values pass through the solve unchanged.
"""
function ydiffusion_coefficients(n, grid, w, T)
    (; Δye, Δyue) = grid
    ny = n[2]
    at, bt, ct = zeros(T, ny), zeros(T, ny), zeros(T, ny)
    for j = 1:ny
        a = 1 / (Δye[j+w] * Δyue[j-1+w])
        c = 1 / (Δye[j+w] * Δyue[j+w])
        at[j] = j == 1 ? zero(T) : a
        ct[j] = j == ny ? zero(T) : c
        bt[j] = j == 1 ? -(c + 2a) : j == ny ? -(a + 2c) : -(a + c)
    end
    an, bn, cn = zeros(T, ny), zeros(T, ny), zeros(T, ny)
    for j = 1:(ny-1)
        a = 1 / (Δyue[j+w] * Δye[j+w])
        c = 1 / (Δyue[j+w] * Δye[j+1+w])
        an[j] = j == 1 ? zero(T) : a
        cn[j] = j == ny - 1 ? zero(T) : c
        bn[j] = -(a + c)
    end
    (; at, bt, ct, an, bn, cn)
end

"Transpose plan, buffers, and coefficients for [`imex_update!`](@ref)."
function plan_ydiffusion(s)
    (; n, bc, T, backend, topo, lay) = s
    bc[2] == :wall || error("implicit y-diffusion requires bc[2] == :wall")
    yl = layout(n, (2, 0, 1), topo.procgrid, topo.coords)
    plan = plan_transpose(lay, yl, topo, backend, T; s.stagehost)
    (;
        yl,
        plan,
        rplan = reverse_plan(plan),
        ybuf = KernelAbstractions.allocate(backend, T, yl.ldims),
        ybuf2 = KernelAbstractions.allocate(backend, T, yl.ldims),
        ybuf3 = KernelAbstractions.allocate(backend, T, yl.ldims),
        map(v -> todevice(backend, v), ydiffusion_coefficients(n, s.grid, s.w, T))...,
    )
end

# rhs = v + θ Dy uⁿ
@kernel function imexrhs_kernel!(
    rhs,
    @Const(vt),
    @Const(ut),
    @Const(a),
    @Const(b),
    @Const(c),
    θ,
    ny,
)
    i, k = @index(Global, NTuple)
    @inbounds begin
        rhs[i, 1, k] = vt[i, 1, k] + θ * (b[1] * ut[i, 1, k] + c[1] * ut[i, 2, k])
        for j = 2:(ny-1)
            rhs[i, j, k] =
                vt[i, j, k] +
                θ * (a[j] * ut[i, j-1, k] + b[j] * ut[i, j, k] + c[j] * ut[i, j+1, k])
        end
        rhs[i, ny, k] = vt[i, ny, k] + θ * (a[ny] * ut[i, ny-1, k] + b[ny] * ut[i, ny, k])
    end
end

# Thomas algorithm for I - θ tridiag(a, b, c); `cp` is the c′ work array.
@kernel function imexsolve_kernel!(x, cp, @Const(a), @Const(b), @Const(c), θ, ny)
    i, k = @index(Global, NTuple)
    @inbounds begin
        β = 1 - θ * b[1]
        cp[i, 1, k] = -θ * c[1] / β
        x[i, 1, k] = x[i, 1, k] / β
        for j = 2:ny
            β = (1 - θ * b[j]) + θ * a[j] * cp[i, j-1, k]
            cp[i, j, k] = -θ * c[j] / β
            x[i, j, k] = (x[i, j, k] + θ * a[j] * x[i, j-1, k]) / β
        end
        for j = (ny-1):-1:1
            x[i, j, k] -= cp[i, j, k] * x[i, j+1, k]
        end
    end
end

"""
Semi-implicit stage update of the velocity (SMR delta form): per component,
transpose `uⁿ` to y-local lines, apply the explicit increment `+ B q` and
the lagged pressure gradient `- α Δt ∇p` in the native layout, transpose
the result, and solve `(I - θ Dy) u* = v + θ Dy uⁿ` with `θ = ½ α Δt ν`.
Wall conditions are in the matrix; halos are stale afterwards and must be
exchanged by the caller. Uses the Poisson real buffer as staging.
"""
function imex_update!(u, q, p, B, α, Δt, s)
    (; backend, lay, w, ranges, T) = s
    yd = s.ydiff
    stagebuf = s.poisson.r
    src = map(l -> (w+1):(w+l), lay.ldims)
    dst = map(l -> 1:l, lay.ldims)
    ny = size(yd.ybuf, 2)
    plane = (size(yd.ybuf, 1), size(yd.ybuf, 3))
    θ = T(α * Δt * s.visc / 2)
    for c in (:x, :y, :z)
        a, b, cc = c == :y ? (yd.an, yd.bn, yd.cn) : (yd.at, yd.bt, yd.ct)
        copybox!(stagebuf, dst, u[c], src, backend)
        transpose!(yd.ybuf, stagebuf, yd.plan, backend)      # uⁿ lines
        r = ranges[c]
        v = view(u[c], r...)
        @. v += B * $view(q[c], r...)
        pressuregrad_component!(u[c], p, c, T(α * Δt), s)
        copybox!(stagebuf, dst, u[c], src, backend)
        transpose!(yd.ybuf2, stagebuf, yd.plan, backend)     # v lines
        imexrhs_kernel!(backend)(
            yd.ybuf3,
            yd.ybuf2,
            yd.ybuf,
            a,
            b,
            cc,
            θ,
            ny;
            ndrange = plane,
        )
        imexsolve_kernel!(backend)(yd.ybuf3, yd.ybuf, a, b, cc, θ, ny; ndrange = plane)
        transpose!(stagebuf, yd.ybuf3, yd.rplan, backend)
        copybox!(u[c], src, stagebuf, dst, backend)
    end
    KernelAbstractions.synchronize(backend)
end
