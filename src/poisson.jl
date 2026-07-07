# Pressure Poisson solver: FFT diagonalization in the periodic directions
# and a direct solve in y, distributed via pencil transposes:
#
#   rfft(x) → transpose → fft(z) → transpose → solve(y) → back
#
# The y solve is a batched Thomas algorithm (walls, one thread per (kx, kz)
# mode) or a Fourier division (fully periodic, uniform y). The operator is
# exactly the discrete divergence∘gradient of operators.jl, so the projected
# velocity is divergence-free to machine precision.

"Modified wavenumber of the 2nd-order periodic Laplacian stencil."
modλ(k, n, Δ) = 2 * (cospi(2 * (k - 1) / n) - 1) / Δ^2

@kernel function thomas_kernel!(
    x,
    cp,
    @Const(ay),
    @Const(by),
    @Const(cy),
    @Const(λ1),
    @Const(λ3),
    ny,
)
    i, k = @index(Global, NTuple)
    @inbounds begin
        λ = λ1[i] + λ3[k]
        # The mean mode (λ == 0, all-Neumann) is singular: replace the first
        # row with x₁ = 0 to pin the pressure level.
        β = λ == 0 ? one(λ) : by[1] + λ
        cp[i, 1, k] = (λ == 0 ? zero(λ) : cy[1]) / β
        x[i, 1, k] = (λ == 0 ? zero(x[i, 1, k]) : x[i, 1, k]) / β
        for j = 2:ny
            β = (by[j] + λ) - ay[j] * cp[i, j-1, k]
            cp[i, j, k] = cy[j] / β
            x[i, j, k] = (x[i, j, k] - ay[j] * x[i, j-1, k]) / β
        end
        for j = (ny-1):-1:1
            x[i, j, k] -= cp[i, j, k] * x[i, j+1, k]
        end
    end
end

@kernel function modedivide_kernel!(x, @Const(λ1), @Const(λ2), @Const(λ3))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        λ = λ1[i] + λ2[j] + λ3[k]
        x[i, j, k] = λ == 0 ? zero(x[i, j, k]) : x[i, j, k] / λ
    end
end

"""
Plans, buffers, and coefficients for [`poisson!`](@ref). Two complex
pencil-sized buffers per orientation stage; the real buffer `r` is both the
input (fill it with the divergence) and the output stage of the pipeline.
"""
function plan_poisson(s)
    (; n, bc, T, backend, topo, lay, grid, w) = s
    r = KernelAbstractions.allocate(backend, T, lay.ldims)
    gc = (n[1] ÷ 2 + 1, n[2], n[3])
    lxc = layout(gc, AXES.x, topo.procgrid, topo.coords)
    lzc = layout(gc, AXES.z, topo.procgrid, topo.coords)
    lyc = layout(gc, AXES.y, topo.procgrid, topo.coords)
    âx = KernelAbstractions.allocate(backend, Complex{T}, lxc.ldims)
    âz = KernelAbstractions.allocate(backend, Complex{T}, lzc.ldims)
    ây = KernelAbstractions.allocate(backend, Complex{T}, lyc.ldims)
    xz = plan_transpose(lxc, lzc, topo, backend, Complex{T})
    zy = plan_transpose(lzc, lyc, topo, backend, Complex{T})
    base = (;
        r,
        âx,
        âz,
        ây,
        lxc,
        lzc,
        lyc,
        xz,
        zy,
        yz = reverse_plan(zy),
        zx = reverse_plan(xz),
        plan_x = plan_rfft(r, 1),
        iplan_x = plan_irfft(âx, n[1], 1),
        plan_z! = plan_fft!(âz, 3),
        iplan_z! = plan_ifft!(âz, 3),
        λ1 = todevice(backend, T[modλ(k, n[1], grid.Δx) for k in lyc.ranges[1]]),
        λ3 = todevice(backend, T[modλ(k, n[3], grid.Δz) for k in lyc.ranges[3]]),
    )
    if bc[2] == :wall
        # div∘grad in y with the wall fluxes dropped (staggered Neumann)
        (; Δye, Δyue) = grid
        ay = zeros(T, n[2])
        by = zeros(T, n[2])
        cy = zeros(T, n[2])
        for j = 1:n[2]
            a = j == 1 ? zero(T) : 1 / (Δye[j+w] * Δyue[j-1+w])
            c = j == n[2] ? zero(T) : 1 / (Δye[j+w] * Δyue[j+w])
            ay[j], cy[j], by[j] = a, c, -(a + c)
        end
        (;
            base...,
            ysolver = :thomas,
            ay = todevice(backend, ay),
            by = todevice(backend, by),
            cy = todevice(backend, cy),
            work = KernelAbstractions.allocate(backend, T, lyc.ldims),
        )
    else
        Δyint = @view grid.Δye[(w+1):(w+n[2])]
        maximum(Δyint) - minimum(Δyint) < 1e-12 * maximum(Δyint) ||
            error("periodic y requires a uniform grid (stretch = identity)")
        (;
            base...,
            ysolver = :fourier,
            # NOTE: cuFFT may not support transforms along the middle
            # dimension; if so, this stage needs a local permutation.
            plan_y! = plan_fft!(ây, 2),
            iplan_y! = plan_ifft!(ây, 2),
            λ2 = todevice(backend, T[modλ(k, n[2], grid.Δye[w+1]) for k in lyc.ranges[2]]),
        )
    end
end

function solve_y!(ps, s)
    if ps.ysolver === :thomas
        thomas_kernel!(s.backend)(
            ps.ây,
            ps.work,
            ps.ay,
            ps.by,
            ps.cy,
            ps.λ1,
            ps.λ3,
            s.n[2];
            ndrange = (size(ps.ây, 1), size(ps.ây, 3)),
        )
    else
        ps.plan_y! * ps.ây
        modedivide_kernel!(s.backend)(ps.ây, ps.λ1, ps.λ2, ps.λ3; ndrange = size(ps.ây))
        ps.iplan_y! * ps.ây
    end
end

"""
Solve the pressure Poisson equation with the divergence field currently in
`setup.poisson.r`, writing the (level-pinned) pressure into the interior of
the ghosted array `p`. Halos of `p` are *not* exchanged here.
"""
function poisson!(p, setup)
    ps = setup.poisson
    (; backend, lay, w) = setup
    mul!(ps.âx, ps.plan_x, ps.r)
    transpose!(ps.âz, ps.âx, ps.xz, backend)
    ps.plan_z! * ps.âz
    transpose!(ps.ây, ps.âz, ps.zy, backend)
    solve_y!(ps, setup)
    transpose!(ps.âz, ps.ây, ps.yz, backend)
    ps.iplan_z! * ps.âz
    transpose!(ps.âx, ps.âz, ps.zx, backend)
    mul!(ps.r, ps.iplan_x, ps.âx)
    copybox!(p, map(l -> (w+1):(w+l), lay.ldims), ps.r, map(l -> 1:l, lay.ldims), backend)
    KernelAbstractions.synchronize(backend)
end
