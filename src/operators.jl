# Symmetry-preserving second-order operators on the staggered grid.
# Convention: `u.x[I, J, K]` lives on the right x-face of cell `(I, J, K)`,
# `u.y` on the top y-face, `u.z` on the back z-face; the divergence is a
# backward difference. Convective fluxes use mass-flux-averaged transporting
# velocities (Δy-weighted where a staggered volume straddles two cells of
# different height) and simple-averaged transported velocities, which makes
# convection skew-symmetric w.r.t. the volume-weighted inner product, also
# on stretched y grids.

@kernel function divergence_kernel!(
    r,
    @Const(ux),
    @Const(uy),
    @Const(uz),
    Δx,
    Δz,
    @Const(Δy),
    w,
)
    i, j, k = @index(Global, NTuple)
    I, J, K = i + w, j + w, k + w
    @inbounds r[i, j, k] =
        (ux[I, J, K] - ux[I-1, J, K]) / Δx +
        (uy[I, J, K] - uy[I, J-1, K]) / Δy[J] +
        (uz[I, J, K] - uz[I, J, K-1]) / Δz
end

"Divergence of `u` into the ghost-free array `r` (the Poisson input buffer)."
function divergence!(r, u, setup)
    (; grid, w, backend) = setup
    divergence_kernel!(backend)(
        r,
        u.x,
        u.y,
        u.z,
        grid.Δx,
        grid.Δz,
        grid.Δy,
        w;
        ndrange = size(r),
    )
end

@kernel function momentum_kernel!(
    qx,
    qy,
    qz,
    @Const(ux),
    @Const(uy),
    @Const(uz),
    A,
    Δt,
    ν,
    Δx,
    Δz,
    @Const(Δy),
    @Const(Δyu),
    fx,
    fy,
    fz,
    o1,
    o2,
    o3,
    jymax,
)
    i, j, k = @index(Global, NTuple)
    I, J, K = i + o1, j + o2, k + o3
    @inbounds begin
        # x momentum, at the right x-face of cell (I, J, K)
        uce = (ux[I, J, K] + ux[I+1, J, K]) / 2   # u at center of cell I+1
        ucw = (ux[I-1, J, K] + ux[I, J, K]) / 2   # u at center of cell I
        vn = (uy[I, J, K] + uy[I+1, J, K]) / 2    # v on the north edge
        un = (ux[I, J, K] + ux[I, J+1, K]) / 2
        vs = (uy[I, J-1, K] + uy[I+1, J-1, K]) / 2
        us = (ux[I, J-1, K] + ux[I, J, K]) / 2
        wt = (uz[I, J, K] + uz[I+1, J, K]) / 2    # w on the top edge
        ut = (ux[I, J, K] + ux[I, J, K+1]) / 2
        wb = (uz[I, J, K-1] + uz[I+1, J, K-1]) / 2
        ub = (ux[I, J, K-1] + ux[I, J, K]) / 2
        conv =
            (uce * uce - ucw * ucw) / Δx +
            (vn * un - vs * us) / Δy[J] +
            (wt * ut - wb * ub) / Δz
        diff =
            (ux[I+1, J, K] - 2 * ux[I, J, K] + ux[I-1, J, K]) / Δx^2 +
            (
                (ux[I, J+1, K] - ux[I, J, K]) / Δyu[J] -
                (ux[I, J, K] - ux[I, J-1, K]) / Δyu[J-1]
            ) / Δy[J] +
            (ux[I, J, K+1] - 2 * ux[I, J, K] + ux[I, J, K-1]) / Δz^2
        qx[I, J, K] = A * qx[I, J, K] + Δt * (-conv + ν * diff + fx)

        # y momentum, at the top y-face of cell (I, J, K); the staggered
        # volume straddles cells J and J+1 → Δy-weighted transporters
        ue = (ux[I, J, K] * Δy[J] + ux[I, J+1, K] * Δy[J+1]) / (2 * Δyu[J])
        ve = (uy[I, J, K] + uy[I+1, J, K]) / 2
        uw = (ux[I-1, J, K] * Δy[J] + ux[I-1, J+1, K] * Δy[J+1]) / (2 * Δyu[J])
        vw = (uy[I-1, J, K] + uy[I, J, K]) / 2
        vcn = (uy[I, J, K] + uy[I, J+1, K]) / 2   # v at center of cell J+1
        vcs = (uy[I, J-1, K] + uy[I, J, K]) / 2   # v at center of cell J
        wtv = (uz[I, J, K] * Δy[J] + uz[I, J+1, K] * Δy[J+1]) / (2 * Δyu[J])
        vt = (uy[I, J, K] + uy[I, J, K+1]) / 2
        wbv = (uz[I, J, K-1] * Δy[J] + uz[I, J+1, K-1] * Δy[J+1]) / (2 * Δyu[J])
        vb = (uy[I, J, K-1] + uy[I, J, K]) / 2
        conv =
            (ue * ve - uw * vw) / Δx +
            (vcn * vcn - vcs * vcs) / Δyu[J] +
            (wtv * vt - wbv * vb) / Δz
        diff =
            (uy[I+1, J, K] - 2 * uy[I, J, K] + uy[I-1, J, K]) / Δx^2 +
            (
                (uy[I, J+1, K] - uy[I, J, K]) / Δy[J+1] -
                (uy[I, J, K] - uy[I, J-1, K]) / Δy[J]
            ) / Δyu[J] +
            (uy[I, J, K+1] - 2 * uy[I, J, K] + uy[I, J, K-1]) / Δz^2
        if J <= jymax
            qy[I, J, K] = A * qy[I, J, K] + Δt * (-conv + ν * diff + fy)
        end

        # z momentum, at the back z-face of cell (I, J, K)
        ue = (ux[I, J, K] + ux[I, J, K+1]) / 2
        we = (uz[I, J, K] + uz[I+1, J, K]) / 2
        uw = (ux[I-1, J, K] + ux[I-1, J, K+1]) / 2
        ww = (uz[I-1, J, K] + uz[I, J, K]) / 2
        vnw = (uy[I, J, K] + uy[I, J, K+1]) / 2
        wn = (uz[I, J, K] + uz[I, J+1, K]) / 2
        vsw = (uy[I, J-1, K] + uy[I, J-1, K+1]) / 2
        ws = (uz[I, J-1, K] + uz[I, J, K]) / 2
        wct = (uz[I, J, K] + uz[I, J, K+1]) / 2   # w at center of cell K+1
        wcb = (uz[I, J, K-1] + uz[I, J, K]) / 2   # w at center of cell K
        conv =
            (ue * we - uw * ww) / Δx +
            (vnw * wn - vsw * ws) / Δy[J] +
            (wct * wct - wcb * wcb) / Δz
        diff =
            (uz[I+1, J, K] - 2 * uz[I, J, K] + uz[I-1, J, K]) / Δx^2 +
            (
                (uz[I, J+1, K] - uz[I, J, K]) / Δyu[J] -
                (uz[I, J, K] - uz[I, J-1, K]) / Δyu[J-1]
            ) / Δy[J] +
            (uz[I, J, K+1] - 2 * uz[I, J, K] + uz[I, J, K-1]) / Δz^2
        qz[I, J, K] = A * qz[I, J, K] + Δt * (-conv + ν * diff + fz)
    end
end

"""
Runge-Kutta register update `q ← A q + Δt F(u)` with the full momentum
right-hand side `F` (convection + diffusion + body force), fused into one
kernel over all three components. `u` must have valid halos. The x/z
components are updated on the cell range (faces ≡ cells in periodic
directions); the inactive top wall face of `u.y` is skipped via `jymax`.
"""
function momentum!(q, u, A, Δt, setup)
    (; grid, visc, bodyforce, ranges, backend) = setup
    rng = ranges.p
    momentum_kernel!(backend)(
        q.x,
        q.y,
        q.z,
        u.x,
        u.y,
        u.z,
        A,
        Δt,
        visc,
        grid.Δx,
        grid.Δz,
        grid.Δy,
        grid.Δyu,
        bodyforce...,
        map(r -> first(r) - 1, rng)...,
        last(ranges.y[2]);
        ndrange = map(length, rng),
    )
end

@kernel function pgradx_kernel!(ux, @Const(p), Δx, o1, o2, o3)
    i, j, k = @index(Global, NTuple)
    I, J, K = i + o1, j + o2, k + o3
    @inbounds ux[I, J, K] -= (p[I+1, J, K] - p[I, J, K]) / Δx
end

@kernel function pgrady_kernel!(uy, @Const(p), @Const(Δyu), o1, o2, o3)
    i, j, k = @index(Global, NTuple)
    I, J, K = i + o1, j + o2, k + o3
    @inbounds uy[I, J, K] -= (p[I, J+1, K] - p[I, J, K]) / Δyu[J]
end

@kernel function pgradz_kernel!(uz, @Const(p), Δz, o1, o2, o3)
    i, j, k = @index(Global, NTuple)
    I, J, K = i + o1, j + o2, k + o3
    @inbounds uz[I, J, K] -= (p[I, J, K+1] - p[I, J, K]) / Δz
end

"""
Subtract the staggered pressure gradient from `u` on active DOFs only —
wall faces are boundary values and keep `u.y = 0`, which is exactly the
homogeneous Neumann condition built into the Poisson operator. `p` must
have valid halos.
"""
function pressuregrad_update!(u, p, setup)
    (; grid, ranges, backend) = setup
    launch(kernel!, φ, extra, rng) = kernel!(backend)(
        φ,
        p,
        extra,
        map(r -> first(r) - 1, rng)...;
        ndrange = map(length, rng),
    )
    launch(pgradx_kernel!, u.x, grid.Δx, ranges.x)
    launch(pgrady_kernel!, u.y, grid.Δyu, ranges.y)
    launch(pgradz_kernel!, u.z, grid.Δz, ranges.z)
end
