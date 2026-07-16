# Pointwise spectral-space operators on the truncated state (y-pencil,
# gdims (kcut+1, m, m), local (l1, m, l3, 3)). Wavenumbers come from the
# state layout's global ranges as 1D device vectors; everything here is
# k-local (no communication).

"Physical wavenumber of compact (odd-size-fftfreq) index `j` on an `m` grid."
compactfreq(j, m, kcut) = j ≤ kcut + 1 ? j - 1 : j - 1 - m

"Make the spectral velocity solenoidal (leaves the k = 0 mode intact)."
@kernel function project_kernel!(uh, @Const(kx), @Const(ky), @Const(kz))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        k1, k2, k3 = kx[i], ky[j], kz[k]
        kk = k1 * k1 + k2 * k2 + k3 * k3
        ux, uy, uz = uh[i, j, k, 1], uh[i, j, k, 2], uh[i, j, k, 3]
        p = (k1 * ux + k2 * uy + k3 * uz) / kk
        p = ifelse(kk == 0, zero(p), p)
        uh[i, j, k, 1] = ux - k1 * p
        uh[i, j, k, 2] = uy - k2 * p
        uh[i, j, k, 3] = uz - k3 * p
    end
end

"Project the spectral velocity `uh` onto divergence-free fields."
function spectral_project!(uh, s)
    project_kernel!(s.backend)(uh, s.kx, s.ky, s.kz; ndrange = size(uh)[1:3])
    KernelAbstractions.synchronize(s.backend)
    uh
end

# The nonlinear term -ik_j σ̂_ij accumulates into `du` in two batches of
# transformed products: the diagonal (σxx, σyy, σzz) initializes, the
# off-diagonal (σxy, σyz, σzx) accumulates (each feeds two components).

@kernel function divdiag_kernel!(du, @Const(σh), @Const(kx), @Const(ky), @Const(kz))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        du[i, j, k, 1] = -im * kx[i] * σh[i, j, k, 1]
        du[i, j, k, 2] = -im * ky[j] * σh[i, j, k, 2]
        du[i, j, k, 3] = -im * kz[k] * σh[i, j, k, 3]
    end
end

@kernel function divoffdiag_kernel!(du, @Const(σh), @Const(kx), @Const(ky), @Const(kz))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        σxy, σyz, σzx = σh[i, j, k, 1], σh[i, j, k, 2], σh[i, j, k, 3]
        du[i, j, k, 1] -= im * (ky[j] * σxy + kz[k] * σzx)
        du[i, j, k, 2] -= im * (kx[i] * σxy + kz[k] * σyz)
        du[i, j, k, 3] -= im * (kx[i] * σzx + ky[j] * σyz)
    end
end

"""
Convective right-hand side `du = -ik_j F[v_i v_j]` (divergence form,
dealiased by construction — the state only holds retained modes). Viscosity
is not included (integrating factor, see `spectral_step!`). The physical
velocities remain in `s.fft.v` afterwards (used for the CFL estimate).
"""
function spectral_rhs!(du, uh, s)
    (; backend) = s
    f = s.fft
    spec_to_phys!(f.v, uh, s)
    ndr = size(uh)[1:3]
    # diagonal products
    @. f.rbuf = f.v * f.v
    phys_to_spec!(f.σh, f.rbuf, s)
    divdiag_kernel!(backend)(du, f.σh, s.kx, s.ky, s.kz; ndrange = ndr)
    # off-diagonal products
    @views begin
        @. f.rbuf[:, :, :, 1] = f.v[:, :, :, 1] * f.v[:, :, :, 2]   # xy
        @. f.rbuf[:, :, :, 2] = f.v[:, :, :, 2] * f.v[:, :, :, 3]   # yz
        @. f.rbuf[:, :, :, 3] = f.v[:, :, :, 3] * f.v[:, :, :, 1]   # zx
    end
    phys_to_spec!(f.σh, f.rbuf, s)
    divoffdiag_kernel!(backend)(du, f.σh, s.kx, s.ky, s.kz; ndrange = ndr)
    KernelAbstractions.synchronize(backend)
    du
end

# Diagnostics. The rfft axis stores only kx ≥ 0: modes with kx > 0 count
# twice (the retained band kx ≤ kcut < n/2 contains no Nyquist plane, so
# every kx > 0 row doubles). `s.dbl` is the local row range with global
# kx > 0.

"Rank-local Σ f over retained modes with conjugate double-counting."
spectralsum(f, a, s) = sum(f, a; init = zero(s.T)) +
                       sum(f, view(a, s.dbl, ntuple(Returns(:), ndims(a) - 1)...); init = zero(s.T))

"Mean kinetic energy `⟨|u|²⟩/2` (Parseval), reduced over all ranks."
spectral_energy(uh, s) =
    MPI.Allreduce(spectralsum(abs2, uh, s) / 2, +, s.topo.cart)

"Dissipation rate `ε = ν Σ k²|û|²`, reduced over all ranks."
function spectral_dissipation(uh, s)
    (; kxr, kyr, kzr) = s
    @views @. s.fft.escratch =
        (abs2(uh[:, :, :, 1]) + abs2(uh[:, :, :, 2]) + abs2(uh[:, :, :, 3])) *
        (kxr^2 + kyr^2 + kzr^2)
    MPI.Allreduce(s.visc * spectralsum(identity, s.fft.escratch, s), +, s.topo.cart)
end

"Maximum `|ik ⋅ û|` over all retained modes and ranks."
function spectral_maxdiv(uh, s)
    (; kxr, kyr, kzr) = s
    @views @. s.fft.escratch = abs(
        kxr * uh[:, :, :, 1] + kyr * uh[:, :, :, 2] + kzr * uh[:, :, :, 3],
    )
    MPI.Allreduce(maximum(s.fft.escratch), max, s.topo.cart)
end
