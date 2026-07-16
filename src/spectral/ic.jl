# Initial conditions. Fields are always specified in physical space on the
# transform grid (cell centers, matching SymmetryCode) and forward
# transformed — the truncation is then automatic and Hermitian symmetry
# never needs distributed repair.

"""
    spectral_velocityfield!(uh, s; x, y, z, project = true)

Set the spectral state from three functions of the physical cell-center
coordinates, evaluated on the local block and forward transformed (hence
dealiased). Projects onto divergence-free fields unless `project = false`.
"""
function spectral_velocityfield!(uh, s; x, y, z, project = true)
    (; T, l, n, lphys) = s
    h = zeros(T, size(s.fft.rbuf))
    coord(gi) = l / n * (gi - T(1) / 2)
    for (c, f) in enumerate((x, y, z)), K in axes(h, 3), J in axes(h, 2), I in axes(h, 1)
        h[I, J, K, c] =
            f(coord(lphys.ranges[1][I]), coord(lphys.ranges[2][J]), coord(lphys.ranges[3][K]))
    end
    copyto!(s.fft.rbuf, h)
    phys_to_spec!(uh, s.fft.rbuf, s)
    project && spectral_project!(uh, s)
    uh
end

# Counter-based noise: a hash of (seed, global index) per point, so the
# initial condition is identical for every rank count and processor grid —
# decomposition invariance holds from t = 0 (no stateful RNG streams).

@inline function splitmix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

@kernel function noise_kernel!(v, seed, o1, o2, o3, n)
    i, j, k, c = @index(Global, NTuple)
    T = eltype(v)
    key =
        ((UInt64(c - 1) * n + (o3 + k - 1)) * n + (o2 + j - 1)) * n + (o1 + i - 1)
    r1 = splitmix64(seed + 2 * key)
    r2 = splitmix64(seed + 2 * key + 1)
    u1 = ((r1 >> 11) + T(0.5)) * T(0x1p-53)   # uniform in (0, 1)
    u2 = ((r2 >> 11) + T(0.5)) * T(0x1p-53)
    @inbounds v[i, j, k, c] = sqrt(-2 * log(u1)) * cospi(2 * u2)
end

"""
    spectral_randomfield!(uh, s; totalenergy = 1, kpeak = 4, seed = 0, profile)

Random solenoidal velocity field with prescribed energy-spectrum profile
(default `k⁴ exp(-2(k/kpeak)²)`), following SymmetryCode's `randomfield`:
white noise (here: counter-based, in physical space — Hermitian symmetry is
automatic and the field is decomposition-invariant), projected, then each
integer-wavenumber shell rescaled to the profile.
"""
function spectral_randomfield!(
    uh,
    s;
    totalenergy = 1,
    kpeak = 4,
    seed = 0,
    profile = k -> k^4 * exp(-2 * (k / kpeak)^2),
)
    (; backend, lphys, kcut) = s
    o = map(r -> first(r) - 1, lphys.ranges)
    noise_kernel!(backend)(
        s.fft.rbuf, UInt64(seed), o..., UInt64(s.n);
        ndrange = size(s.fft.rbuf),
    )
    KernelAbstractions.synchronize(backend)
    phys_to_spec!(uh, s.fft.rbuf, s)
    spectral_project!(uh, s)
    kdiag = floor(Int, sqrt(3) * kcut)
    E = shell_energies(uh, s, 0:kdiag)
    totalprofile = sum(k -> profile(k), 0:kdiag)
    for (sh, e) in zip(0:kdiag, E)
        e0 = totalenergy * profile(sh) / totalprofile
        rescale_shell!(uh, s, sh, e > 0 ? sqrt(e0 / e) : zero(s.T))
    end
    uh
end

"""
Taylor-Green vortex with peak velocity `V0` (canonical 3D form; `Re = V0/ν`
for `l = 2π`).
"""
taylorgreen!(uh, s; V0 = 1) = spectral_velocityfield!(
    uh,
    s;
    x = (x, y, z) -> V0 * sinpi(2x / s.l) * cospi(2y / s.l) * cospi(2z / s.l),
    y = (x, y, z) -> -V0 * cospi(2x / s.l) * sinpi(2y / s.l) * cospi(2z / s.l),
    z = (x, y, z) -> zero(V0 * x),
)
