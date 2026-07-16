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
