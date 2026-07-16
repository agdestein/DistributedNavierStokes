# HIT diagnostics on the truncated spectral state: K41 turbulence
# statistics (ported from SymmetryCode's `turbulence_statistics`) and the
# shell-binned energy spectrum. All reductions are local sums + Allreduce.

"""
3D homogeneous-isotropic-turbulence statistics (Kolmogorov K41 scaling),
reduced over all ranks. Matches SymmetryCode's `turbulence_statistics`
field for field, including the `kmax_eta = (πn/l)·η` resolution check with
the transform-grid Nyquist convention.
"""
function spectral_stats(uh, s)
    (; visc, n, l) = s
    e = spectral_energy(uh, s)
    diss = spectral_dissipation(uh, s)
    uavg = sqrt(2 * e / 3)
    l_kol = (visc^3 / diss)^(1 / 4)
    l_tay = sqrt(15 * visc / diss) * uavg
    l_int = uavg^3 / diss
    t_int = l_int / uavg
    t_tay = l_tay / uavg
    t_kol = sqrt(visc / diss)
    Re_int = l_int * uavg / visc
    Re_tay = l_tay * uavg / visc
    kmax_eta = (π * n / l) * l_kol
    (; e, uavg, diss, l_int, l_tay, l_kol, t_int, t_tay, t_kol, Re_int, Re_tay, kmax_eta)
end

"""
Shell-binned kinetic energy spectrum `(; κ, E)` over the complete shells
`κ = 0:kcut`, reduced over all ranks. `sum(E) = ⟨|u|²⟩/2` up to the corner
modes outside the largest complete shell.
"""
function spectral_spectrum(uh, s)
    E = shell_energies(uh, s, 0:s.kcut)
    (; κ = 0:s.kcut, E)
end
