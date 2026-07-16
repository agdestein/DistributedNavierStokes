# Integer-wavenumber shell energies and shell-clamp forcing (maintain the
# initial energy in the lowest shells, as in SymmetryCode's
# `energy_shells` / `maintain_shell_energy!`). Shell membership is
# `s = floor(√(k²_int))`, i.e. s² ≤ |k_int|² < (s+1)², computed inline from
# the integer wavenumber vectors — no stored mode-shaped index arrays.

"""
Energy in each integer-wavenumber shell of `shells`, reduced over all
ranks. Uses the rfft double-counting weights; one pass over the state per
shell (fine for the few forced shells and setup-time use).
"""
function shell_energies(uh, s, shells)
    (; T, ikxr, ikyr, ikzr, w1r) = s
    esc = s.fft.escratch
    E = map(collect(shells)) do sh
        @views @. esc = ifelse(
            floor(Int32, sqrt(T(ikxr^2 + ikyr^2 + ikzr^2))) == Int32(sh),
            w1r * (abs2(uh[:, :, :, 1]) + abs2(uh[:, :, :, 2]) + abs2(uh[:, :, :, 3])) / 2,
            zero(T),
        )
        sum(esc)
    end
    MPI.Allreduce(E, +, s.topo.cart)
end

"Multiply all modes of shell `sh` by `factor`."
function rescale_shell!(uh, s, sh, factor)
    (; T, ikxr, ikyr, ikzr) = s
    @. uh *= ifelse(
        floor(Int32, sqrt(T(ikxr^2 + ikyr^2 + ikzr^2))) == Int32(sh),
        T(factor),
        one(T),
    )
    uh
end

"""
    forcing = shellforcing(uh, s; shells = 1:2)

Shell-clamp forcing: record the current (initial) energy in the given
shells; the returned closure `forcing(uh, s)` rescales those shells back to
their reference energy. Pass it to `spectral_solve!(; forcing)`.
"""
function shellforcing(uh, s; shells = 1:2)
    eref = shell_energies(uh, s, shells)
    (uh2, s2) -> begin
        e = shell_energies(uh2, s2, shells)
        for (sh, e0, e1) in zip(shells, eref, e)
            rescale_shell!(uh2, s2, sh, e1 > 0 ? sqrt(e0 / e1) : one(s2.T))
        end
        nothing
    end
end
