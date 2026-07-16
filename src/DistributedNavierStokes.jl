"""
Distributed multi-GPU solver for the incompressible Navier-Stokes equations
on staggered Cartesian grids: symmetry-preserving finite volumes, pencil
decomposition with FFT/tridiagonal pressure solves, explicit low-storage
Runge-Kutta time stepping. See DESIGN.md and CODE_DESIGN.md.
"""
module DistributedNavierStokes

using AbstractFFTs: plan_rfft, plan_irfft, plan_brfft, plan_fft!, plan_ifft!, plan_bfft!
using FFTW: FFTW
using KernelAbstractions
using LinearAlgebra: mul!
using MPI: MPI
using Printf: @sprintf

include("layout.jl")
include("pack.jl")
include("transpose.jl")
include("halo.jl")
include("poisson.jl")
include("ydiffusion.jl")
include("operators.jl")
include("timestep.jl")
include("setup.jl")
include("fields.jl")
include("stats.jl")
include("spectral/fft.jl")
include("spectral/operators.jl")
include("spectral/step.jl")
include("spectral/setup.jl")
include("spectral/ic.jl")

export setup, solve!, exchange_halo!
export scalarfield, vectorfield, velocityfield!, project!, maxdiv
export channelstats, channelprofiles, spectrumstats, energyspectrum
export spectral_setup, spectral_solve!, specvelocity, spectral_velocityfield!,
    taylorgreen!, spectral_project!, spectral_energy, spectral_dissipation,
    spectral_maxdiv, spec_to_phys!, phys_to_spec!

end
