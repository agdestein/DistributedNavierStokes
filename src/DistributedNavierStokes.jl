"""
Distributed multi-GPU solver for the incompressible Navier-Stokes equations
on staggered Cartesian grids: symmetry-preserving finite volumes, pencil
decomposition with FFT/tridiagonal pressure solves, explicit low-storage
Runge-Kutta time stepping. See DESIGN.md and CODE_DESIGN.md.
"""
module DistributedNavierStokes

using AbstractFFTs: plan_rfft, plan_irfft, plan_fft!, plan_ifft!
using FFTW: FFTW
using KernelAbstractions
using LinearAlgebra: mul!
using MPI: MPI
using Printf: @sprintf

include("layout.jl")
include("pack.jl")
include("transpose.jl")

end
