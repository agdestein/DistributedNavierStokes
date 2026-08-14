# CUDA-specific memory reclamation for the spectral FFT pipeline
# (loaded automatically with CUDA.jl). CUDA.jl's out-of-place real-FFT
# plans each carry a hidden field-sized buffer: plan_rfft's serves only
# `ldiv!` (never called on device plans here), and plan_brfft's protects
# the C2R input from cuFFT's destructive out-of-place transform — but the
# pipeline's backward input is scratch by design. Freeing both saves ~6
# field equivalents; executing the backward transform directly also skips
# a full-field device copy per call. `CUDA.unsafe_free!` is idempotent,
# so the plan finalizer freeing the buffer again is harmless.
module DistributedNavierStokesCUDAExt

using CUDA
using CUDA.CUFFT: CuFFTPlan, unsafe_execute_trailing!
using DistributedNavierStokes: DistributedNavierStokes
const DNS = DistributedNavierStokes

function DNS.trim_plan!(p::CuFFTPlan)
    p.buffer === nothing || CUDA.unsafe_free!(p.buffer)
    p
end

function DNS.brfft_scratch!(y::CuArray, plan::CuFFTPlan, x::CuArray)
    unsafe_execute_trailing!(plan, x, y)
    y
end

end
