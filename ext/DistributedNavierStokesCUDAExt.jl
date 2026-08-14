# CUDA-specific memory reclamation for the spectral FFT pipeline
# (loaded automatically with CUDA.jl). CUDA.jl's out-of-place real-FFT
# plans each carry a hidden field-sized buffer: plan_rfft's serves only
# `ldiv!` (never called on device plans here), and plan_brfft's protects
# the C2R input from cuFFT's destructive out-of-place transform — but the
# pipeline's backward input is scratch by design. Freeing both saves ~6
# field equivalents; executing the backward transform directly also skips
# a full-field device copy per call. `unsafe_free!` is idempotent, so the
# plan finalizer freeing the buffer again is harmless.
#
# The direct executor is internal API whose name moved across CUDA.jl
# versions (CUFFT submodule `unsafe_execute_trailing!` → cuFFT package
# `unsafe_execute_external_batches!`), so both are probed at load time.
# Without either, the C2R plan keeps its buffer and `mul!` stays on the
# (copying) public path — correct, just not as lean.
module DistributedNavierStokesCUDAExt

using CUDA
using CUDA.CUFFT: CuFFTPlan
using LinearAlgebra: mul!
using DistributedNavierStokes: DistributedNavierStokes
const DNS = DistributedNavierStokes

const CUFFTMOD = parentmodule(CuFFTPlan)
const direct_exec =
    isdefined(CUFFTMOD, :unsafe_execute_external_batches!) ?
    CUFFTMOD.unsafe_execute_external_batches! :
    isdefined(CUFFTMOD, :unsafe_execute_trailing!) ? CUFFTMOD.unsafe_execute_trailing! :
    nothing

# Output eltype T <: Real marks the C2R (brfft) plan, whose buffer is
# load-bearing for `mul!`; only trim it when the direct path replaces it.
function DNS.trim_plan!(p::CuFFTPlan{T}) where {T}
    if p.buffer !== nothing && (T <: Complex || direct_exec !== nothing)
        CUDA.unsafe_free!(p.buffer)
    end
    p
end

function DNS.brfft_scratch!(y::CuArray, plan::CuFFTPlan, x::CuArray)
    if direct_exec === nothing
        mul!(y, plan, x)
    else
        direct_exec(plan, x, y)
    end
    y
end

end
