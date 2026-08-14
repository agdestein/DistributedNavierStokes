# Stub NCCL_jll for clusters where CUDA.jl uses a local toolkit
# (`local_toolkit = true`): the real NCCL_jll then refuses to select any
# artifact by policy (a local toolkit is expected to bring a local NCCL),
# leaving `libnccl` undefined. This stub resolves the cluster's own NCCL
# (e.g. the Snellius `NCCL/2.26.6-...-CUDA-12.8.0` module, on
# LD_LIBRARY_PATH via `module load`) instead. NCCL.jl only consumes
# `libnccl` and `is_available` from the JLL, and the base NCCL ABI
# (init/send/recv/collectives) is stable across the 2.x series, so the
# version here only needs to satisfy NCCL.jl's compat bound.
#
# Machine-local: `Pkg.develop(path = "examples/snellius/NCCL_jll")` in the
# examples environment on the cluster only (Manifests are gitignored).
# Override the library path with $JULIA_NCCL_PATH if it is not on the
# loader path.
module NCCL_jll

libnccl::String = "libnccl.so.2"
__init__() = (global libnccl = get(ENV, "JULIA_NCCL_PATH", "libnccl.so.2"))
is_available() = true

export libnccl

end
