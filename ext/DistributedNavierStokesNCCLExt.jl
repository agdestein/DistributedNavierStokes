# NCCL transport for the transpose layer (spectral solver): loaded
# automatically when CUDA.jl and NCCL.jl are both present. NCCL talks to
# the GPUs directly (no UCX), so it works on clusters whose MPI stack
# cannot handle CUDA.jl device buffers — the measured Snellius bottleneck
# (CAMPAIGN.md S0). One NCCL communicator per processor-grid axis, built
# once in `spectral_setup`; each transpose is one grouped send/recv round
# on the current CUDA stream, so it orders naturally with the pack/unpack
# kernels around it.
module DistributedNavierStokesNCCLExt

using CUDA
using NCCL
using MPI: MPI
using DistributedNavierStokes: DistributedNavierStokes
const DNS = DistributedNavierStokes

function DNS.nccl_subcomm(comm::MPI.Comm)
    nranks = MPI.Comm_size(comm)
    nranks == 1 && return nothing
    rank = MPI.Comm_rank(comm)
    # NCCL's out-of-band bootstrap: rank 0 mints the ID, MPI distributes it.
    unique_id = MPI.bcast(rank == 0 ? NCCL.UniqueID() : nothing, comm; root = 0)
    NCCL.Communicator(nranks, rank; unique_id)
end

function DNS.nccl_alltoallv!(
    comm::NCCL.Communicator,
    sendbuf::CuArray{<:Complex},
    sendcounts,
    recvbuf::CuArray{<:Complex},
    recvcounts,
)
    # NCCL has no complex datatype; counts are in elements of the real type.
    T = real(eltype(sendbuf))
    sb, rb = reinterpret(T, sendbuf), reinterpret(T, recvbuf)
    me = NCCL.rank(comm)
    stream = CUDA.stream()
    sd = rd = 0
    NCCL.groupStart()
    try
        for peer = 0:(NCCL.size(comm)-1)
            sc, rc = 2 * sendcounts[peer+1], 2 * recvcounts[peer+1]
            if peer == me
                # Self-segment: a stream-ordered device copy, not an NCCL op.
                copyto!(view(rb, (rd+1):(rd+rc)), view(sb, (sd+1):(sd+sc)))
            else
                sc > 0 && NCCL.Send(view(sb, (sd+1):(sd+sc)), comm; dest = peer, stream)
                rc > 0 && NCCL.Recv!(view(rb, (rd+1):(rd+rc)), comm; source = peer, stream)
            end
            sd += sc
            rd += rc
        end
    finally
        NCCL.groupEnd()
    end
    nothing
end

end
