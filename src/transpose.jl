# Global transposes (pencil redistributions). Both the source and
# destination distributions are block-contiguous in global index space, so
# the segment rank r sends to peer c is the intersection of r's source box
# with c's destination box — three range intersections, nothing more.

"Processor-grid axis over which data moves when going from `src` to `dst`."
function swapped_axis(src, dst)
    d = findfirst(==(0), src.axes) # src's local dimension
    a = dst.axes[d]
    @assert a != 0 && src.axes[findfirst(==(0), dst.axes)] == a "layouts not adjacent"
    a
end

intersect_box(b1, b2) = map(intersect, b1, b2)
localize(box, ranges) = map((b, r) -> b .- (first(r) - 1), box, ranges)
setcoord(coords, a, c) = ntuple(i -> i == a ? c : coords[i], length(coords))

"""
Precompute a transpose plan from layout `src` to layout `dst` (adjacent
orientations of the same global array). Send/receive boxes are in local
array indices; buffers are contiguous with peer segments in subcommunicator
rank order (= processor-grid coordinate order along the swapped axis).
With `stagehost = true` the plan carries host mirrors of the buffers and
the MPI call goes through them (for MPI libraries without device support).
"""
function plan_transpose(src, dst, topo, backend, T; stagehost = false)
    @assert src.gdims == dst.gdims
    a = swapped_axis(src, dst)
    (; procgrid, coords) = topo
    peers = 0:(procgrid[a]-1)
    sendboxes = map(peers) do c
        peer = layout(dst.gdims, dst.axes, procgrid, setcoord(coords, a, c))
        localize(intersect_box(src.ranges, peer.ranges), src.ranges)
    end
    recvboxes = map(peers) do c
        peer = layout(src.gdims, src.axes, procgrid, setcoord(coords, a, c))
        localize(intersect_box(dst.ranges, peer.ranges), dst.ranges)
    end
    sendcounts = map(b -> prod(length, b), sendboxes)
    recvcounts = map(b -> prod(length, b), recvboxes)
    (;
        subcomm = topo.subcomms[a],
        sendboxes,
        recvboxes,
        sendcounts,
        recvcounts,
        sendbuf = KernelAbstractions.allocate(backend, T, sum(sendcounts)),
        recvbuf = KernelAbstractions.allocate(backend, T, sum(recvcounts)),
        hostsendbuf = stagehost ? Vector{T}(undef, sum(sendcounts)) : nothing,
        hostrecvbuf = stagehost ? Vector{T}(undef, sum(recvcounts)) : nothing,
    )
end

"The plan for the opposite direction, sharing the forward plan's buffers."
reverse_plan(p) = (;
    p.subcomm,
    sendboxes = p.recvboxes,
    recvboxes = p.sendboxes,
    sendcounts = p.recvcounts,
    recvcounts = p.sendcounts,
    sendbuf = p.recvbuf,
    recvbuf = p.sendbuf,
    hostsendbuf = p.hostrecvbuf,
    hostrecvbuf = p.hostsendbuf,
)

"""
Redistribute `src` (in the plan's source layout) into `dst` (destination
layout). Backend synchronization around the MPI call is handled here — GPU
kernels are asynchronous, and MPI must not touch a buffer a kernel is still
writing (or read one a kernel is still reading).
"""
function transpose!(dst, src, plan, backend)
    if length(plan.sendcounts) == 1
        # Single rank along this axis: a device copy, no MPI (also makes
        # slab runs and single-GPU runs work without GPU-aware MPI).
        copybox!(dst, plan.recvboxes[1], src, plan.sendboxes[1], backend)
        KernelAbstractions.synchronize(backend)
        return
    end
    pack!(plan.sendbuf, src, plan.sendboxes, backend)
    KernelAbstractions.synchronize(backend)
    if plan.hostsendbuf === nothing
        MPI.Alltoallv!(
            MPI.VBuffer(plan.sendbuf, plan.sendcounts),
            MPI.VBuffer(plan.recvbuf, plan.recvcounts),
            plan.subcomm,
        )
    else
        copyto!(plan.hostsendbuf, plan.sendbuf)
        MPI.Alltoallv!(
            MPI.VBuffer(plan.hostsendbuf, plan.sendcounts),
            MPI.VBuffer(plan.hostrecvbuf, plan.recvcounts),
            plan.subcomm,
        )
        copyto!(plan.recvbuf, plan.hostrecvbuf)
    end
    unpack!(dst, plan.recvbuf, plan.recvboxes, backend)
    KernelAbstractions.synchronize(backend)
end
