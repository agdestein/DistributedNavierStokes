# Distributed 3D real FFTs for the pseudo-spectral solver, made
# truncation-aware: modes killed by the 2/3 rule are never packed, sent, or
# stored. The spectral state lives on the retained-mode grid
# (kcut+1, m, m), m = 2kcut+1, of the transform grid n³ — the rfft axis
# keeps kx = 0..kcut, the full axes keep |k| ≤ kcut in "compact" (wrapped,
# odd-size-fftfreq) order. Pipeline stages (forward):
#
#   x-pencil:  rfft(x), keep rows 1..kcut+1
#   transpose x→z              (gdims (kcut+1, n, n))
#   z-pencil:  fft(z), keep fine kz positions {1..kcut+1, n-kcut+1..n}
#   transpose z→y              (gdims (kcut+1, n, m), fine-split boxes)
#   y-pencil:  fft(y), truncate dim 2 to compact m (the state)
#
# Truncated dims travel in compact order; the only place compact and fine
# (physical FFT-array) positions meet inside a transpose is dim 3 (kz) of
# the z-side buffer, handled by splitting each per-peer box at the
# positive/negative frequency boundary (`finesplit3`) — the pack/unpack
# machinery takes explicit box lists, so no compaction copies are needed.
#
# Batching: the three components of a field travel in a single Alltoallv.
# Fields are 4D arrays `(l1, l2, l3, 3)`; a 4D array reshaped to 3D merges
# the batch into dim 3 (column-major), so batched plans are the per-field
# boxes replicated with dim-3 shifts, peer-major.

"Merge the batch dimension of a 4D array into dim 3."
flat3(a) = reshape(a, size(a, 1), size(a, 2), :)

"Shift a box's dim-3 range by `off` (batch stacking uses dim 3)."
shift3(b, off) = (b[1], b[2], b[3] .+ off)

"""
Split a box whose dim-3 range is in compact (truncated) indices `1:m` at the
positive/negative frequency boundary `kcut+1`, mapping the negative part to
its fine position in an FFT array of size `nfine`. Parts stay in ascending
compact order (dim 3 is the slowest pack dimension, so the buffer segment
order is unchanged).
"""
finesplit3(b, kcut, m, nfine) = [
    (b[1], b[2], r) for r in (
        intersect(b[3], 1:(kcut+1)),
        intersect(b[3], (kcut+2):m) .+ (nfine - m),
    ) if !isempty(r)
]

"""
Batched transpose plan between adjacent layouts `src` and `dst` for `nb`
fields stacked along dim 4 of buffers whose dim-3 sizes are `srcdim3` and
`dstdim3` (which may exceed the layout's — retained modes at fine positions
in a larger FFT buffer). `splitsrc`/`splitdst` = `(kcut, m, nfine)` remaps
that side's dim-3 boxes from compact to fine positions via
[`finesplit3`](@ref). With a single peer the receive buffer aliases the
send buffer (pack → unpack, no copy, no MPI).
"""
function plan_spectral_transpose(
    src,
    dst,
    topo,
    backend,
    T;
    nb,
    srcdim3,
    dstdim3,
    splitsrc = nothing,
    splitdst = nothing,
    stagehost = false,
)
    a = swapped_axis(src, dst)
    (; procgrid, coords) = topo
    peers = 0:(procgrid[a]-1)
    sb = map(peers) do c
        peer = layout(dst.gdims, dst.axes, procgrid, setcoord(coords, a, c))
        localize(intersect_box(src.ranges, peer.ranges), src.ranges)
    end
    rb = map(peers) do c
        peer = layout(src.gdims, src.axes, procgrid, setcoord(coords, a, c))
        localize(intersect_box(dst.ranges, peer.ranges), dst.ranges)
    end
    split(boxes, spl) = isnothing(spl) ? map(b -> [b], boxes) : map(b -> finesplit3(b, spl...), boxes)
    batch(perpeer, d3) = [shift3(b, f * d3) for bs in perpeer for f in 0:(nb-1) for b in bs]
    sendcounts = nb .* map(b -> prod(length, b), sb)
    recvcounts = nb .* map(b -> prod(length, b), rb)
    sendbuf = KernelAbstractions.allocate(backend, Complex{T}, sum(sendcounts))
    single = length(peers) == 1
    (;
        subcomm = topo.subcomms[a],
        sendboxes = batch(split(sb, splitsrc), srcdim3),
        recvboxes = batch(split(rb, splitdst), dstdim3),
        sendcounts,
        recvcounts,
        sendbuf,
        recvbuf = single ? sendbuf :
                  KernelAbstractions.allocate(backend, Complex{T}, sum(recvcounts)),
        hostsendbuf = stagehost && !single ? Vector{Complex{T}}(undef, sum(sendcounts)) : nothing,
        hostrecvbuf = stagehost && !single ? Vector{Complex{T}}(undef, sum(recvcounts)) : nothing,
    )
end

"""
Redistribute `src` into `dst` (both reshaped to 3D via [`flat3`](@ref))
according to a [`plan_spectral_transpose`](@ref) plan. Like
[`transpose!`](@ref) but with multi-box-per-peer lists (fine splits and
batching); the single-peer case skips MPI through the aliased buffers.
"""
function spectral_transpose!(dst, src, plan, backend)
    pack!(plan.sendbuf, src, plan.sendboxes, backend)
    KernelAbstractions.synchronize(backend)
    if length(plan.sendcounts) > 1
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
    end
    unpack!(dst, plan.recvbuf, plan.recvboxes, backend)
    KernelAbstractions.synchronize(backend)
end

# Expand the state's compact dim 2 (ky) to fine positions in the y-pencil
# FFT buffer, zeroing the dead band, and back. The forward truncation also
# applies the 1/n³ normalization (û = F[u]/n³, so Σ|û|² = ⟨|u|²⟩; the
# backward transforms are unnormalized, making the round trip exact).

@kernel function pad_y_kernel!(ây, @Const(uh), kcut, m, n)
    i, j, k, c = @index(Global, NTuple)
    @inbounds if kcut + 1 < j < n - kcut + 1
        ây[i, j, k, c] = zero(eltype(ây))
    else
        ây[i, j, k, c] = uh[i, j ≤ kcut + 1 ? j : j - (n - m), k, c]
    end
end

@kernel function truncate_y_kernel!(uh, @Const(ây), kcut, m, n, invn3)
    i, j, k, c = @index(Global, NTuple)
    @inbounds uh[i, j, k, c] = ây[i, j ≤ kcut + 1 ? j : j + (n - m), k, c] * invn3
end

"""
Backward pipeline: truncated spectral state `uh` (y-pencil, batch of 3) →
dealiased physical velocities `v` (x-pencil, transform grid). `uh` is
read-only; the pipeline buffers are scratch.
"""
function spec_to_phys!(v, uh, s)
    (; backend, kcut, m, n) = s
    f = s.fft
    pad_y_kernel!(backend)(f.ây, uh, kcut, m, n; ndrange = size(f.ây))
    KernelAbstractions.synchronize(backend)
    f.byplan! * f.ây
    spectral_transpose!(flat3(f.âz), flat3(f.ây), f.yz, backend)
    @views f.âz[:, :, (kcut+2):(n-kcut), :] .= 0
    f.bzplan! * f.âz
    spectral_transpose!(flat3(f.âx), flat3(f.âz), f.zx, backend)
    @views f.âx[(kcut+2):end, :, :, :] .= 0
    mul!(v, f.brplan, f.âx)   # brfft destroys âx; âx is scratch
    KernelAbstractions.synchronize(backend)
    v
end

"""
Forward pipeline: physical fields `w` (x-pencil, batch of 3) → truncated
spectral `uh` (y-pencil). Modes beyond the 2/3 cutoff are dropped at each
stage before they are communicated.
"""
function phys_to_spec!(uh, w, s)
    (; backend, kcut, m, n) = s
    f = s.fft
    mul!(f.âx, f.rplan, w)
    spectral_transpose!(flat3(f.âz), flat3(f.âx), f.xz, backend)
    f.zplan! * f.âz
    spectral_transpose!(flat3(f.ây), flat3(f.âz), f.zy, backend)
    f.yplan! * f.ây
    truncate_y_kernel!(backend)(uh, f.ây, kcut, m, n, s.T(1) / s.T(n)^3; ndrange = size(uh))
    KernelAbstractions.synchronize(backend)
    uh
end
