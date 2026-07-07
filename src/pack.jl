# Copying boxes (rectangular index regions) of 3D arrays to and from
# contiguous buffer segments. Used by both the transposes and the halo
# exchange. A box is an NTuple{3, UnitRange{Int}} of local array indices.

@kernel function pack_kernel!(buf, @Const(a), off, o1, o2, o3, n1, n2)
    i, j, k = @index(Global, NTuple)
    @inbounds buf[off+i+n1*(j-1)+n1*n2*(k-1)] = a[o1+i, o2+j, o3+k]
end

@kernel function unpack_kernel!(a, @Const(buf), off, o1, o2, o3, n1, n2)
    i, j, k = @index(Global, NTuple)
    @inbounds a[o1+i, o2+j, o3+k] = buf[off+i+n1*(j-1)+n1*n2*(k-1)]
end

@kernel function copybox_kernel!(a, @Const(b), d1, d2, d3, s1, s2, s3)
    i, j, k = @index(Global, NTuple)
    @inbounds a[d1+i, d2+j, d3+k] = b[s1+i, s2+j, s3+k]
end

"Pack `a[boxes[1]]…a[boxes[end]]` contiguously into `buf` (column-major per box)."
function pack!(buf, a, boxes, backend)
    off = 0
    for b in boxes
        n = prod(length, b)
        n == 0 && continue
        o = map(r -> first(r) - 1, b)
        pack_kernel!(backend)(
            buf, a, off, o..., length(b[1]), length(b[2]);
            ndrange = map(length, b),
        )
        off += n
    end
end

"Inverse of [`pack!`](@ref)."
function unpack!(a, buf, boxes, backend)
    off = 0
    for b in boxes
        n = prod(length, b)
        n == 0 && continue
        o = map(r -> first(r) - 1, b)
        unpack_kernel!(backend)(
            a, buf, off, o..., length(b[1]), length(b[2]);
            ndrange = map(length, b),
        )
        off += n
    end
end

"Copy box `bsrc` of `src` into box `bdst` of `dst` (equal box sizes)."
function copybox!(dst, bdst, src, bsrc, backend)
    @assert map(length, bdst) == map(length, bsrc)
    prod(length, bdst) == 0 && return
    copybox_kernel!(backend)(
        dst, src, map(r -> first(r) - 1, bdst)..., map(r -> first(r) - 1, bsrc)...;
        ndrange = map(length, bdst),
    )
end
