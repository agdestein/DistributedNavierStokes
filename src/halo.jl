# Halo exchange: ghost layers are filled by MPI neighbors, periodic wraps,
# or physical wall conditions — decided per dimension by the cartesian
# topology (`MPI.PROC_NULL` neighbors appear exactly at walls). Dimensions
# are exchanged in fixed order and sent slabs span the full extent
# (including ghosts) of the other dimensions, so edge/corner ghosts are
# correct without diagonal communication.

@kernel function mirror_kernel!(φ, jdst, jsrc, s)
    i, k = @index(Global, NTuple)
    @inbounds φ[i, jdst, k] = s * φ[i, jsrc, k]
end

@kernel function setplane_kernel!(φ, j, v)
    i, k = @index(Global, NTuple)
    @inbounds φ[i, j, k] = v
end

"""
Fill wall ghost layers of `φ` in the y direction on `side` (`:lo`/`:hi`).
The staggering `stag` decides the fill: the wall-normal component `:y` has
its wall-face DOF set to zero and ghosts filled antisymmetrically;
tangential components (`:x`, `:z`) mirror antisymmetrically about the wall
plane (no-slip with zero wall velocity); `:p` mirrors symmetrically
(homogeneous Neumann).
"""
function fill_wall!(φ, stag, side, setup)
    (; w, backend) = setup
    plane = (size(φ, 1), size(φ, 3))
    nl = size(φ, 2) - 2w
    mirror!(jdst, jsrc, s) = mirror_kernel!(backend)(φ, jdst, jsrc, s; ndrange = plane)
    if stag == :y
        jwall = side == :lo ? w : nl + w
        setplane_kernel!(backend)(φ, jwall, zero(eltype(φ)); ndrange = plane)
        for m = 1:(side==:lo ? w-1 : w)
            side == :lo && mirror!(jwall - m, jwall + m, -1)
            side == :hi && mirror!(jwall + m, jwall - m, -1)
        end
    else
        s = stag == :p ? 1 : -1
        for m = 1:w
            side == :lo && mirror!(w + 1 - m, w + m, s)
            side == :hi && mirror!(nl + w + m, nl + w + 1 - m, s)
        end
    end
end

"Box spanning `r` in dimension `d` and the full extent of the others."
facebox(d, r, ext) = ntuple(i -> i == d ? r : (1:ext[i]), 3)

"""
Fill all ghost layers of `φ` with staggering `stag ∈ (:x, :y, :z, :p)`:
MPI exchange with neighbor ranks, periodic wrap where a periodic dimension
is rank-local, and wall fills where the topology has no neighbor.
"""
function exchange_halo!(φ, stag, setup)
    (; topo, axes, bc, w, backend, halo) = setup
    ext = size(φ)
    for d = 1:3
        a = axes[d]
        nl = ext[d] - 2w
        lointerior = facebox(d, (w+1):2w, ext)
        hiinterior = facebox(d, (nl+1):(nl+w), ext)
        loghost = facebox(d, 1:w, ext)
        highost = facebox(d, (nl+w+1):(nl+2w), ext)
        if a == 0 # dimension local to this rank
            if bc[d] == :periodic
                copybox!(φ, loghost, φ, hiinterior, backend)
                copybox!(φ, highost, φ, lointerior, backend)
            else
                fill_wall!(φ, stag, :lo, setup)
                fill_wall!(φ, stag, :hi, setup)
            end
        else
            (; lo, hi) = topo.neighbors[a]
            nb = prod(length, loghost)
            sendbuf = view(halo.sendbuf, 1:nb)
            recvbuf = view(halo.recvbuf, 1:nb)
            # hi-ward: my top interior slab becomes hi's lo ghosts
            pack!(sendbuf, φ, (hiinterior,), backend)
            KernelAbstractions.synchronize(backend)
            MPI.Sendrecv!(sendbuf, recvbuf, topo.cart; dest = hi, source = lo)
            lo == MPI.PROC_NULL || unpack!(φ, recvbuf, (loghost,), backend)
            # lo-ward: my bottom interior slab becomes lo's hi ghosts
            pack!(sendbuf, φ, (lointerior,), backend)
            KernelAbstractions.synchronize(backend)
            MPI.Sendrecv!(sendbuf, recvbuf, topo.cart; dest = lo, source = hi)
            hi == MPI.PROC_NULL || unpack!(φ, recvbuf, (highost,), backend)
            if bc[d] == :wall
                lo == MPI.PROC_NULL && fill_wall!(φ, stag, :lo, setup)
                hi == MPI.PROC_NULL && fill_wall!(φ, stag, :hi, setup)
            end
        end
    end
    KernelAbstractions.synchronize(backend)
end

"Exchange halos of all three velocity components."
function exchange_halo!(u::NamedTuple, setup)
    exchange_halo!(u.x, :x, setup)
    exchange_halo!(u.y, :y, setup)
    exchange_halo!(u.z, :z, setup)
end
