# Pencil layouts: which processor-grid axis each global dimension is
# distributed over, and the resulting local index ranges. A layout is plain
# data (a NamedTuple); the MPI cartesian communicator provides the topology.

"Balanced block `c` (0-based) of `n` points over `p` ranks."
blockrange(n, p, c) = (c * n ÷ p + 1):((c + 1) * n ÷ p)

"""
Processor-grid axis assignment `axes[d] ∈ (0, 1, 2)` for each global
dimension `d` (0 = local), for the three pencil orientations. The adjacency
is `x ↔ z ↔ y`: each transpose swaps the local dimension with one
distributed dimension along a single processor-grid axis, so the Poisson
sequence FFT(x) → FFT(z) → solve(y) is one transpose per arrow.
"""
const AXES = (; x = (0, 2, 1), z = (1, 2, 0), y = (1, 0, 2))

"""
Layout descriptor: local index ranges (in global indices) of the block owned
by the rank at processor-grid coordinates `coords` (0-based), for a global
array of size `gdims` distributed according to `axes`.
"""
function layout(gdims, axes, procgrid, coords)
    ranges = ntuple(3) do d
        a = axes[d]
        a == 0 ? (1:gdims[d]) : blockrange(gdims[d], procgrid[a], coords[a])
    end
    (; gdims, axes, ranges, ldims = map(length, ranges))
end

"""
Create the 2D cartesian process topology on `comm`. `periodic[a]` must hold
for processor-grid axis `a` iff the global dimension mapped to `a` (in the
native orientation) is periodic. Returns the cartesian communicator, this
rank's 0-based grid coordinates, per-axis subcommunicators (for transposes),
and per-axis `(lo, hi)` neighbor ranks (`MPI.PROC_NULL` beyond walls).
"""
function topology(comm, procgrid, periodic)
    prod(procgrid) == MPI.Comm_size(comm) ||
        error("procgrid $procgrid does not match $(MPI.Comm_size(comm)) ranks")
    cart = MPI.Cart_create(comm, procgrid; periodic = collect(periodic), reorder = false)
    coords = Tuple(MPI.Cart_coords(cart))
    subcomms = ntuple(a -> MPI.Cart_sub(cart, ntuple(==(a), 2)), 2)
    neighbors = ntuple(2) do a
        lo, hi = MPI.Cart_shift(cart, a - 1, 1)
        (; lo, hi)
    end
    (; cart, coords, procgrid, subcomms, neighbors, rank = MPI.Comm_rank(cart))
end
