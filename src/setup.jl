# Problem setup: one nested NamedTuple holding the grid, the decomposition,
# and all preallocated plans, buffers, and coefficients. Constructed once;
# the time loop allocates nothing.

"""
    setup(; n, kwargs...)

Set up a problem on a global grid of `n = (nx, ny, nz)` cells. Directions x
and z are periodic and uniform; y is periodic or `:wall` (no-slip) and may
be stretched by `stretch`, a monotone map of [0, 1] onto itself applied to
the y face distribution.

Keywords:

- `lims = ((0, 2π), (0, 2π), (0, 2π))`: physical domain extent per direction.
- `bc = (:periodic, :periodic, :periodic)`: `bc[2]` may be `:wall`.
- `stretch = identity`: y-face distribution map.
- `visc = 1e-3`: kinematic viscosity.
- `bodyforce = (0, 0, 0)`: constant body force (e.g. channel driving).
- `procgrid = nothing`: 2D processor grid; default near-square.
- `order = 2`: discretization order (4 not implemented yet).
- `backend = CPU()`: KernelAbstractions backend (e.g. `CUDABackend()`).
- `T = Float64`: element type.
- `comm = MPI.COMM_WORLD`: MPI communicator to decompose over.
"""
function setup(;
    n,
    lims = ((0.0, 2π), (0.0, 2π), (0.0, 2π)),
    bc = (:periodic, :periodic, :periodic),
    stretch = identity,
    visc = 1e-3,
    bodyforce = (0.0, 0.0, 0.0),
    procgrid = nothing,
    order = 2,
    backend = CPU(),
    T = Float64,
    comm = MPI.COMM_WORLD,
)
    bc[1] == bc[3] == :periodic || error("x and z must be periodic")
    bc[2] in (:periodic, :wall) || error("bc[2] must be :periodic or :wall")
    order == 2 || error("only order 2 is implemented so far")
    w = order ÷ 2

    MPI.Initialized() || MPI.Init()
    nranks = MPI.Comm_size(comm)
    procgrid = something(procgrid, squarest(nranks))

    # Native orientation :x — the axes/periodicity of the process topology
    # follow its dimension → axis mapping.
    axes = AXES.x
    periodic = ntuple(a -> bc[findfirst(==(a), axes)] == :periodic, 2)
    topo = topology(comm, procgrid, periodic)
    lay = layout(n, axes, procgrid, topo.coords)
    all(d -> axes[d] == 0 || lay.ldims[d] >= w, 1:3) ||
        error("local block smaller than ghost width; use fewer ranks")

    grid = griddata(n, lims, bc, stretch, lay, w, backend, T)
    ranges = (;
        x = activeranges(lay, n, bc, w, 1),
        y = activeranges(lay, n, bc, w, 2),
        z = activeranges(lay, n, bc, w, 3),
        p = activeranges(lay, n, bc, w, 0),
    )

    ext = lay.ldims .+ 2w
    nhalo = w * maximum(d -> prod(ext) ÷ ext[d], 1:3)
    halo = (;
        sendbuf = KernelAbstractions.allocate(backend, T, nhalo),
        recvbuf = KernelAbstractions.allocate(backend, T, nhalo),
    )

    s = (;
        n,
        lims,
        bc,
        visc = T(visc),
        bodyforce = T.(bodyforce),
        order,
        w,
        T,
        backend,
        topo,
        axes,
        lay,
        grid,
        ranges,
        halo,
        rk = rk3,
    )
    (; s..., poisson = plan_poisson(s))
end

"Near-square factorization `(p1, p2)` of `P` with `p1 ≥ p2`."
function squarest(P)
    best = (P, 1)
    for a = 1:isqrt(P)
        P % a == 0 && (best = (P ÷ a, a))
    end
    best
end

"""
Active-DOF local array-index ranges (into ghosted arrays) for velocity
component `c` (0 for cell-centered quantities). Faces are indexed by the
cell they top: face `j` is the right/top face of cell `j`. In a periodic
direction faces 1:n are active; in the wall direction the wall faces (0 and
ny) are boundary values, so active y-faces of `u.y` are 1:ny-1 — the rank
owning the last cell drops its top face.
"""
function activeranges(lay, n, bc, w, c)
    ntuple(3) do d
        gr = lay.ranges[d]
        hi = c == d && bc[d] == :wall ? min(last(gr), n[d] - 1) : last(gr)
        (w+1):(hi-first(gr)+1+w)
    end
end

"""
Grid metrics and coordinates. Uniform in x and z (`Δx`, `Δz` scalars);
y metrics are 1D arrays over the local cell range including `w` ghost
layers (`Δy[j]`: cell widths; `Δyu[j]`: distance between cell centers `j`
and `j+1`, the width of `u.y`'s staggered volume), on the backend for
kernels. `x/y/zf` and `x/y/zc` are local face/center coordinate vectors
(faces indexed by the cell they top), on the host, for initial conditions.
"""
function griddata(n, lims, bc, stretch, lay, w, backend, T)
    (nx, ny, nz) = n
    Lx, Ly, Lz = map(l -> l[2] - l[1], lims)
    Δx = T(Lx / nx)
    Δz = T(Lz / nz)

    # Global extended y faces: yf[j] for j in -w:ny+w, index offset w + 1.
    yface(j) = lims[2][1] + Ly * stretch(j / ny)
    yf = [yface(j) for j = 0:ny]
    yfe = zeros(T, ny + 2w + 1)
    yfe[(w+1):(w+ny+1)] .= yf
    for m = 1:w
        if bc[2] == :periodic
            yfe[w+1-m] = yf[1] - (yf[end] - yf[end-m])
            yfe[w+ny+1+m] = yf[end] + (yf[1+m] - yf[1])
        else # mirror at the walls
            yfe[w+1-m] = 2yf[1] - yf[1+m]
            yfe[w+ny+1+m] = 2yf[end] - yf[end-m]
        end
    end
    Δye = diff(yfe)                        # cell widths, cells 1-w:ny+w
    yce = (yfe[1:(end-1)] .+ yfe[2:end]) ./ 2  # cell centers
    Δyue = [diff(yce); Δye[end]]           # center distances (last entry padding)

    # Local slices (global cells jlo-w:jhi+w → extended index .+ w).
    jr = lay.ranges[2]
    js = (first(jr)):(last(jr)+2w)         # == (jlo-w:jhi+w) .+ w

    (;
        Δx,
        Δz,
        Δy = todevice(backend, Δye[js]),
        Δyu = todevice(backend, Δyue[js]),
        Δymin = minimum(Δye[(w+1):(w+ny)]),
        xf = [lims[1][1] + i * Lx / nx for i in lay.ranges[1]],
        xc = [lims[1][1] + (i - 1 / 2) * Lx / nx for i in lay.ranges[1]],
        yf = [yfe[j+w+1] for j in jr],
        yc = [yce[j+w] for j in jr],
        zf = [lims[3][1] + k * Lz / nz for k in lay.ranges[3]],
        zc = [lims[3][1] + (k - 1 / 2) * Lz / nz for k in lay.ranges[3]],
        Δye,                               # global extended (host), for Poisson setup
        Δyue,
    )
end
