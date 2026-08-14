# Setup for the pseudo-spectral solver: cubic periodic box, transform grid
# n³, spectral state truncated at the 2/3 cutoff. One nested NamedTuple
# holding the decomposition, all layouts, FFT plans, transpose plans, and
# preallocated buffers; the time loop allocates nothing.

"""
    spectral_setup(; n, kwargs...)

Set up a pseudo-spectral problem on a cubic periodic box with `n` transform
grid points per direction (even). The spectral state holds only the
retained modes `|k| ≤ kcut` (2/3 rule: `kcut = n ÷ 3`), in the y-pencil
orientation; physical fields live on the transform grid in x-pencils.

Keywords:

- `l = 2π`: box side length.
- `visc = 1e-3`: kinematic viscosity.
- `kcut = n ÷ 3`: truncation wavenumber (alias-free needs `3kcut ≤ n`).
- `procgrid = nothing`: 2D processor grid; default `(1, nranks)` slabs —
  the x→z stage is then communication-free, measured 2-2.5× faster than
  near-square grids at 2-4 GPUs (CAMPAIGN.md S0; `(P, 1)` instead makes
  z→y communication-free but measured slower). Pass an explicit grid at
  rank counts beyond the measured range.
- `mpibuf = :auto`: as in [`setup`](@ref), plus `:nccl` — transpose
  payloads go over NCCL instead of MPI (GPU backend only; requires
  `using CUDA, NCCL` to load the package extension, and one distinct
  device per rank). Bypasses MPI/UCX for the bandwidth-critical path;
  MPI still handles reductions and I/O.
- `backend = CPU()`: KernelAbstractions backend (e.g. `CUDABackend()`).
- `T = Float64`: element type.
- `comm = MPI.COMM_WORLD`: MPI communicator to decompose over.
"""
function spectral_setup(;
    n,
    l = 2π,
    visc = 1e-3,
    kcut = n ÷ 3,
    procgrid = nothing,
    mpibuf = :auto,
    backend = CPU(),
    T = Float64,
    comm = MPI.COMM_WORLD,
)
    n % 2 == 0 || error("n must be even")
    1 ≤ kcut || error("kcut must be at least 1")
    3kcut ≤ n || error("aliasing: 3kcut ≤ n is required (2/3 rule)")
    mpibuf in (:auto, :device, :host, :nccl) ||
        error("mpibuf must be :auto, :device, :host, or :nccl")

    MPI.Initialized() || MPI.Init()
    usenccl = mpibuf == :nccl
    if usenccl
        hasmethod(nccl_subcomm, Tuple{MPI.Comm}) || error(
            "mpibuf = :nccl requires the NCCL extension: load CUDA.jl and NCCL.jl first",
        )
        backend isa CPU && error("mpibuf = :nccl requires a GPU backend")
    end
    stagehost =
        mpibuf == :host || (mpibuf == :auto && !(backend isa CPU) && !MPI.has_cuda())
    nranks = MPI.Comm_size(comm)
    procgrid = something(procgrid, (1, nranks))
    topo = topology(comm, procgrid, (true, true))
    ncclcomms = usenccl ? map(nccl_subcomm, topo.subcomms) : (nothing, nothing)

    m = 2kcut + 1
    nb = 3
    lphys = layout((n, n, n), AXES.x, procgrid, topo.coords)
    lxt = layout((kcut + 1, n, n), AXES.x, procgrid, topo.coords)
    lzf = layout((kcut + 1, n, n), AXES.z, procgrid, topo.coords)
    lzt = layout((kcut + 1, n, m), AXES.z, procgrid, topo.coords)
    lyt = layout((kcut + 1, n, m), AXES.y, procgrid, topo.coords)
    lspec = layout((kcut + 1, m, m), AXES.y, procgrid, topo.coords)
    for lay in (lphys, lxt, lzf, lspec)
        all(>(0), lay.ldims) ||
            error("empty local block (procgrid $procgrid too large for n = $n, kcut = $kcut)")
    end

    alloc(dims...) = KernelAbstractions.allocate(backend, Complex{T}, dims...)
    v = KernelAbstractions.zeros(backend, T, lphys.ldims..., nb)
    rbuf = KernelAbstractions.allocate(backend, T, lphys.ldims..., nb)
    âx = alloc(n ÷ 2 + 1, lphys.ldims[2], lphys.ldims[3], nb)
    âz = alloc(lzf.ldims..., nb)
    ây = alloc(lyt.ldims..., nb)
    σh = alloc(lspec.ldims..., nb)
    escratch = KernelAbstractions.allocate(backend, T, lspec.ldims...)

    xz = plan_spectral_transpose(
        lxt, lzf, topo, backend, T;
        nb, srcdim3 = size(âx, 3), dstdim3 = n, stagehost, ncclcomms,
    )
    zy = plan_spectral_transpose(
        lzt, lyt, topo, backend, T;
        nb, srcdim3 = n, dstdim3 = lyt.ldims[3],
        splitsrc = (kcut, m, n), stagehost, ncclcomms,
        reuse = xz,   # stages never overlap; the x↔z payload is the larger
    )
    rplan = trim_plan!(plan_rfft(rbuf, 1))
    brplan = trim_plan!(plan_brfft(âx, n, 1))

    # Physical wavenumbers of the state layout's local ranges.
    kfac = T(2π / l)
    kx = todevice(backend, T[kfac * (gi - 1) for gi in lspec.ranges[1]])
    ky = todevice(backend, T[kfac * compactfreq(j, m, kcut) for j in lspec.ranges[2]])
    kz = todevice(backend, T[kfac * compactfreq(gk, m, kcut) for gk in lspec.ranges[3]])
    # Integer counterparts (for exact shell binning) and the rfft
    # double-count weight of dim 1 (kx = 0 once, kx > 0 twice).
    ikx = todevice(backend, Int32[gi - 1 for gi in lspec.ranges[1]])
    iky = todevice(backend, Int32[compactfreq(j, m, kcut) for j in lspec.ranges[2]])
    ikz = todevice(backend, Int32[compactfreq(gk, m, kcut) for gk in lspec.ranges[3]])
    w1 = todevice(backend, T[gi == 1 ? 1 : 2 for gi in lspec.ranges[1]])

    (;
        n,
        l = T(l),
        visc = T(visc),
        kcut,
        m,
        T,
        backend,
        topo,
        stagehost,
        mpimode = usenccl ? :nccl : stagehost ? :host : :device,
        lphys,
        lspec,
        kx,
        ky,
        kz,
        kxr = reshape(kx, :, 1, 1),
        kyr = reshape(ky, 1, :, 1),
        kzr = reshape(kz, 1, 1, :),
        ikxr = reshape(ikx, :, 1, 1),
        ikyr = reshape(iky, 1, :, 1),
        ikzr = reshape(ikz, 1, 1, :),
        w1r = reshape(w1, :, 1, 1),
        # local kx > 0 rows (counted twice in spectral sums)
        dbl = (first(lspec.ranges[1]) == 1 ? 2 : 1):lspec.ldims[1],
        fft = (;
            nb,
            v,
            rbuf,
            âx,
            âz,
            ây,
            σh,
            escratch,
            rplan,
            brplan,
            zplan! = plan_fft!(âz, 3),
            bzplan! = plan_bfft!(âz, 3),
            yplan! = plan_fft!(ây, 2),
            byplan! = plan_bfft!(ây, 2),
            xz,
            zx = reverse_plan(xz),
            zy,
            yz = reverse_plan(zy),
        ),
    )
end

"Truncated spectral velocity state (zeros): `(l1, m_local, l3, 3)` complex."
specvelocity(s) =
    KernelAbstractions.zeros(s.backend, Complex{s.T}, s.lspec.ldims..., s.fft.nb)
