# Snapshot / restart I/O for the spectral solver. The state is exactly the
# retained (dealiased) coefficients — there are no ghost modes to strip —
# and is written with collective MPI-IO subarray views in *global index
# order*: `<prefix>.bin` holds the (kcut+1, m, m, 3) complex array
# column-major, one component after the other, so the byte layout is
# independent of rank count and processor grid (a run on 1 GPU and on 128
# produces the same file up to FFT-reordering roundoff in the values). A
# `<prefix>.toml` sidecar carries the metadata. No HDF5: MPI-IO comes with
# the MPI dependency we already have (CODE_DESIGN.md §2), and TOML is stdlib.
#
# Components are written through separate views (a byte displacement per
# component) so per-call MPI counts stay far below the Cint limit even for
# large local blocks (few ranks, big grids).

"MPI subarray filetype of this rank's block of one spectral component."
function specblocktype(s)
    (; kcut, m, lspec) = s
    t = MPI.Types.create_subarray(
        (kcut + 1, m, m),
        lspec.ldims,
        map(r -> first(r) - 1, lspec.ranges),
        MPI.Datatype(Complex{s.T}),
    )
    MPI.Types.commit!(t)
    t
end

"""
    spectral_save(prefix, uh, s; time = 0.0, step = 0, meta = (;))

Save the spectral state `uh` to `<prefix>.bin` (collective MPI-IO, global
index order, rank-count independent) with a `<prefix>.toml` metadata
sidecar. `meta` is a NamedTuple of extra scalars/vectors to store (e.g.
forcing reference energies). Atomic: both files appear under their final
names only when complete. Returns `prefix`.
"""
function spectral_save(prefix, uh, s; time = 0.0, step = 0, meta = (;))
    (; T, kcut, m, topo) = s
    datafile, metafile = "$prefix.bin", "$prefix.toml"
    fieldbytes = sizeof(Complex{T}) * (kcut + 1) * m * m
    uhh = convert(Array{Complex{T},4}, uh)   # host stage (no-op on CPU)
    topo.rank == 0 && rm("$datafile.tmp"; force = true)
    MPI.Barrier(topo.cart)
    ftype = specblocktype(s)
    fh = MPI.File.open(topo.cart, "$datafile.tmp"; write = true, create = true)
    for c = 1:3
        MPI.File.set_view!(fh, (c - 1) * fieldbytes, MPI.Datatype(Complex{T}), ftype)
        MPI.File.write_all(fh, view(uhh, :, :, :, c))
    end
    MPI.File.close(fh)
    if topo.rank == 0
        d = Dict{String,Any}(
            "format" => "DistributedNavierStokes spectral state",
            "version" => 1,
            "eltype" => string(T),
            "n" => s.n,
            "kcut" => kcut,
            "m" => m,
            "l" => Float64(s.l),
            "visc" => Float64(s.visc),
            "time" => Float64(time),
            "step" => Int(step),
            "order" => "column-major (kx, ky, kz, component) complex; " *
                       "kx = 0:kcut, ky/kz wrapped 0:kcut then -kcut:-1",
            "data" => basename(datafile),
        )
        isempty(pairs(meta)) ||
            (d["meta"] = Dict{String,Any}(string(k) => v for (k, v) in pairs(meta)))
        open(io -> TOML.print(io, d; sorted = true), "$metafile.tmp", "w")
        mv("$metafile.tmp", metafile; force = true)
        mv("$datafile.tmp", datafile; force = true)
    end
    MPI.Barrier(topo.cart)
    prefix
end

"""
    spectral_load!(uh, prefix, s) -> (; time, step, meta)

Load a [`spectral_save`](@ref) snapshot into `uh` (collective MPI-IO; any
rank count and processor grid, independent of the writer's). The retained
modes must match the setup (`kcut`, `l`, eltype); the transform grid `n`
need not — a snapshot restarts on a finer grid for higher-Re continuation.
Returns the sidecar time, step, and `meta` dictionary.
"""
function spectral_load!(uh, prefix, s)
    (; T, kcut, m, lspec, topo) = s
    d = MPI.bcast(topo.rank == 0 ? TOML.parsefile("$prefix.toml") : nothing, topo.cart)
    d["eltype"] == string(T) && d["kcut"] == kcut && d["l"] == Float64(s.l) || error(
        "snapshot $prefix (eltype $(d["eltype"]), kcut $(d["kcut"]), l $(d["l"])) " *
        "does not match setup (eltype $T, kcut $kcut, l $(Float64(s.l)))",
    )
    fieldbytes = sizeof(Complex{T}) * (kcut + 1) * m * m
    uhh = Array{Complex{T},4}(undef, lspec.ldims..., 3)
    ftype = specblocktype(s)
    fh = MPI.File.open(topo.cart, "$prefix.bin"; read = true)
    for c = 1:3
        MPI.File.set_view!(fh, (c - 1) * fieldbytes, MPI.Datatype(Complex{T}), ftype)
        MPI.File.read_all!(fh, view(uhh, :, :, :, c))
    end
    MPI.File.close(fh)
    copyto!(uh, uhh)
    (; time = d["time"], step = d["step"], meta = get(d, "meta", Dict{String,Any}()))
end

"Checkpoint prefixes `<prefix>_<step>` with complete file pairs, ascending by step."
function checkpointlist(prefix)
    dir = isempty(dirname(prefix)) ? "." : dirname(prefix)
    base = basename(prefix)
    isdir(dir) || return String[]
    steps = Int[]
    for f in readdir(dir)
        mm = match(Regex("^\\Q$base\\E_(\\d+)\\.toml\$"), f)
        mm === nothing && continue
        isfile(joinpath(dir, "$(base)_$(mm.captures[1]).bin")) || continue
        push!(steps, parse(Int, mm.captures[1]))
    end
    sort!(steps)
    [joinpath(dir, @sprintf("%s_%09d", base, st)) for st in steps]
end

"""
    spectral_latest(prefix, s) -> String or nothing

Newest complete checkpoint written by [`checkpointer`](@ref) under `prefix`
(rank 0 scans the directory, the result is broadcast), or `nothing` when
none exists. Load it with [`spectral_load!`](@ref).
"""
spectral_latest(prefix, s) = MPI.bcast(
    s.topo.rank == 0 ?
    (l = checkpointlist(prefix); isempty(l) ? nothing : last(l)) : nothing,
    s.topo.cart,
)

"""
    checkpointer(prefix; interval = 900.0, stopfile = nothing, keep = 2, meta = (;))

Processor for [`spectral_solve!`](@ref) writing restart checkpoints
`<prefix>_<step>` (via [`spectral_save`](@ref), so atomic and rank-count
independent): every `interval` seconds of wall clock, and — when `stopfile`
is given — as soon as that file exists, in which case it also returns
`:stop` so the solve ends gracefully after the checkpoint. This is the
SLURM time-limit pattern: the job script traps the `--signal=B:USR1@...`
warning, touches `stopfile`, and resubmits itself (see
examples/snellius/spectral_h100.sh). Only the newest `keep` checkpoints are
retained. Rank 0 takes the decision and broadcasts it, so all ranks write
collectively. On restart, find the checkpoint with
[`spectral_latest`](@ref) and pass its `step` as the solver's `nstart` so
checkpoint numbering continues.
"""
function checkpointer(prefix; interval = 900.0, stopfile = nothing, keep = 2, meta = (;))
    tlast = Ref(time())
    (state, s) -> begin
        (; rank, cart) = s.topo
        due, stop = MPI.bcast(
            rank == 0 ?
            (
                time() - tlast[] ≥ interval,
                stopfile !== nothing && isfile(stopfile),
            ) : nothing,
            cart,
        )
        due || stop || return nothing
        spectral_save(
            @sprintf("%s_%09d", prefix, state.n),
            state.uh,
            s;
            time = state.t,
            step = state.n,
            meta,
        )
        tlast[] = time()
        if rank == 0
            old = checkpointlist(prefix)
            for p in old[1:(end-min(keep, length(old)))]
                rm("$p.bin"; force = true)
                rm("$p.toml"; force = true)
            end
        end
        # all ranks see the retention policy applied once the step completes
        MPI.Barrier(cart)
        stop ? :stop : nothing
    end
end

"""
    snapshotsaver(prefix; times, meta = (;), stats = true, tstart = -Inf)

Processor for [`spectral_solve!`](@ref) that saves the state the first time
`t` reaches each entry of `times`, as `<prefix>_0001`, `<prefix>_0002`, … in
order. Pass the same `times` as `tstops` to the solver so steps land on them
exactly; the sidecar records the actual `t`. With `stats = true` each
sidecar's `meta` also carries the K41 numbers measured at save time —
`eta` (Kolmogorov length), `t_int`, `e`, `diss` — so snapshot spacing in
`t_int` units and filter widths in `Δ/η` are self-documenting
(data-campaign hygiene; see [`etacells`](@ref)).

When restarting from a checkpoint, pass its time as `tstart`: entries
strictly before it are skipped *without* saving (they exist from the
earlier job — without the skip, the initial processor call would rewrite
every earlier index with the restart-time field). File numbering is by
position in the full `times` list, so it is restart-invariant; an entry
equal to `tstart` is re-saved, byte-identical since restarts are exact.
"""
function snapshotsaver(prefix; times, meta = (;), stats = true, tstart = -Inf)
    times = sort!(Float64[t for t in times])
    i0 = findfirst(t -> t ≥ tstart - 1e-10 * (abs(t) + 1), times)
    inext = Ref(something(i0, length(times) + 1))
    (state, s) -> begin
        while inext[] ≤ length(times) &&
            state.t ≥ times[inext[]] - 1e-10 * (abs(times[inext[]]) + 1)
            m = if stats
                st = spectral_stats(state.uh, s)
                (; meta..., eta = st.l_kol, t_int = st.t_int, e = st.e, diss = st.diss)
            else
                meta
            end
            spectral_save(
                @sprintf("%s_%04d", prefix, inext[]),
                state.uh,
                s;
                time = state.t,
                step = state.n,
                meta = m,
            )
            inext[] += 1
        end
    end
end

"""
    statswriter(; file, nupdate = 10)

Processor appending one CSV row of the K41 statistics
([`spectral_stats`](@ref)) every `nupdate` steps: `t`, `step`, then every
statistic (ε, L_int, t_int, η, …), then `walltime` — the stationarity
drift record the run archive keeps next to the data. `walltime` is rank
0's Unix epoch time at the write, so diffing any two rows gives the
wall-clock cost of that stretch of simulation; it stays monotone across
checkpoint restarts (chunk boundaries show up as gaps). It is the one
column that is not decomposition-invariant. Rank 0 writes, append mode
(a restarted run continues the series), header only when the file is
new, flushed per row. Collective (statistics are reductions).
"""
function statswriter(; file, nupdate = 10)
    io = Ref{Union{IOStream,Nothing}}(nothing)
    (state, s) -> begin
        state.n % nupdate == 0 || return nothing
        st = spectral_stats(state.uh, s)
        if s.topo.rank == 0
            if io[] === nothing
                mkpath(dirname(abspath(file)))
                fresh = !isfile(file) || filesize(file) == 0
                io[] = open(file, "a")
                fresh && println(io[], join(("t", "step", string.(keys(st))..., "walltime"), ","))
            end
            println(io[], join((state.t, state.n, values(st)..., time()), ","))
            flush(io[])
        end
        nothing
    end
end

"""
    spectral_from_rfft!(uh, F, s)

Fill the truncated state from spectral velocities `F` in full rfft layout —
three arrays sized `(nf÷2+1, nf, nf)` (a tuple or `(; x, y, z)`, wrapped
negative frequencies, the shared û = F[u]/n³ normalization), e.g. a
SymmetryCode DNS state. Requires `nf ≥ 3kcut` (the source's dealiased cube
must cover the state) and a single-rank setup: import once on `COMM_SELF`,
[`spectral_save`](@ref) the result, and any decomposition restarts from
the file (the V0 twin-validation recipe —
`examples/symmetrycode_import.jl`).
"""
function spectral_from_rfft!(uh, F, s)
    (; kcut, m) = s
    prod(s.topo.procgrid) == 1 ||
        error("serial import only — run on MPI.COMM_SELF, then spectral_save")
    Fs = values(F)
    length(Fs) == 3 || error("F must hold three velocity components")
    nf = size(Fs[1], 2)
    all(f -> size(f) == (nf ÷ 2 + 1, nf, nf), Fs) || error("expected full rfft layout")
    nf ≥ 3kcut || error("source grid nf = $nf cannot hold kcut = $kcut retained modes")
    A = Array{eltype(uh),4}(undef, kcut + 1, m, m, 3)
    fine(g) = (k = compactfreq(g, m, kcut); k ≥ 0 ? k + 1 : nf + k + 1)
    for c = 1:3, k = 1:m, j = 1:m, i = 1:(kcut+1)
        A[i, j, k, c] = Fs[c][i, fine(j), fine(k)]
    end
    copyto!(uh, A)
    uh
end
