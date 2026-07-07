# DistributedNavierStokes

*(working title)*

Design and (eventually) implementation of an MPI-parallelized, multi-GPU
incompressible Navier-Stokes DNS solver, targeting grids beyond what fits on a
single GPU (> 800³). Companion project to
[IncompressibleNavierStokes.jl](https://github.com/agdestein/IncompressibleNavierStokes.jl),
which remains the single-node, differentiable, methods-research code.

See [DESIGN.md](DESIGN.md) for the design document and decision log.
