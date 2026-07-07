# Run with:
#   julia --project=test -e 'using Pkg; Pkg.resolve(); include("test/runtests.jl")'
# Serial unit tests run in this process; multi-rank tests are launched under
# the MPI-provided mpiexec for several rank counts and processor grids.

using DistributedNavierStokes
using KernelAbstractions
using MPI
using Random
using Test

const DNS = DistributedNavierStokes

MPI.Init()

@testset "DistributedNavierStokes" begin
    include("serialtests.jl")

    @testset "blockrange" begin
        for n in (1, 5, 8, 13), p in (1, 2, 3, 4)
            blocks = [DNS.blockrange(n, p, c) for c = 0:(p-1)]
            @test first(blocks[1]) == 1
            @test last(blocks[end]) == n
            @test all(first(blocks[c+1]) == last(blocks[c]) + 1 for c = 1:(p-1))
            @test maximum(length, blocks) - minimum(length, blocks) ≤ 1
        end
    end

    mpitests = joinpath(@__DIR__, "mpitests.jl")
    project = Base.active_project()
    env = copy(ENV)
    env["OMPI_MCA_rmaps_base_oversubscribe"] = "true" # ignored by non-OpenMPI
    @testset "mpiexec -n $n" for n in (1, 2, 4)
        cmd = `$(MPI.mpiexec()) -n $n $(Base.julia_cmd()) --startup-file=no --project=$project $mpitests`
        @test success(run(setenv(ignorestatus(cmd), env)))
    end
end
