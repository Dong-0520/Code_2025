include("../src/element_type.jl")
using .ElementType

using LinearAlgebra
using Test

@testset verbose=true "getSbpOperators1d" begin
    for (degree, boundary_nodes) in [(3, true), (3, false), (4, true), (4, false)]
        n_nodes = degree + 1

        w, Q, R, x = getSbpOperators1d(degree, boundary_nodes = boundary_nodes)
        H = diagm(w)
        D = inv(H) * Q
        xL = -1
        xR = 1

        if boundary_nodes
            tL = zeros(1, length(x))
            tR = zeros(1, length(x))
            tL[1, 1] = 1
            tR[1, length(x)] = 1
        else
            tL, tR = R
        end
        E = tR' * tR - tL' * tL

        @testset verbose=true "degree $degree D with$(boundary_nodes ? "" : "out") boundary nodes polynomial accuracy" begin
            @test D * ones(n_nodes)≈zeros(n_nodes) atol=1e-10
            for k in 1:degree
                @test D * x' .^ k≈k * x' .^ (k - 1) atol=1e-10
            end
        end

        @testset verbose=true "degree $degree tL with$(boundary_nodes ? "" : "out") boundary nodes polynomial accuracy" begin
            for k in 0:degree
                @test dot(tL, x .^ k)≈xL^k atol=1e-10
            end
        end

        @testset verbose=true "degree $degree tR with$(boundary_nodes ? "" : "out") boundary nodes polynomial accuracy" begin
            for k in 0:degree
                @test dot(tR, x .^ k)≈xR^k atol=1e-10
            end
        end

        @testset verbose=true "degree $degree operator with$(boundary_nodes ? "" : "out") boundary nodes SBP property" begin
            @test Q + Q'≈E atol=1e-10
        end
    end
end
