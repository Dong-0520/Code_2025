include("../src/element_type.jl")
using .ElementType
using Test
using StatsBase: sample

@testset verbose=true "match_coordinates" begin
    N = 20
    us = Coords.([rand(1, N), rand(2, N), rand(3, N)])
    p = sample(1:N, N; replace = false)
    for u in us
        p1, p2 = match_coords(u, u[p])
        @test u[p2] == u[p]
        @test u[p][p1] == u

        u_noisy = [coord .+ 1e-13 * randn() for coord in u][p]
        p1, p2 = match_coords(u, u_noisy)
        @test isapprox(u[p2], u_noisy)
        @test isapprox(u_noisy[p1], u)
    end
end

