using DifferentialEquations, LinearAlgebra, BlockArrays, SparseArrays
using Trixi
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics                                                                                                                                                                                                                                                                          

include("../src/SBPLite.jl")
using .SBPLite
include("plotting_helper.jl")

using Test


@testset for order in 2:5
    tol = 1e-12
    ref = QuadRefElemGL(order)
    r = ref.rst[1,:]
    s = ref.rst[2,:]
    y = 1 .+ r .+ s .+ r .* (s .^ (order - 1))
    dydr = 1 .+ s .^ (order - 1)
    dyds = 1 .+ (order - 1) * r .* (s.^(order - 2))
    @test isapprox(ref.D[1] * y, dydr, atol = tol)
    @test isapprox(ref.D[2] * y, dyds, atol = tol)
end
r = refTri.rst[1,:]
s = refTri.rst[2,:]
y = 1 .+ r .+ s .+ r .* (s .^ (order - 1))

r = ref.rst[1,:]
s = ref.rst[2,:]
y = 1 .+ r .+ s .+ r .* (s .^ (order - 1))
dydr = 1 .+ s .^ (order - 1)
dyds = 1 .+ (order - 1) * r .* (s.^(order - 2))
maximum(abs.(ref.D[1] * y - dydr))
maximum(abs.(ref.D[2] * y - dyds))

r = ref_test.rst[1,:]
s = ref_test.rst[2,:]
y = 1 .+ r .+ s .+ r .* (s .^ (order - 1))
dydr = 1 .+ s .^ (order - 1)
dyds = 1 .+ (order - 1) * r .* (s.^(order - 2))
maximum(abs.(ref_test.D[1] * y - dydr))
maximum(abs.(ref_test.D[2] * y - dyds))




maximum(abs.(ref.D[1] - ref_test.D[1]))
maximum(abs.(ref.D[2] - ref_test.D[2]))
maximum(abs.(ref.rst - ref_test.rst))
maximum(abs.(ref.rst_q - ref_test.rst_q))
maximum(abs.(ref.rst_f - ref_test.rst_f))
maximum(abs.(ref.f_mask[1] - ref_test.f_mask[1]))
maximum(abs.(ref.f_mask[2] - ref_test.f_mask[2]))
maximum(abs.(ref.f_mask[3] - ref_test.f_mask[3]))
maximum(abs.(ref.f_mask[4] - ref_test.f_mask[4]))
maximum(abs.(ref.H - ref_test.H))
maximum(abs.(ref.H_inv - ref_test.H_inv))
maximum(abs.(ref.H_face - ref_test.H_face))
maximum(abs.(ref.Q[1] - ref_test.Q[1]))
maximum(abs.(ref.Q[2] - ref_test.Q[2]))
maximum(abs.(ref.E[1] - ref_test.E[1]))
maximum(abs.(ref.E[2] - ref_test.E[2]))
maximum(abs.(ref.Qt[1] - ref_test.Qt[1]))
maximum(abs.(ref.Qt[2] - ref_test.Qt[2]))
maximum(abs.(ref.Qt_inv - ref_test.Qt_inv))
maximum(abs.(ref.R[1] - ref_test.R[1]))
maximum(abs.(ref.R[2] - ref_test.R[2]))
maximum(abs.(ref.R[3] - ref_test.R[3]))
maximum(abs.(ref.R[4] - ref_test.R[4]))
maximum(abs.(ref.n_rst - ref_test.n_rst))