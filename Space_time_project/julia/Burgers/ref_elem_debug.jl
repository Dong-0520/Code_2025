using LinearAlgebra, BlockArrays, SparseArrays
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
include("../src/tensor_product_sbp.jl")
include("../src/cg_triangle.jl")


xt, yt, Pt, Qxt, Qyt, Ext, Eyt = cg_triangle.triangle_from_Jesse(1)
n = size(xt)[1]
quadξ, quadη, quadH, quadQξ, quadQη, quadEξ, quadEη = cg_triangle.quad_from_triangle_SBP(n, xt, yt, Pt, Qxt, Qyt, Ext, Eyt)


p1 = scatter(quadξ, quadη, label = "Jesse's triangle")
for (i, (xi, yi)) in enumerate(zip(quadξ, quadη))
    annotate!(p1, xi, yi, text(string(i), :left, 15))
end
p1





p2 = scatter(refTri.rst[1,:], refTri.rst[2,:], label = "Hicken's triangle")
for (i, (xi, yi)) in enumerate(zip(refTri.rst[1,:], refTri.rst[2,:]))
    annotate!(p2, xi, yi, text(string(i), :left, 10))
end
p2