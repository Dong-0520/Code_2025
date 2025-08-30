# 在这个文档种，我们尝试理解如何给定一个element，构建一个高阶的 SBP operator 以得到 enriched residual
# 好像做不到这种事情

using Pkg

using LinearAlgebra, SparseArrays, Trixi, BlockArrays
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics        

include("../src/SBPLite.jl")
using .SBPLite
@load joinpath(@__DIR__, "lin_adv_grid__50.jld2") grid




cell_id = 1
cell1 = grid.cells[cell_id]
ref_elem = cell1.ref_data[]