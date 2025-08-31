
using Pkg

using LinearAlgebra, SparseArrays, Trixi, BlockArrays
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization, ArraysOfArrays
using Statistics        

include("../../../src/SBPLite.jl")
using .SBPLite
# include("myGrid.jl")
include("../src/plot_helper.jl")
include("../src/plot_helper_grid.jl")
include("../src/shock_speed.jl")

# @load joinpath(@__DIR__, "Euler_grid_10_nonalign.jld2") grid
println("Reading ref element... \n")
order = 1
ref = TriangleDiagELG(order, 2 * order)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)
println("Reading mesh... \n")
grid = read_mesh(joinpath(@__DIR__, "Euler_grid_2_nonalign.msh"), ref_elems_data, Base.identity)
# @load joinpath(@__DIR__, "grid_nonalign_order1.jld2") grid
println("Mesh read done. \n")