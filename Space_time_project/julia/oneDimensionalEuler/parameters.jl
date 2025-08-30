
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
include("src/myGrid.jl")
include("src/plot_helper.jl")
include("src/contributions.jl")
include("src/dissipation.jl")
include("src/shock_speed.jl")
include("src/pseudo_transient.jl")
include("src/nonlinear_solver.jl")
γ = 1.4

ρL = 3.0
uL = 2.0
pL = 10.0
ρR = 3.0
uR = -1.0
pR = 5.0
ρstarL = 5.013935083865458
ρstarR = 7.676714537106275
ustar = 0.7943931077390836
pstar = 20.855902598748525
EL = pL / (γ - 1) + 0.5 * ρL * uL^2
EstarL = pstar / (γ - 1) + 0.5 * ρstarL * ustar^2
EstarR = pstar / (γ - 1) + 0.5 * ρstarR * ustar^2
ER = pR / (γ - 1) + 0.5 * ρR * uR^2

order = 2
ref = TriangleDiagELG(order, 2 * order)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)

S1 = one_wave_speed(ρL, uL, pL, pstar)
S2 = ustar
S3 = three_wave_speed(ρR, uR, pR, pstar)

# grid = read_mesh(joinpath(@__DIR__, "Euler_mesh/Euler_small_grid__12.msh"), ref_elems_data, Base.identity)
# isfile(joinpath(@__DIR__, "Euler_mesh/Euler_small_grid__12.msh"))

@load joinpath(@__DIR__, "Euler_mesh/Euler_grid_test2_12.jld2") grid # 已经work的case，mesh size较小



analytic_W = zeros(n_cells(grid), length(grid.xyz[1]), 3) # ρ = 3, u = 2, p = 10
for cell_id in 1:n_cells(grid)


    xyzk = grid.xyz[cell_id]
    xk = [x[1] for x in xyzk]
    yk = [x[2] for x in xyzk]
    # for all element on the left of S1
    if all(yk .< (1/S1) .* xk .+ 1e-5)
        analytic_W[cell_id, :,1] .= ρL
        analytic_W[cell_id, :,2] .= uL
        analytic_W[cell_id, :,3] .= pL
    end
    if all(yk .> (1/S1) .* xk .- 1e-5) && all(yk .> (1/S2) .* xk .- 1e-5)
        analytic_W[cell_id, :,1] .= ρstarL
        analytic_W[cell_id, :,2] .= ustar
        analytic_W[cell_id, :,3] .= pstar
    end
    if all(yk .< (1/S2) .* xk .+ 1e-5) && all(yk .> (1/S3) .* xk .- 1e-5)
        analytic_W[cell_id, :,1] .= ρstarR
        analytic_W[cell_id, :,2] .= ustar
        analytic_W[cell_id, :,3] .= pstar
    end
    if all(yk .< (1/S3) .* xk .+ 1e-5)
        analytic_W[cell_id, :,1] .= ρR
        analytic_W[cell_id, :,2] .= uR
        analytic_W[cell_id, :,3] .= pR
    end

end

analytic_U = zeros(size(analytic_W))
analytic_U[:, :, 1] = deepcopy(analytic_W[:, :, 1])
analytic_U[:, :, 2] = analytic_W[:, :, 2] .* analytic_W[:, :, 1]
analytic_U[:, :, 3] = analytic_W[:, :, 3] ./ 0.4 + 0.5 * analytic_W[:, :, 1] .* analytic_W[:, :, 2].^2

# plot_one_variable(grid, analytic_W; xlimit = [-0.51, 0.51], ylimit = [-0.01, 0.151], size = (2400, 800), variable = 3)

newGrid = MyGrid(grid; num_of_variables = 3)
IC = set_IC(grid; WL = [3, 2, 10], WR = [3, -1, 5], γ = 1.4)

# U0 = deepcopy(analytic_U) .* 0 .+ 2

WBCL = zeros(Float64, size(analytic_U[1,:,:]))
WBCL[:, 1] .= 3.
WBCL[:, 2] .= 2.
WBCL[:, 3] .= 10.
FBCL = evaluate_Fx(WBCL)
WBCR = zeros(Float64, size(analytic_U[1,:,:]))
WBCR[:, 1] .= 3.
WBCR[:, 2] .= -1.
WBCR[:, 3] .= 5.
FBCR = evaluate_Fx(WBCR)
