
# 这些代码被放在每个文件夹里的local_parameter了
# using Pkg

# using LinearAlgebra, SparseArrays, Trixi, BlockArrays
# using JLD2
# using Plots
# using DoubleFloats
# using Logging
# using ProgressBars
# using LoopVectorization, ArraysOfArrays
# using Statistics        

# include("../../../src/SBPLite.jl")
# using .SBPLite
# # include("myGrid.jl")
# include("plot_helper.jl")
# include("plot_helper_grid.jl")
# include("shock_speed.jl")

# @load joinpath(@__DIR__, "../Euler_grid_10_nonalign.jld2") grid


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


S1 = one_wave_speed(ρL, uL, pL, pstar)
S2 = ustar
S3 = three_wave_speed(ρR, uR, pR, pstar)

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
analytic_W[5, :, 1] .= ρstarR
analytic_W[6, :, 1] .= ρstarL
analytic_W[[5, 6], :, 2] .= ustar
analytic_W[[5, 6], :, 3] .= pstar



W0_for_solver = analytic_W + 0.1 * randn(size(analytic_W))

analytic_U = zeros(size(analytic_W))
analytic_U[:, :, 1] = deepcopy(analytic_W[:, :, 1])
analytic_U[:, :, 2] = analytic_W[:, :, 2] .* analytic_W[:, :, 1]
analytic_U[:, :, 3] = analytic_W[:, :, 3] ./ 0.4 + 0.5 * analytic_W[:, :, 1] .* analytic_W[:, :, 2].^2

U0_for_solver = zeros(size(W0_for_solver))
U0_for_solver[:, :, 1] = deepcopy(W0_for_solver[:, :, 1])
U0_for_solver[:, :, 2] = W0_for_solver[:, :, 2] .* W0_for_solver[:, :, 1]
U0_for_solver[:, :, 3] = W0_for_solver[:, :, 3] ./ 0.4 + 0.5 * W0_for_solver[:, :, 1] .* W0_for_solver[:, :, 2].^2

# plot_one_variable(grid, analytic_W; xlimit = [-0.51, 0.51], ylimit = [-0.01, 0.151], size = (2400, 800), variable = 3)
function set_IC(grid; WL = [3, 2, 10], WR = [3, -1, 5], γ = 1.4)

    IC = Dict()

    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_1")
        temp_u = zeros(length(grid.xyz_q[cell_id]), 3)
        temp_u[:, 1] .= WL[1]
        temp_u[:, 2] .= WL[1] * WL[2]
        temp_u[:, 3] .= WL[3] / (γ - 1) + 0.5  * WL[1] * WL[2]^2
        # IC[(cell_id, face_id)] = temp_u
        # vec(IC[faceindex]')
        # IC[FaceIndex(cell_id, face_id)] = temp_u
        IC[FaceIndex(cell_id, face_id)] = vec(temp_u')
    end
    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_2")
        temp_u = zeros(length(grid.xyz_q[cell_id]), 3)
        temp_u[:, 1] .= WR[1]
        temp_u[:, 2] .= WR[1] * WR[2]
        temp_u[:, 3] .= WR[3] / (γ - 1) + 0.5  * WR[1] * WR[2]^2
        # IC[(cell_id, face_id)] = temp_u
        # IC[FaceIndex(cell_id, face_id)] = temp_u
        IC[FaceIndex(cell_id, face_id)] = vec(temp_u')
    end

    return IC


end

# newGrid = MyGrid(grid; num_of_variables = 3)
IC = set_IC(grid; WL = [3, 2, 10], WR = [3, -1, 5], γ = 1.4)

function evaluate_Fx(Wk::AbstractMatrix{<:Real}; γ = 1.4)

    num_nodes, num_vars = size(Wk)
    result = zeros(eltype(Wk), num_nodes, num_vars)

    ρ = Wk[:, 1]
    u = Wk[:, 2]
    p = Wk[:, 3]

    E = @. p / (γ - 1) + 0.5 * ρ * u^2

    @. result[:, 1] = ρ * u
    @. result[:, 2] = p + ρ * u^2
    @. result[:, 3] = u * (E + p)

    return vec(permutedims(result, (2, 1)))
end

function set_FBC(grid; WL = [3.0, 2.0, 10.0], WR = [3.0, -1.0, 5.0], γ = 1.4, num_of_variables = 3)
    FBC = Dict{FaceIndex, Vector{Float64}}()
    num_of_nodes = length(grid.xyz[1])
    for faceindex in get_face_set(grid, "LEFT_INFLOW")
        WBC = zeros(Float64, (num_of_nodes, num_of_variables))
        WBC[:, 1] .= WL[1]
        WBC[:, 2] .= WL[2]
        WBC[:, 3] .= WL[3]
        FBC[faceindex] = evaluate_Fx(WBC)
    end
    for faceindex in get_face_set(grid, "RIGHT_INFLOW")
        WBC = zeros(Float64, (num_of_nodes, num_of_variables))
        WBC[:, 1] .= WR[1]
        WBC[:, 2] .= WR[2]
        WBC[:, 3] .= WR[3]
        FBC[faceindex] = evaluate_Fx(WBC)
    end

    return FBC
end
# WBCL = zeros(Float64, size(analytic_U[1,:,:]))
# WBCL[:, 1] .= 3.
# WBCL[:, 2] .= 2.
# WBCL[:, 3] .= 10.
# FBCL = evaluate_Fx(WBCL)
# WBCR = zeros(Float64, size(analytic_U[1,:,:]))
# WBCR[:, 1] .= 3.
# WBCR[:, 2] .= -1.
# WBCR[:, 3] .= 5.
# FBCR = evaluate_Fx(WBCR)
FBC = set_FBC(grid)