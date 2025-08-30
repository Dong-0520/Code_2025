using LinearAlgebra, SparseArrays
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics
using SparseConnectivityTracer, ADTypes
using NonlinearSolve     
using Polynomials                                                                                                                                                                                                                                                     

include("../../../src/SBPLite.jl")
using .SBPLite
include("../../plotting_helper.jl")

order = 2
ref = TriangleDiagELGL(order, 2 * order - 1)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)

# check BurgerMesh_SIAM_const.jl for parameters
a = 3/2
x_star =  0.125
grid = read_mesh(joinpath(@__DIR__, "Burgers_const_12.msh"), ref_elems_data, Base.identity)

analytic_u = analytic_sol(grid)
# check analytic solution is coorectly stored
plotlyjs()
p = scatter3d()
@inbounds for i in 1:n_cells(grid)
    coords = grid.xyz_q[i]
    x = [coords[j][1] for j in 1:length(coords)]
    y = [coords[j][2] for j in 1:length(coords)]
    # scatter3d!(p, x, y, zcolor = analytic_u[i, :], markersize = 1, zlims = [0.0, 2.0], label = "")
    scatter3d!(p, x, y, analytic_u[i, :], markersize = 1, zlims = [0.0, 2.0], label = "", color = :viridis)
end
p
gr()

zlimit = [0.5, 2.5]
nodes_per_cell = length(grid.xyz_q[1])
interfaces_align_shock = find_interfaces_align_shock(grid)
p1 = (grid, analytic_sol(grid), interfaces_align_shock)

U0 = initial_guess(grid, uL = 1.5, uR = 1.5)
all_U, diffs = space_time_solve(U0, p1, pseudo_dt = 5.0, num_of_pseudo_time_step = 1000)

U0 = all_U[argmin(diffs)]
all_U, diffs = space_time_solve(U0, p1, pseudo_dt = 7.0, num_of_pseudo_time_step = 1000)

x_coords = get_xCoords_for_plotting_sol(grid)
true_u = get_final_u_for_plotting_sol(grid, analytic_u)
num_u = get_final_u_for_plotting_sol(grid, all_U[argmin(diffs)])
Pnorms = get_Pnorm_for_finalTimeInterface(grid)


L2_error = compute_L2_error_atFianlInterface(true_u, num_u, Pnorms)

# order 2 
# mesh 5 -> 3.2871378391903464e-13
# mesh 8 -> 1.042813428207084e-13
# mesh 10 -> 1.3523088660803586e-13
# mesh 12 -> 6.083874608755199e-13

grid5 = read_mesh(joinpath(@__DIR__, "Burgers_const_5.msh"), ref_elems_data, Base.identity)
grid8 = read_mesh(joinpath(@__DIR__, "Burgers_const_8.msh"), ref_elems_data, Base.identity)
grid10 = read_mesh(joinpath(@__DIR__, "Burgers_const_10.msh"), ref_elems_data, Base.identity)
grid12 = read_mesh(joinpath(@__DIR__, "Burgers_const_12.msh"), ref_elems_data, Base.identity)

dxs_ave_length = [get_average_length_of_sides(grid5), get_average_length_of_sides(grid8), get_average_length_of_sides(grid10), get_average_length_of_sides(grid12)]
errors = [3.2871378391903464e-13, 1.042813428207084e-13, 1.3523088660803586e-13, 6.083874608755199e-13]
fit(log10.(dxs_ave_length[1:3]), log10.(errors[1:3]),1)[1]




plot(diffs)
gif(observe_iteration(grid, all_U[1:20:end], zlimit, show_true_sol = false), joinpath(@__DIR__, "numerical_test_discts.gif"), fps=5)
argmin(diffs), minimum(diffs)
u1 = all_U[argmin(diffs)]
observe_numerical_sol(grid, u1, zlimit, show_true_sol = false)




using Plots

using Plots
using Colors  # 用于颜色映射


function plot_numerical_sol_const(grid, u; xlim=(0.0, 1.0), ylim=(0.0, 0.5), width=1000, height=500, ulim = (1, 2))
    num_of_cells = n_cells(grid)
    p = plot(size=(width, height), xlims=xlim, ylims=ylim, colorbar=true, clims=ulim, xlabel = "x", ylabel = "t", left_margin=5Plots.mm, bottom_margin=5Plots.mm)  # <-- 设置 colorbar 范围 (1,2)


    triangulation = []
    values = []

    for cell_id in 1:num_of_cells
        vertices_IDS = vertices(grid.cells[cell_id])
        vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
        vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
        vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

        v1x, v1y = vertice1[1], vertice1[2]
        v2x, v2y = vertice2[1], vertice2[2]
        v3x, v3y = vertice3[1], vertice3[2]

        # 计算该单元数值解的平均值
        u_avg = mean(u[cell_id, :])
        push!(values, u_avg)

        # 存储三角形信息
        push!(triangulation, ([v1x, v2x, v3x], [v1y, v2y, v3y]))
    end

    cmap = cgrad(:viridis)  # 选择颜色方案

    # 画出填充三角形
    for (tri, val) in zip(triangulation, values)
        plot!(p, Shape(tri[1], tri[2]), lw=0.5, linecolor=:black, fill_z=fill(val, 3), fillalpha=0.8, c=:viridis, label=false)
    end

    return p
end


sol_plot = plot_numerical_sol_const(grid, all_U[end], height = 400, ylim = [0, 0.5])
save(joinpath(@__DIR__, "numerical_test_discts_const.png"), sol_plot)


# ----------------- run these functins ----------------- #

function analytic_sol(grid::Grid; tol = 1e-10, a = a, x_star = x_star)

    result = zeros((n_cells(grid), length(grid.xyz_q[1])))
    @inbounds for i in 1:n_cells(grid)
        curr_coords = grid.xyz_q[i]
        # if all(coords -> coords[1] < a * coords[2] + x_star, curr_coords)
        if  all(coords -> coords[1] <= a * coords[2] + x_star + tol, curr_coords)
            result[i, :] .= 2.0
        else
            result[i, :] .= 1.0
        end
    end
    return result
    
end

function mysign(x)
    x >= 0 ? 1 : -1
end

function RHS_forSolver_diss(du, u, p)
    grid, analytic_u = p

    # volume discretization
    @inbounds Threads.@threads for cell_id in 1:n_cells(grid)
        
        Dx = grid.VOL[cell_id][1]
        Dt = grid.VOL[cell_id][2]
        

        local_sol = u[cell_id, :]
        ux = 1/3 * Dx * (local_sol .^ 2) .+ 1/3 * local_sol .* (Dx * local_sol)
        ut = Dt * local_sol 

        du[cell_id, :] = (-ux .- ut)
    end

    @inbounds Threads.@threads for i in 1:length(grid.face_interfaces)
        interface = grid.face_interfaces[i]
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        refk = grid.cells[ck].ref_data[]
        refv = grid.cells[cv].ref_data[]
        Rγk, Rγv = Matrix(refk.R[lfγk]), Matrix(refv.R[lfγv])

        uk, uv = u[ck, :], u[cv, :]
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        # normal vector of physical element
        maskγk, maskγv = @views refk.f_mask[lfγk], refv.f_mask[lfγv]
        normalγk, normalγv = @views grid.geometric_terms.N_f[ck][:, maskγk], grid.geometric_terms.N_f[cv][:, maskγv]
        Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
        Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]

    
        # 替代布尔判断的代码
        indicator = Nxγk .* (uk_face - uk_adj)
        spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4 .* (indicator .>= 0) + ((uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 .+ mysign.(Nxγk) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)) .* (indicator .< 0)

        rk_spatial = grid.FAC[ck][lfγk] * Diagonal(Nxγk) * (0.5 .* uk_face.^2 .- spatial_flux_k)
        rv_spatial = grid.FAC[cv][lfγv] * Diagonal(Nxγv) * (0.5 .* uv_face.^2 .- spatial_flux_k[Pk])


        rk_temporal = grid.FAC[ck][lfγk] * Diagonal(Ntγk) * (0.5 .* uk_face .- 0.5 .* uk_adj)
        rv_temporal = grid.FAC[cv][lfγv] * Diagonal(Ntγv) * (0.5 .* uv_face .- 0.5 .* uv_adj)

        
        du[ck, :] += ( rk_spatial .+ rk_temporal )
        du[cv, :] += ( rv_spatial .+ rv_temporal )
    end

    # initial condition
    bottom_inflow1 = collect(get_face_set(grid, "BOTTOM_INFLOW_1"))
    bottom_inflow2 = collect(get_face_set(grid, "BOTTOM_INFLOW_2"))
    @inbounds Threads.@threads for i in 1:length(bottom_inflow1)
        cell_id, face_id = bottom_inflow1[i]
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        # Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = R * analytic_u[cell_id, :]

        du[cell_id, :] += (grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj))
    end

    @inbounds Threads.@threads for i in 1:length(bottom_inflow2)
        cell_id, face_id = bottom_inflow2[i]
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = R * analytic_u[cell_id, :]

        du[cell_id, :] += (grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj))
    end

    left_inflow = collect(get_face_set(grid, "LEFT_INFLOW"))
    @inbounds Threads.@threads for i in 1:length(left_inflow)
        cell_id, face_id = left_inflow[i]
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = R * analytic_u[cell_id, :]

        FAC = grid.FAC[cell_id][face_id] * Diagonal(Nxγ)
        inflow = ((u_face + abs.(u_face)) .* u_face)./3 - (u_adj + abs.(u_adj)) .* u_adj ./ 3

        du[cell_id, :] += FAC * inflow
    end

    right_inflow = collect(get_face_set(grid, "RIGHT_INFLOW"))
    @inbounds Threads.@threads for i in 1:length(right_inflow)
        cell_id, face_id = right_inflow[i]
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = R * analytic_u[cell_id, :]

        FAC = grid.FAC[cell_id][face_id] * Diagonal(Nxγ)
        inflow = ((u_face - abs.(u_face)) .* u_face)./3 - (u_adj - abs.(u_adj)) .* u_adj ./ 3

        du[cell_id, :] += FAC * inflow

        # result[cell_id, :] += (grid.FAC[cell_id][face_id] * Diagonal(Nxγ) * ((u_face - abs.(u_face)) .* u_face)./3 - (u_adj - abs.(u_adj)) .* u_adj ./ 3)
    end

    return nothing

end


function space_time_RK4(U, t, pseudo_dt, p)
    c1, c2, c3, c4 = 0.0, 0.40128709, 0.56449983, 0.87678807
    b1, b2, b3, b4 = 0.20334721, 0.19932974, 0.28339585, 0.31392720
    a21 = 0.40128709
    a31, a32 = 0.28224991, 0.28224991
    a41, a42, a43 = 0.25972925, 0.25479937, 0.36225945

    # k1 = RHS_test(U, p)
    # k2 = RHS_test(U + 0.5 * pseudo_dt * k1, p)
    # k3 = RHS_test(U + 0.5 * pseudo_dt * k2, p)
    # k4 = RHS_test(U + pseudo_dt * k3, p)
    # return U + (pseudo_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    k1 = RHS_test(U, p)
    k2 = RHS_test(U + a21 * pseudo_dt * k1, p)
    k3 = RHS_test(U + a31 * pseudo_dt * k1 + a32 * pseudo_dt * k2, p)
    k4 = RHS_test(U + a41 * pseudo_dt * k1 + a42 * pseudo_dt * k2 + a43 * pseudo_dt * k3, p)
    return U + pseudo_dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4)

end

function space_time_solve(U0::Array{Float64, 2}, p::Any; pseudo_dt = 1.0, num_of_pseudo_time_step = 100)
    num_of_cells = n_cells(p[1])
    num_of_nodes = Int(length(p[1].xyz_q[1]))
    all_U = Vector{Matrix{Float64}}()
    push!(all_U, U0)
    diffs = Vector{Float64}()

    for pseudo_step in 1:num_of_pseudo_time_step
        curr_U = deepcopy(all_U[pseudo_step])
        next_U = space_time_RK4(curr_U, 0.0, pseudo_dt, p)
        max_diff = maximum(abs.(next_U - curr_U))
        push!(all_U, next_U)
        push!(diffs, max_diff)

        if max_diff < 1e-12
            println("\n Solution converged at pseudo time step: $pseudo_step \n")
            return all_U, diffs
        elseif max_diff > 100
            println("Solution diverges at pseudo time step: $pseudo_step \n")
            return all_U, diffs
        else
            println("Pseudo time step: $pseudo_step, max diff: ", max_diff, "\n")
        end
    end
    print("\n Solution not converged, pseudo_dt = $pseudo_dt \n")
    return all_U, diffs
end


function initial_guess(grid; uL = 2.0, uR = 1.0, tol = 1e-10, a = a, x_star = x_star)
    result = zeros((n_cells(grid), length(grid.xyz_q[1])))
    @inbounds for i in 1:n_cells(grid)
        curr_coords = grid.xyz_q[i]
        # if all(coords -> coords[1] < a * coords[2] + x_star, curr_coords)
        if  all(coords -> coords[1] <= a * coords[2] + x_star + tol, curr_coords)
            result[i, :] .= uL
        else
            result[i, :] .= uR
        end
    end
    return result
    
end

function find_interfaces_align_shock(grid)
    interfaces_set = Set{FaceInterface}()
    for (cell_id, face_id) in get_face_set(grid, "SHOCK")
        neighbor_id, neighbor_face_id = Tuple(findall(x -> x == FaceIndex((cell_id, face_id)), grid.topology.face_face_neighbours)[1])
        interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((neighbor_id, neighbor_face_id)))
        push!(interfaces_set, interface)
    end
    return interfaces_set
end

function RHS_test(u, p)
    grid, analytic_u, interfaces_align_shock = p
    result = zeros(n_cells(grid), nodes_per_cell)

    # volume discretization
    Threads.@threads for cell_id in 1:n_cells(grid)
        
        Dx = grid.VOL[cell_id][1]
        Dt = grid.VOL[cell_id][2]
        

        local_sol = u[cell_id, :]
        ux = 1/3 * Dx * (local_sol .^ 2) .+ 1/3 * local_sol .* (Dx * local_sol)
        ut = Dt * local_sol 

        result[cell_id, :] = grid.geometric_terms.J_q[cell_id] .* (-ux .- ut)
        # result[cell_id, :] = (-ux .- ut)
    end

    # non_align_shock_interfaces = collect(setdiff(Set(grid.face_interfaces), interfaces_align_shock))
    # Threads.@threads for i in 1:length(non_align_shock_interfaces)
    #     interface = non_align_shock_interfaces[i]
    Threads.@threads for i in 1:length(grid.face_interfaces)
        interface = grid.face_interfaces[i]
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        refk = grid.cells[ck].ref_data[]
        refv = grid.cells[cv].ref_data[]
        Rγk, Rγv = Matrix(refk.R[lfγk]), Matrix(refv.R[lfγv])

        uk, uv = u[ck, :], u[cv, :]
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        # normal vector of physical element
        maskγk, maskγv = @views refk.f_mask[lfγk], refv.f_mask[lfγv]
        normalγk, normalγv = @views grid.geometric_terms.N_f[ck][:, maskγk], grid.geometric_terms.N_f[cv][:, maskγv]
        Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
        Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]

    
        # 替代布尔判断的代码

        indicator = Nxγk .* (uk_face - uk_adj)
        spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4 .* (indicator .>= 0) + ((uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 .+ mysign.(Nxγk) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)) .* (indicator .< 0)
        rk_spatial = grid.FAC[ck][lfγk] * Diagonal(Nxγk) * (0.5 .* uk_face.^2 .- spatial_flux_k)
        rv_spatial = grid.FAC[cv][lfγv] * Diagonal(Nxγv) * (0.5 .* uv_face.^2 .- spatial_flux_k[Pk])


        rk_temporal = grid.FAC[ck][lfγk] * Diagonal(Ntγk) * (0.5 .* uk_face .- 0.5 .* uk_adj)
        rv_temporal = grid.FAC[cv][lfγv] * Diagonal(Ntγv) * (0.5 .* uv_face .- 0.5 .* uv_adj)

        
        result[ck, :] += grid.geometric_terms.J_q[ck] .* ( rk_spatial .+ rk_temporal )
        result[cv, :] += grid.geometric_terms.J_q[cv] .* ( rv_spatial .+ rv_temporal )
        # result[ck, :] += ( rk_spatial .+ rk_temporal )
        # result[cv, :] += ( rv_spatial .+ rv_temporal )
    end

    # initial condition
    bottom_inflow1 = collect(get_face_set(grid, "BOTTOM_INFLOW_1"))
    bottom_inflow2 = collect(get_face_set(grid, "BOTTOM_INFLOW_2"))
    @inbounds Threads.@threads for i in 1:length(bottom_inflow1)
        cell_id, face_id = bottom_inflow1[i]
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        # Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = R * analytic_u[cell_id, :]

        result[cell_id, :] += grid.geometric_terms.J_q[cell_id] .* (grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj))
        # result[cell_id, :] += (grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj))
    end

    @inbounds Threads.@threads for i in 1:length(bottom_inflow2)
        cell_id, face_id = bottom_inflow2[i]
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = R * analytic_u[cell_id, :]

        result[cell_id, :] += grid.geometric_terms.J_q[cell_id] .* (grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj))
        # result[cell_id, :] += (grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj))
    end

    left_inflow = collect(get_face_set(grid, "LEFT_INFLOW"))
    @inbounds Threads.@threads for i in 1:length(left_inflow)
        cell_id, face_id = left_inflow[i]
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = R * analytic_u[cell_id, :]

        FAC = grid.geometric_terms.J_q[cell_id] .* grid.FAC[cell_id][face_id] * Diagonal(Nxγ)
        # FAC = grid.FAC[cell_id][face_id] * Diagonal(Nxγ)
        inflow = ((u_face + abs.(u_face)) .* u_face)./3 - (u_adj + abs.(u_adj)) .* u_adj ./ 3

        result[cell_id, :] += FAC * inflow
    end

    right_inflow = collect(get_face_set(grid, "RIGHT_INFLOW"))
    @inbounds Threads.@threads for i in 1:length(right_inflow)
        cell_id, face_id = right_inflow[i]
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = R * analytic_u[cell_id, :]

        FAC = grid.geometric_terms.J_q[cell_id] .* grid.FAC[cell_id][face_id] * Diagonal(Nxγ)
        # FAC = grid.FAC[cell_id][face_id] * Diagonal(Nxγ)
        inflow = ((u_face - abs.(u_face)) .* u_face)./3 - (u_adj - abs.(u_adj)) .* u_adj ./ 3

        result[cell_id, :] += FAC * inflow
    end

    return result
end


function get_sidelength_perCell(grid::Grid, cell::AbstractCell)

    vertices_IDS = vertices(cell)
    vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
    vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
    vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

    v1x = vertice1[1]
    v1y = vertice1[2]
    v2x = vertice2[1]
    v2y = vertice2[2]
    v3x = vertice3[1]
    v3y = vertice3[2]

    l1 = sqrt((v1x - v2x)^2 + (v1y - v2y)^2)
    l2 = sqrt((v2x - v3x)^2 + (v2y - v3y)^2)
    l3 = sqrt((v3x - v1x)^2 + (v3y - v1y)^2)

    return l1, l2, l3
    
end

function get_average_length_of_sides(grid)
    num_of_cells = n_cells(grid)
    total_length = 0.0
    for i in 1:num_of_cells
        l1, l2, l3 = get_sidelength_perCell(grid, grid.cells[i])
        total_length += l1 + l2 + l3
    end
    return total_length / (3 * num_of_cells)

end

function get_xCoords_for_plotting_sol(grid)
    """
    this function return the x coordinates for plotting numerical solution at each final time of slabs
    """
    TOP_INFLOW = collect(union(get_face_set(grid, "TOP_INFLOW_1"), get_face_set(grid, "TOP_INFLOW_2")))
    ref = grid.cells[TOP_INFLOW[1][1]].ref_data[]
    coords = Dict()
    for (cell_id, face_id) in TOP_INFLOW
        R = ref.R[face_id]
        # mask = ref.f_mask[face_id]
        x = map(x -> x[1], R * grid.xyz_q[cell_id])
        coords[cell_id] = x
    end
    return coords

end

function get_final_u_for_plotting_sol(grid, u)
    TOP_INFLOW = collect(union(get_face_set(grid, "TOP_INFLOW_1"), get_face_set(grid, "TOP_INFLOW_2")))
    ref = grid.cells[TOP_INFLOW[1][1]].ref_data[]
    num_sol = Dict()
    for (cell_id, face_id) in TOP_INFLOW
        R = ref.R[face_id]
        temp_u = map(x -> x[1], R * u[cell_id, :])
        num_sol[cell_id] = temp_u
    end
    return num_sol
    
end

function get_Pnorm_for_finalTimeInterface(grid)
    """
    works for straight interfaces
    """
    TOP_INFLOW = collect(union(get_face_set(grid, "TOP_INFLOW_1"), get_face_set(grid, "TOP_INFLOW_2")))
    ref = grid.cells[TOP_INFLOW[1][1]].ref_data[]
    Pnorm = Dict()
    for (cell_id, face_id) in TOP_INFLOW
        R = ref.R[face_id]
        face_coords = R * grid.xyz_q[cell_id]
        @assert face_coords[1][2] ≈ face_coords[end][2]
        Pnorm[cell_id] = (abs(face_coords[1][1] - face_coords[end][1]) / 2) * ref.H_face
        @assert sum(diag(Pnorm[cell_id])) ≈ abs(face_coords[1][1] - face_coords[end][1])
    end
    return Pnorm
end

function compute_L2_error_atFianlInterface(analytic_u, num_u, Pnorms)
    error = 0.0
    for cell_id in collect(keys(analytic_u))
        local_error = analytic_u[cell_id] - num_u[cell_id]
        error += local_error' * Pnorms[cell_id] * local_error
    end
    return sqrt(error)
    
end