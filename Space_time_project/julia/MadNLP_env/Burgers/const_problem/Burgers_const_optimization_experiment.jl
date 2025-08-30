# 这个文件是纯跑代码跑实验，要理解代码的具体实现和细节，请查看 lin_adv_optimization_study.jl

using Pkg

using LinearAlgebra, SparseArrays, Trixi, BlockArrays, ArraysOfArrays
using JLD2
# import Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics        
using ForwardDiff


include("../../../src/SBPLite.jl")
# include(joinpath(@__DIR__, "..//plot_helper.jl"))
using .SBPLite

order = 2
ref = TriangleDiagELG(order, 2 * order)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)

grid = read_mesh(joinpath(@__DIR__, "../Burgers_const_10.msh"), ref_elems_data, identity)
@save joinpath(@__DIR__, "Burgers_const_10.jld2") grid
@load joinpath(@__DIR__, "Burgers_const_10.jld2") grid
# ax, at = 1.0, 1.0
# bottomBC1(x::Coord) = -1
# bottomBC2(x::Coord) = 1
x_star = 0.4
a = 1.6
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
function bottomBC(x::Coord; x_star = x_star)
    if x[1] <= x_star
        return 2.0
    else
        return 1.0
    end
 end
leftBC(x::Coord) = 2
rightBC(x::Coord) = 1


Pkg.activate("MadNLP_env")


function mysign(x)
    x >= 0 ? 1 : -1
end


function RHS(du, u, p)
    grid = p

    # volume discretization
    for cell_id in 1:n_cells(grid)

        Dx = grid.VOL[cell_id][1]
        Dt = grid.VOL[cell_id][2]
        

        local_sol = u[cell_id, :]
        ux = 1/3 * Dx * (local_sol .^ 2) .+ 1/3 * local_sol .* (Dx * local_sol)
        ut = Dt * local_sol 

        # result[cell_id, :] = (-ux .- ut)
        du[cell_id, :] = (-ux .- ut)
    end

    for i in 1:length(grid.face_interfaces)
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
        # spatial_flux_k = (uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6
        # spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4 .* (indicator .>= 0) + ((uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 ) .* (indicator .< 0) .+ mysign.(Nxγk) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)

        # 同样的逻辑适用于 spatial_flux_v
        # indicator_v = Nxγv .* (uv_face - uv_adj)
        # spatial_flux_v = (uv_face.^2 + uv_adj.^2) ./ 4 .* (indicator_v .>= 0) + ((uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 .+ mysign.(Nxγv) .* max.(abs.(uv_face), abs.(uv_adj)) .* (uv_face .- uv_adj)) .* (indicator_v .< 0)

        rk_spatial = grid.FAC[ck][lfγk] * Diagonal(Nxγk) * (0.5 .* uk_face.^2 .- spatial_flux_k)
        rv_spatial = grid.FAC[cv][lfγv] * Diagonal(Nxγv) * (0.5 .* uv_face.^2 .- spatial_flux_k[Pk])


        rk_temporal = grid.FAC[ck][lfγk] * Diagonal(Ntγk) * (0.5 .* uk_face .- 0.5 .* uk_adj)
        rv_temporal = grid.FAC[cv][lfγv] * Diagonal(Ntγv) * (0.5 .* uv_face .- 0.5 .* uv_adj)

        
        # result[ck, :] += ( rk_spatial .+ rk_temporal )
        # result[cv, :] += ( rv_spatial .+ rv_temporal )
        du[ck, :] += ( rk_spatial .+ rk_temporal )
        du[cv, :] += ( rv_spatial .+ rv_temporal )
    end



    bottom_faces = collect(union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2")))
    for i in 1:length(bottom_faces)
        cell_id, face_id = bottom_faces[i]
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        # Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = bottomBC.(R * grid.xyz_q[cell_id])

        # result[cell_id, :] += (grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj))
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
        u_adj = ones(Float64, length(u_face)) .+ 1.0 # left boundary condition is zero

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
        u_adj = ones(Float64, length(u_face))  # right boundary condition is zero

        FAC = grid.FAC[cell_id][face_id] * Diagonal(Nxγ)
        inflow = ((u_face - abs.(u_face)) .* u_face)./3 - (u_adj - abs.(u_adj)) .* u_adj ./ 3

        du[cell_id, :] += FAC * inflow

    end

end

function initial_guess(grid)
    nodes_per_cell = length(grid.xyz_q[1])
    U0 = zeros(Float64, n_cells(grid), nodes_per_cell)
    for cell_id in 1:n_cells(grid)
        for node_id in 1:nodes_per_cell
            U0[cell_id, node_id] = analytic_u(grid.xyz_q[cell_id][node_id])
        end
    end
    return U0
    
end


# ---------------------- slab 1 ---------------------- #
using NonlinearSolve, LineSearches, ADTypes, SparseConnectivityTracer, SparseDiffTools, NLsolve
U0_for_solver = analytic_sol(grid, a = a, x_star = x_star)

u0_vec = vec(U0_for_solver)
du = similar(U0_for_solver); fill!(du, 0.0)


# --- 构造 in-place 残差 ---
# 用 NLsolve 来解

function f_nl!(F, u)
    U = reshape(u, size(U0_for_solver))
    du = similar(U); fill!(du, 0.0)
    f_forSolver!(du, U, para)      # 你的核心残差
    F[:] = vec(du)
end

function f_forSolver!(du, u, p)
    # du must be pre-allocated to the same shape as u
    # p is your packed parameters tuple (grid, newGrid, IC, FBCL, FBCR)
    RHS(du, u, p)
    return nothing
end

para = grid

res = nlsolve(
  f_nl!, u0_vec;
  method    = :trust_region,
  linsolve  = :dogleg,      # :cg
  autodiff  = :central,   
  xtol      = 1e-6,
  ftol      = 1e-6,
  autoscale = true,
  show_trace= true,
  iterations = 10000,
)

oscillating_u = reshape(res.zero, size(U0_for_solver))
using Plots
# plot_u_2D(grid, num_u)
include(joinpath(@__DIR__, "../plot_helper_grid.jl"))
# plot_u_interactive(grid, reshape(u0_vec, size(U0_for_solver)))
plot_u_interactive(grid, oscillating_u)
plot_mesh(grid)
@save joinpath(@__DIR__, "oscillating_solution_const_1coord.jld2") oscillating_u grid

@load joinpath(@__DIR__, "oscillating_solution_const_1coord.jld2") oscillating_u grid

using JuMP, MadNLP, Random, MadNLPHSL
# @load joinpath(@__DIR__, "lin_adv_oscillating_solution_sin.jld2") oscillating_u grid

include(joinpath(@__DIR__, "../../src_simpleGrid/SimpleGrid.jl"))

function RHS_for_solution(du, u, para)
    grid, a = para
    ref = grid.ref

    # volume discretization
    for cell_id in 1:length(grid.cells)
        J_k = Diagonal(grid.geometric_terms.J_q[cell_id])

        Dx = grid.VOL[cell_id][1]
        Dt = grid.VOL[cell_id][2]
        

        local_sol = u[cell_id, :]
        ux = 1/3 * Dx * (local_sol .^ 2) .+ 1/3 * local_sol .* (Dx * local_sol)
        ut = Dt * local_sol 

        # result[cell_id, :] = (-ux .- ut)
        du[cell_id, :] = J_k * (-ux .- ut)
    end

    for i in 1:length(grid.face_interfaces)
        interface = grid.face_interfaces[i]
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        # refk = grid.cells[ck].ref_data[]
        # refv = grid.cells[cv].ref_data[]
        Rγk, Rγv = Matrix(ref.R[lfγk]), Matrix(ref.R[lfγv])

        uk, uv = u[ck, :], u[cv, :]
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        # normal vector of physical element
        maskγk, maskγv = @views ref.f_mask[lfγk], ref.f_mask[lfγv]
        normalγk, normalγv = @views grid.geometric_terms.N_f[ck][:, maskγk], grid.geometric_terms.N_f[cv][:, maskγv]
        Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
        Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]

    
        # 替代布尔判断的代码
        indicator = Nxγk .* (uk_face - uk_adj)
        spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4 .* (indicator .>= 0) + ((uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 .+ mysign.(Nxγk) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)) .* (indicator .< 0)
        # spatial_flux_k = (uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6
        # spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4 .* (indicator .>= 0) + ((uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 ) .* (indicator .< 0) .+ mysign.(Nxγk) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)

        # 同样的逻辑适用于 spatial_flux_v
        # indicator_v = Nxγv .* (uv_face - uv_adj)
        # spatial_flux_v = (uv_face.^2 + uv_adj.^2) ./ 4 .* (indicator_v .>= 0) + ((uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 .+ mysign.(Nxγv) .* max.(abs.(uv_face), abs.(uv_adj)) .* (uv_face .- uv_adj)) .* (indicator_v .< 0)

        rk_spatial = ref.H_inv * Matrix(Rγk)' * ref.H_face *  Diagonal(Nxγk) * (0.5 .* uk_face.^2 .- spatial_flux_k)
        rv_spatial = ref.H_inv * Matrix(Rγv)' * ref.H_face * Diagonal(Nxγv) * (0.5 .* uv_face.^2 .- spatial_flux_k[Pk])


        rk_temporal = ref.H_inv * Matrix(Rγk)' * ref.H_face * Diagonal(Ntγk) * (0.5 .* uk_face .- 0.5 .* uk_adj)
        rv_temporal = ref.H_inv * Matrix(Rγv)' * ref.H_face * Diagonal(Ntγv) * (0.5 .* uv_face .- 0.5 .* uv_adj)

        # result[ck, :] += ( rk_spatial .+ rk_temporal )
        # result[cv, :] += ( rv_spatial .+ rv_temporal )
        du[ck, :] += ( rk_spatial .+ rk_temporal )
        du[cv, :] += ( rv_spatial .+ rv_temporal )
    end



    for (cell_id, face_id) in union(grid.face_sets["BOTTOM_INFLOW_1"], grid.face_sets["BOTTOM_INFLOW_2"])
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = bottomBC.(R * grid.xyz_SBP[cell_id])

        # result[cell_id, :] += (grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj))
        du[cell_id, :] += ref.H_inv * Matrix(R)' * ref.H_face * Diagonal(Ntγ) * (u_face .- u_adj)
    end

    for (cell_id, face_id) in grid.face_sets["LEFT_INFLOW"]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Nxγ = normal[1, :]
        # Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = ones(Float64, length(u_face)) .+ 1.0 # left boundary condition is zero

        FAC = ref.H_inv * Matrix(R)' * ref.H_face * Diagonal(Nxγ)
        inflow = ((u_face + abs.(u_face)) .* u_face)./3 - (u_adj + abs.(u_adj)) .* u_adj ./ 3

        du[cell_id, :] += FAC * inflow
    end

    for (cell_id, face_id) in grid.face_sets["RIGHT_INFLOW"]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = ones(Float64, length(u_face))  # right boundary condition is zero

        FAC = ref.H_inv * Matrix(R)' * ref.H_face * Diagonal(Nxγ)
        inflow = ((u_face - abs.(u_face)) .* u_face)./3 - (u_adj - abs.(u_adj)) .* u_adj ./ 3

        du[cell_id, :] += FAC * inflow

    end
end

function wrap_up_results(buffer)
    initial_guess = buffer.initial_guess
    initial_guess[buffer.indices_moving_coords] = buffer.final_solution[1:length(buffer.indices_moving_coords)]

    new_xyz_gmsh = initial_guess[1:buffer.index_mesh]
    new_xyz_gmsh_coords = [Coord{2, Float64}((new_xyz_gmsh[i], new_xyz_gmsh[buffer.index_mesh ÷ 2 + i])) for i in 1:6]
    final_simple_grid = construct_simpleGrid(
        grid.cells, buffer.ref, new_xyz_gmsh_coords, 
        grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
    )
    final_u = reshape(buffer.final_solution[length(buffer.indices_moving_coords)+1:end], (length(grid.VOL), size(grid.VOL[1][1])[1]))

    return final_simple_grid, final_u

end


function pde_model_with_solver_struct(buffer::SpaceTimeBuffer; lin_solver = MadNLPHSL.Ma57Solver, 
                                                                αmesh = 1.0, αshk = 1.0, max_iter = 100)
    # model = Model(() -> MadNLP.Optimizer(
    #     linear_solver=lin_solver,
    #     print_level=MadNLP.INFO,
    #     max_iter=max_iter
    # ))  
    model = Model(() -> MadNLP.Optimizer(
        linear_solver=lin_solver,
        print_level=MadNLP.INFO,
        max_iter=max_iter,                       # 减少最大迭代次数
        
        # 🔥 更严格的收敛条件
        tol=1e-16,                          # 主要容差（从1e-8改为1e-12）
        
        # 🔥 禁用可接受解机制，强制达到严格收敛
        acceptable_tol=1e-16,               # 设置比主容差更严格
        acceptable_iter=0,                  # 禁用可接受解
        ))
    

    grid = buffer.grid
    num_of_element = length(grid.VOL) # 因为可能要用自己写的 SimpleGrid， 这里先用 VOL 来算 element/node 数量
    num_of_nodes = size(grid.VOL[1][1])[1]

    num_of_xyz_gmsh = length(grid.xyz_gmsh)

    # 计算 参考单元的 失真度
    ref = buffer.ref
    area_of_ref_elem = sum(ref.H)
    G_ref = I(2) # ∂ ξ / ∂ ξ = I_2
    δ_ref = norm(G_ref) / det(G_ref)^0.5
    δref2 = δ_ref^2
    r_msh_ref = area_of_ref_elem * δ_ref^2
    
    n_total = length(buffer.indices_moving_coords) + length(buffer.initial_guess) - buffer.index_mesh

    # 储存最开始的 xyz_gmsh 坐标, 在 solver 迭代更新的过程中，可以保证只有 能动的在动，不能动的一直没变
    initial_xyz_gmsh = buffer.initial_guess[1:buffer.index_mesh]
    new_xyz_gmsh = deepcopy(initial_xyz_gmsh) # 初始化 new_xyz_gmsh 为初始值
    new_xyz_gmsh_coords = [Coord{2, Float64}((new_xyz_gmsh[i], new_xyz_gmsh[num_of_xyz_gmsh + i])) for i in 1:num_of_xyz_gmsh]
    # current_grid[] = construct_simpleGrid(
    #     grid.cells, buffer.ref, new_xyz_gmsh_coords, 
    #     grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
    # )
            

    # 不是所有的坐标都能动，边界上只有一个坐标，非边界上两个都能动，但corner上两个都不能动
    u0_vec = vcat(buffer.initial_guess[buffer.indices_moving_coords], buffer.initial_guess[buffer.index_mesh+1:end])
    if length(u0_vec) != n_total
        error("The length of u0_vec must be equal to n_total")
    end
    @variable(model, u[i=1:n_total], start = u0_vec[i])


    # PDE 残差作为等式约束
    function pde_residual(x::T...) where {T<:Real}
        mesh_u_vec = collect(x)  
        mesh_vec = mesh_u_vec[1: length(buffer.indices_moving_coords)] # 前几个放的是可以动的坐标
        u_vec = mesh_u_vec[(length(buffer.indices_moving_coords) + 1):end]  # 后面放的是解变量

        new_xyz_gmsh = Vector{T}(initial_xyz_gmsh)  # ✅ 类型匹配
        new_xyz_gmsh[buffer.indices_moving_coords] = mesh_vec
        new_xyz_gmsh_coords = [Coord{2, T}((new_xyz_gmsh[i], new_xyz_gmsh[num_of_xyz_gmsh + i])) for i in 1:num_of_xyz_gmsh]

        current_grid = construct_simpleGrid(
            grid.cells, buffer.ref, 
            new_xyz_gmsh_coords,
            grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
        )


        Umatrix = reshape(u_vec, (num_of_element, num_of_nodes))
        dU = similar(Umatrix, T) 
        fill!(dU, zero(T))  # 初始化 dU
        new_para = (current_grid, buffer.a)
        RHS_for_solution(dU, Umatrix, new_para)  # 计算残差
        return vec(dU)

    end

    function mesh_shock_objective(x::T...) where {T<:Real}

        # 🎯 目标面积：总面积/单元数
        target_area = T(0.4 / 13)  # 或者动态计算：total_domain_area / num_of_element
    

        mesh_u_vec = collect(x)  
        mesh_vec = mesh_u_vec[1: length(buffer.indices_moving_coords)] # 前几个放的是可以动的坐标
        u_vec = mesh_u_vec[(length(buffer.indices_moving_coords) + 1):end]  # 后面放的是解变量

        new_xyz_gmsh = Vector{T}(initial_xyz_gmsh)  # ✅ 类型匹配
        new_xyz_gmsh[buffer.indices_moving_coords] = mesh_vec
        new_xyz_gmsh_coords = [Coord{2, T}((new_xyz_gmsh[i], new_xyz_gmsh[num_of_xyz_gmsh + i])) for i in 1:num_of_xyz_gmsh]

        current_grid = construct_simpleGrid(
            grid.cells, buffer.ref, 
            new_xyz_gmsh_coords,
            grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
        )
        
            # 计算每个单元的 mesh distortion
        r_mesh = zeros(T, num_of_element)
        for cell_id in 1:num_of_element
            xK = [coord[1] for coord in current_grid.xyz_SBP[cell_id]]
            tK = [coord[2] for coord in current_grid.xyz_SBP[cell_id]]
            xt = [xK, tK]
            GK = zeros(T, 2,2)
            for a in 1:2, b in 1:2
                GK[a,b] = mean(ref.D[a] * xt[b])  # 累加第 b 方向导数 * 物理坐标分量 a
            end
            # 修复：使用绝对值避免负数开方
            # 更robust的处理方式
            det_GK = abs(det(GK))
            norm_GK = norm(GK)
            
            if abs(det_GK) < 1e-12
                # 接近奇异的情况，给一个很大的惩罚
                δK = T(1e6)
            else
                # 正常情况
                δK = norm_GK / sqrt(det_GK)
            end
            r_msh_K = area_of_ref_elem * δK^2
            # r_mesh[cell_id] = r_msh_K - r_msh_ref  # 减去参考单元的失真度
            r_mesh[cell_id] = max(r_msh_K - r_msh_ref, 0)
        end
        
        # 计算每个单元的 shock tracking residual
        r_shk = zeros(T, num_of_element)
        Umatrix = reshape(u_vec, (num_of_element, num_of_nodes))
        for cell_id in 1:num_of_element
            num_uk = Umatrix[cell_id, :]
            quadrature_k = current_grid.geometric_terms.J_q[cell_id] .* diag(ref.H)
            area_K = area_of_ref_elem * current_grid.geometric_terms.J_q[cell_id][1]
            num_̄uk = sum(quadrature_k .* num_uk) / area_K
            r_shk[cell_id] = sum((num_uk .- num_̄uk).^2 .* quadrature_k) / area_K  # 计算 shock tracking residual
            # r_shk[cell_id] = sum((num_uk .- num_̄uk).^2 .* quadrature_k)
        end

        return αmesh * sum(r_mesh.^2) + αshk * sum(r_shk.^2)
        
    end

    # 注册函数
    # register(model, :pde_residual, n_total, pde_residual; autodiff = true)
    register(model, :mesh_shock_objective, n_total, mesh_shock_objective; autodiff = true)

    # 解决方案：为每个残差分量创建单独的函数
    residual_size = length(buffer.initial_guess) - buffer.index_mesh
    
    # 先注册所有分量函数
    component_functions = []
    for i in 1:residual_size
        component_func = function(x::T...) where {T<:Real}
            full_residual = pde_residual(x...)
            return full_residual[i]
        end
        push!(component_functions, component_func)
        register(model, Symbol("pde_residual_comp_$i"), n_total, component_func; autodiff = true)
    end
    
    # 使用 @eval 在全局作用域创建约束
    for i in 1:residual_size
        # 在全局作用域中动态创建约束
        eval(quote
            @NLconstraint($model, $(Symbol("pde_residual_comp_$i"))($(u[:])...) == 0.0)
        end)
    end

    # @NLconstraint(model, con[i=1:residual_size], 
    #             pde_residual_vector(u...)[i] == 0.0)
    # 设置网格质量和shock指标作为目标函数
    @NLobjective(model, Min, mesh_shock_objective(u...))

    # ---------------------------------------------------
    # 记录初始状态
    initial_obj = mesh_shock_objective(u0_vec...)
    println("🚀 开始优化，初始目标函数值: $initial_obj")
    
    # 计时求解
    solve_time = @elapsed begin
        optimize!(model)
    end
    
    # 获取最终解
    final_u = value.(u)
    final_obj = objective_value(model)
    final_residual = norm(pde_residual(final_u...))  # 计算PDE残差的范数
    converged = termination_status(model) == MOI.LOCALLY_SOLVED
    
    # 更新 buffer 状态
    buffer.final_solution = final_u
    buffer.initial_objective = initial_obj
    buffer.final_objective = final_obj
    buffer.final_residual = final_residual
    buffer.converged = converged
    buffer.solve_time = solve_time

    # 输出总结
    println("✅ 优化完成!")
    println("  求解时间: $(solve_time) 秒")
    println("  初始目标函数值: $initial_obj")
    println("  最终目标函数值: $final_obj")
    println("  目标函数减少: $(initial_obj - final_obj)")
    println("  减少比例: $(100 * (initial_obj - final_obj) / initial_obj)%")
    println("  最终PDE残差范数: $final_residual")
    println("  是否收敛: $converged")
    println("  终止状态: $(termination_status(model))")

    return model  # 返回更新后的 buffer
end

# oscillating_u[1, :] .= 2.0
# oscillating_u[2, :] .= 2.0
# oscillating_u[3, :] .= 1.0
# oscillating_u[4, :] .= 1.0

buffer = SpaceTimeBuffer(grid, [0.0, 1.0], oscillating_u)
buffer.indices_moving_coords = [3]
buffer.indices_moving_coords
model1 = pde_model_with_solver_struct(buffer, αmesh = 0, αshk = 2000, max_iter = 50, lin_solver = MadNLPHSL.Ma57Solver)

using Plots
include(joinpath(@__DIR__, "../plot_helper_simple_grid.jl"))
final_simple_grid, final_u = wrap_up_results(buffer)
plot_u_interactive(final_simple_grid, final_u)
plot_mesh(final_simple_grid)

@save joinpath(@__DIR__, "../optimization_results//buffer_one_coordinates_const_problem.jld2") buffer final_simple_grid final_u

@load joinpath(@__DIR__, "../optimization_results//buffer_one_coordinates_const_problem.jld2") buffer final_simple_grid final_u




plot_u_interactive(grid, oscillating_u)

plot_mesh(grid)


# -------------------------------------------------------------------
# plot for presentation here
xlimit = (-0.01, 1.01)
ylimit = (-0.01, 0.11)

BOTTOM_INFLOW = collect(union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2")))
x_coords_initial_before_opt = Dict()
for (cell_id, face_id) in BOTTOM_INFLOW
    x_coords_initial_before_opt[cell_id] = map(x -> x[1], ref.R[face_id] * grid.xyz_q[cell_id])
end
bottom_bc_before_opt = Dict()
bottom_bc_before_opt[2] = 2 * ones(Float64, length(x_coords_initial_before_opt[2]))
bottom_bc_before_opt[3] = ones(Float64, length(x_coords_initial_before_opt[3]))


# elements having final time are 9, 13, 10, 8
# cooresponding face indices are 1, 1, 1, 1
TOP_INFLOW = collect(union(get_face_set(grid, "TOP_INFLOW_1"), get_face_set(grid, "TOP_INFLOW_2")))
x_coords_final_before_opt = Dict()
for (cell_id, face_id) in TOP_INFLOW
    x_coords_final_before_opt[cell_id] = map(x -> x[1], ref.R[face_id] * grid.xyz_q[cell_id])
end
top_bc_before_opt = Dict()
top_bc_before_opt[1] = 2 * ones(Float64, length(x_coords_final_before_opt[1]))
top_bc_before_opt[4] = ones(Float64, length(x_coords_final_before_opt[4]))

@load joinpath(@__DIR__, "lin_adv_oscillating_solution.jld2") oscillating_u grid

gr()

final_solution_comparision_before_opt = plot(xlims = xlimit, grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    plot!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], ref.R[face_id] * oscillating_u[cell_id, :], 
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], ref.R[face_id] * oscillating_u[cell_id, :], 
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], top_bc_before_opt[cell_id],
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], top_bc_before_opt[cell_id],
           label = "", color = :red, markersize = 4)
    first_red = false
end
title!(final_solution_comparision_before_opt, "Numerical vs Analytic at t = 0.1 \n before optimization")
xlabel!(final_solution_comparision_before_opt, "x")
ylabel!(final_solution_comparision_before_opt, "u")
savefig(final_solution_comparision_before_opt, joinpath(@__DIR__, "../plots_const_burgers/burgers_final_sol_compare_before_opt_const_problem.png"))

initial_solution_comparision_before_opt = plot(xlims = xlimit, grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in BOTTOM_INFLOW
    plot!(initial_solution_comparision_before_opt, x_coords_initial_before_opt[cell_id], ref.R[face_id] * oscillating_u[cell_id, :], 
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(initial_solution_comparision_before_opt, x_coords_initial_before_opt[cell_id], ref.R[face_id] * oscillating_u[cell_id, :], 
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(initial_solution_comparision_before_opt, x_coords_initial_before_opt[cell_id], bottom_bc_before_opt[cell_id],
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(initial_solution_comparision_before_opt, x_coords_initial_before_opt[cell_id], bottom_bc_before_opt[cell_id],
           label = "", color = :red, markersize = 4)
    first_red = false
end 
title!(initial_solution_comparision_before_opt, "Numerical vs Analytic at t = 0 \n before optimization")
xlabel!(initial_solution_comparision_before_opt, "x")
ylabel!(initial_solution_comparision_before_opt, "u")
savefig(initial_solution_comparision_before_opt, joinpath(@__DIR__, "../plots_const_burgers/burgers_initial_sol_compare_before_opt_const_problem.png"))


# @load joinpath(@__DIR__, "..//optimization_results//buffer_two_coordinates.jld2") buffer final_simple_grid final_u
# include(joinpath(@__DIR__, "../../src_simpleGrid/SimpleGrid.jl"))
# include(joinpath(@__DIR__, "../plot_helper_simple_grid.jl"))
# final_simple_grid, final_u = wrap_up_results(buffer)
# plot_u_interactive(final_simple_grid, final_u)
# plot_mesh(final_simple_grid)

x_coords_final_after_opt = Dict()
for (cell_id, face_id) in TOP_INFLOW
    x_coords_final_after_opt[cell_id] = map(x -> x[1], ref.R[face_id] * final_simple_grid.xyz_SBP[cell_id])
end
top_bc_after_opt = Dict()
top_bc_after_opt[1] = 2 * ones(Float64, length(x_coords_final_after_opt[1]))
top_bc_after_opt[4] = ones(Float64, length(x_coords_final_after_opt[4]))




x_coords_initial_after_opt = Dict()
for (cell_id, face_id) in BOTTOM_INFLOW
    x_coords_initial_after_opt[cell_id] = map(x -> x[1], ref.R[face_id] * final_simple_grid.xyz_SBP[cell_id])
end
bottom_bc_after_opt = Dict()
bottom_bc_after_opt[2] = 2 * ones(Float64, length(x_coords_initial_after_opt[2]))
bottom_bc_after_opt[3] = ones(Float64, length(x_coords_initial_after_opt[3]))

gr()
initial_solution_comparision_after_opt = plot(xlims = xlimit, grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in BOTTOM_INFLOW
    plot!(initial_solution_comparision_after_opt, x_coords_initial_after_opt[cell_id], ref.R[face_id] * final_u[cell_id, :],
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(initial_solution_comparision_after_opt, x_coords_initial_after_opt[cell_id], ref.R[face_id] * final_u[cell_id, :],
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(initial_solution_comparision_after_opt, x_coords_initial_after_opt[cell_id], bottom_bc_after_opt[cell_id],
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(initial_solution_comparision_after_opt, x_coords_initial_after_opt[cell_id], bottom_bc_after_opt[cell_id],
           label = "", color = :red, markersize = 4)
    first_red = false
end
title!(initial_solution_comparision_after_opt, "Numerical vs Analytic at t = 0 \n after optimization")
xlabel!(initial_solution_comparision_after_opt, "x")
ylabel!(initial_solution_comparision_after_opt, "u")
savefig(initial_solution_comparision_after_opt, joinpath(@__DIR__, "../plots_const_burgers/burgers_initial_sol_compare_after_opt_const_problem.png"))

final_solution_comparision_after_opt = plot(xlims = xlimit, grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    plot!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], ref.R[face_id] * final_u[cell_id, :],
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], ref.R[face_id] * final_u[cell_id, :],
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], top_bc_after_opt[cell_id],
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], top_bc_after_opt[cell_id],
           label = "", color = :red, markersize = 4)
    first_red = false
end
title!(final_solution_comparision_after_opt, "Numerical vs Analytic at t = 0.1 \n after optimization")
xlabel!(final_solution_comparision_after_opt, "x")
ylabel!(final_solution_comparision_after_opt, "u")
savefig(final_solution_comparision_after_opt, joinpath(@__DIR__, "../plots_const_burgers/burgers_final_sol_compare_after_opt_const_problem.png"))

using Plots.Measures
gr()
num_of_cells = length(grid.VOL)
num_of_nodes = size(grid.VOL[1][1], 1)
plot_size = (1200, 400)
mesh_comparision = scatter(xlabel = "x", ylabel = "t", title = "", 
                                    xlims = xlimit, ylims = ylimit, size = plot_size, 
                                    margin = 5mm, legend = (0.5, -0.25), grid = false, bottom_margin = 20mm)
x_min, x_max = xlimit
y_min, y_max = ylimit
x_range = x_max - x_min
y_range = y_max - y_min
first_blue = true
first_red = true
for cell_id in 1:num_of_cells
    # 🔥 优化前的网格（蓝色）
    vertices_IDS = vertices(grid.cells[cell_id])
    vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
    vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
    vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

    v1x, v1y = vertice1[1], vertice1[2]
    v2x, v2y = vertice2[1], vertice2[2]
    v3x, v3y = vertice3[1], vertice3[2]

    plot!(mesh_comparision, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
          # 🔥 只在第一次添加标签
          label = first_blue ? "before" : "", 
          color = :blue, 
          linewidth = 1.5)
    
    first_blue = false  # 后续不再添加蓝色标签

    # 🔥 优化后的网格（红色）
    vertices_IDS_2 = vertices(final_simple_grid.cells[cell_id])
    vertice1_2 = final_simple_grid.xyz_gmsh[vertices_IDS_2[1]]
    vertice2_2 = final_simple_grid.xyz_gmsh[vertices_IDS_2[2]]
    vertice3_2 = final_simple_grid.xyz_gmsh[vertices_IDS_2[3]]

    v1x_2, v1y_2 = vertice1_2[1], vertice1_2[2]
    v2x_2, v2y_2 = vertice2_2[1], vertice2_2[2]
    v3x_2, v3y_2 = vertice3_2[1], vertice3_2[2]

    plot!(mesh_comparision, [v1x_2, v2x_2, v3x_2, v1x_2], [v1y_2, v2y_2, v3y_2, v1y_2], 
          # 🔥 只在第一次添加标签
          label = first_red ? "after" : "", 
          color = :purple, 
          linewidth = 1.5)
    
    first_red = false  # 后续不再添加红色标签
end
scatter!(mesh_comparision, 
          [-0.1, 0.1], [0.0, 0.2], 
          label = "", color = :red, markersize = 5)  # 添加蓝色标签的占位符
plot!(mesh_comparision, [0.4, 0.55], [0.0, 0.1], label = "True shock", color = :red, markersize = 5)
mesh_comparision
savefig(mesh_comparision, joinpath(@__DIR__, "../plots_const_burgers/burgers_mesh_comparision_const_problem.png"))

mesh_nonaligned = scatter(xlabel = "x", ylabel = "t", title = "", 
                                    xlims = xlimit, ylims = ylimit, size = plot_size, 
                                    margin = 5mm, legend = (0.5, -0.25), grid = false, bottom_margin = 20mm)
first_blue = true
# 画网格
for cell_id in 1:num_of_cells
    # 🔥 优化前的网格（蓝色）
    vertices_IDS = vertices(grid.cells[cell_id])
    vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
    vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
    vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

    v1x, v1y = vertice1[1], vertice1[2]
    v2x, v2y = vertice2[1], vertice2[2]
    v3x, v3y = vertice3[1], vertice3[2]

    plot!(mesh_nonaligned, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
          # 🔥 只在第一次添加标签
          label = first_blue ? "mesh (before)" : "", 
          color = :blue, 
          linewidth = 1.5)

    first_blue = false

    
    x = [grid.xyz_q[cell_id][node_id][1] for node_id in 1:num_of_nodes]
    y = [grid.xyz_q[cell_id][node_id][2] for node_id in 1:num_of_nodes]


    scatter!(mesh_nonaligned, x, y, markersize=3, label="", markercolor=:blue)

end
scatter!(mesh_nonaligned, 
          [-0.1, 0.1], [0.0, 0.2], 
          label = "", color = :red, markersize = 5)  # 添加蓝色标签的占位符
plot!(mesh_nonaligned, [0.4, 0.55], [0.0, 0.1], label = "True shock", color = :red)
savefig(mesh_nonaligned, joinpath(@__DIR__, "../plots_const_burgers/burgers_mesh_nonaligned_const_problem.png"))

mesh_after_opt = scatter(xlabel = "x", ylabel = "t", title = "", 
                                    xlims = xlimit, ylims = ylimit, size = plot_size, 
                                    margin = 5mm, legend = (0.5, -0.25), grid = false, bottom_margin = 20mm)
first_blue = true
for cell_id in 1:num_of_cells
    # 🔥 优化前的网格（蓝色）
    vertices_IDS = vertices(final_simple_grid.cells[cell_id])
    vertice1 = final_simple_grid.xyz_gmsh[vertices_IDS[1]]
    vertice2 = final_simple_grid.xyz_gmsh[vertices_IDS[2]]
    vertice3 = final_simple_grid.xyz_gmsh[vertices_IDS[3]]

    v1x, v1y = vertice1[1], vertice1[2]
    v2x, v2y = vertice2[1], vertice2[2]
    v3x, v3y = vertice3[1], vertice3[2]

    plot!(mesh_after_opt, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
        # 🔥 只在第一次添加标签
        label = first_blue ? "mesh (after)" : "", 
        color = :blue, 
        linewidth = 1.5)

    first_blue = false

    x = [final_simple_grid.xyz_SBP[cell_id][node_id][1] for node_id in 1:num_of_nodes]
    y = [final_simple_grid.xyz_SBP[cell_id][node_id][2] for node_id in 1:num_of_nodes]


    scatter!(mesh_after_opt, x, y, markersize=3, label="", markercolor=:blue)


end
scatter!(mesh_after_opt, 
        [-0.1, 0.1], [0.0, 0.2], 
        label = "", color = :red, markersize = 5)  # 添加蓝色标签的占位符
plot!(mesh_after_opt, [0.4, 0.55], [0.0, 0.1], label = "True shock", color = :red, markersize = 5)
mesh_after_opt
savefig(mesh_after_opt, joinpath(@__DIR__, "../plots_const_burgers/burgers_mesh_after_opt_const_problem.png"))

gr()
error_comparison_final = scatter(grid = false, xlabel = "x", ylabel = "Error", title = "Error Comparison at t = 0.1")
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    scatter!(error_comparison_final, x_coords_final_before_opt[cell_id], abs.(top_bc_before_opt[cell_id] - ref.R[face_id] * oscillating_u[cell_id, :]), color = :blue, label = first_blue ? "before" : "")
    scatter!(error_comparison_final, x_coords_final_after_opt[cell_id], abs.(top_bc_after_opt[cell_id] - ref.R[face_id] * final_u[cell_id, :]), color = :red, label = first_red ? "after" : "")
    first_blue = false
    first_red = false
end
error_comparison_final
savefig(error_comparison_final, joinpath(@__DIR__, "../plots_const_burgers/burgers_error_comparison_final_const_problem.png"))

error_comparison_initial = scatter(grid = false, xlabel = "x", ylabel = "Error", title = "Error Comparison at t = 0.0")
first_blue = true
first_red = true
for (cell_id, face_id) in BOTTOM_INFLOW
    scatter!(error_comparison_initial, x_coords_initial_before_opt[cell_id], abs.(bottom_bc_before_opt[cell_id] - ref.R[face_id] * oscillating_u[cell_id, :]), color = :blue, label = first_blue ? "before" : "")
    scatter!(error_comparison_initial, x_coords_initial_after_opt[cell_id], abs.(bottom_bc_after_opt[cell_id] - ref.R[face_id] * final_u[cell_id, :]), color = :red, label = first_red ? "after" : "")
    first_blue = false
    first_red = false
end
error_comparison_initial
savefig(error_comparison_initial, joinpath(@__DIR__, "../plots_const_burgers/burgers_error_comparison_initial_const_problem.png"))

