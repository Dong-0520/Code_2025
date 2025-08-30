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

order = 3
ref = TriangleDiagELG(order, 2 * order)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)

grid = read_mesh(joinpath(@__DIR__, "../const_problem/lin_adv_grid_nonalign_75_half.msh"), ref_elems_data, identity)
@save joinpath(@__DIR__, "lin_adv_grid_nonalign_75_half.jld2") grid
@load joinpath(@__DIR__, "lin_adv_grid_nonalign_75_half.jld2") grid
ax, at = 1.0, 1.0
# bottomBC1(x::Coord) = -1
# bottomBC2(x::Coord) = 1
function bottomBC(x::Coord)
    if x[1] <= -0.1
        return -cospi(x[1]) - 0.5
    else
        return cospi(x[1]) + 0.5
    end
end
leftBC(x::Coord) = -cospi(x[1] - ax * x[2]) - 0.5
function analytic_sol(x1::Float64; x2 = 0.2, ax = ax)
    if x1 - ax * x2 <= -0.1
        return -cospi(x1 - ax * x2) - 0.5
    else
        return cospi(x1 - ax * x2) + 0.5
    end
end

# function bottomBC(x::Coord)
#     if x[1] <= -0.1
#         return -cospi(x[1]) - 1
#     else
#         return cospi(x[1]) + 1
#     end
# end
# leftBC(x::Coord) = -cospi(-1 - ax * x[2]) - 1


Pkg.activate("MadNLP_env")


# function RHS(du, u, p)

#     grid, a = p

#     for cell_id in 1:n_cells(grid)
#         J_k = Diagonal(grid.geometric_terms.J_q[cell_id])
#         Dx = grid.VOL[cell_id][1]
#         Dt = grid.VOL[cell_id][2]
#         local_sol = @view u[cell_id, :]
#         ux = ax * J_k * Dx * local_sol
#         ut = at * J_k * Dt * local_sol

#         du[cell_id, :] = -ux - ut
#     end

#     for interface in grid.face_interfaces
#         c1, lf1 = interface.face_1
#         c2, lf2 = interface.face_2
#         P1, P2 = interface.P1, interface.P2
#         elem1 = grid.cells[c1].ref_data[]
#         elem2 = grid.cells[c2].ref_data[]
#         R_1, R_2 = elem1.R[lf1], elem2.R[lf2]
#         mask1, mask2 = @views elem1.f_mask[lf1], elem2.f_mask[lf2]


#         normal_1, normal_2 = @views grid.geometric_terms.N_f[c1][:, mask1], grid.geometric_terms.N_f[c2][:, mask2]

#         an1, an2 = normal_1 .* a, normal_2 .* a
#         lambda_1, lambda_2 = an1[1, :] + an1[2, :], an2[1, :] + an2[2, :]

#         u1_face, u2_face = R_1 * u[c1, :], R_2 * u[c2, :]
#         u1_adj, u2_adj = u2_face[P2], u1_face[P1]
#         flux_1 = 0.5 * ((lambda_1 .+ abs.(lambda_1)) .* u1_face .+ (lambda_1 .- abs.(lambda_1)) .* u1_adj)
#         flux_2 = 0.5 * ((lambda_2 .+ abs.(lambda_2)) .* u2_face .+ (lambda_2 .- abs.(lambda_2)) .* u2_adj)
#         du[c1, :] += elem1.H_inv * Matrix(R_1)' * elem1.H_face * (lambda_1 .* u1_face .- flux_1)
#         du[c2, :] += elem2.H_inv * Matrix(R_2)' * elem2.H_face * (lambda_2 .* u2_face .- flux_2)

#     end


#     for (cell_id, face_id) in union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2"))
#         cell = get_cells(grid, cell_id)
#         elem = grid.cells[cell_id].ref_data[]
#         R = elem.R[face_id]
#         mask = @views elem.f_mask[face_id]
#         normal = grid.geometric_terms.N_f[cell_id][:, mask]
#         an = normal .* a
#         gamma = an[1, :] + an[2, :]

#         u_face = R * u[cell_id, :]
#         u_adj = bottomBC.(R * grid.xyz_q[cell_id])
#         flux = gamma .* u_adj
#         du[cell_id, :] += elem.H_inv * Matrix(R)' * elem.H_face * (gamma .* u_face .- flux)
#     end


#     for (cell_id, face_id) in get_face_set(grid, "LEFT_INFLOW")
#         cell = get_cells(grid, cell_id)
#         elem = grid.cells[cell_id].ref_data[]
#         R = elem.R[face_id]
#         mask = @views elem.f_mask[face_id]
#         normal = grid.geometric_terms.N_f[cell_id][:, mask]
#         an = normal .* a
#         gamma = an[1, :] + an[2, :]

#         u_face = R * u[cell_id, :]
#         u_adj = leftBC.(R * grid.xyz_q[cell_id])
#         flux = gamma .* u_adj
#         du[cell_id, :] += elem.H_inv * Matrix(R)' * elem.H_face * (gamma .* u_face .- flux)
#     end

#     return nothing
# end

# function set_up_U0(grid; perturbation = 1)
#     U0 = zeros(Float64, (n_cells(grid), length(grid.xyz_q[1])))
#     for cell_id in 1:n_cells(grid)
#         x = [x[1] for x in grid.xyz_q[cell_id]]
#         t = [x[2] for x in grid.xyz_q[cell_id]]
#         if all(t .> x .+ 0.1 .- 1e-5)
#             U0[cell_id, :] = -cospi.(x.-t) .- 1
#         else
#             U0[cell_id, :] = cospi.(x.-t) .+ 1
#         end
#     end
#     return U0 .+ rand(size(U0)) .* perturbation
    
# end

# using NonlinearSolve, LineSearches, ADTypes, SparseConnectivityTracer, SparseDiffTools, NLsolve
# U0_for_solver = set_up_U0(grid; perturbation = 0.1)

# u0_vec = vec(U0_for_solver)
# du = similar(U0_for_solver); fill!(du, 0.0)


# # --- 构造 in-place 残差 ---
# # 用 NLsolve 来解

# function f_nl!(F, u)
#     U = reshape(u, size(U0_for_solver))
#     du = similar(U); fill!(du, 0.0)
#     f_forSolver!(du, U, para)      # 你的核心残差
#     F[:] = vec(du)
# end

# function f_forSolver!(du, u, p)
#     # du must be pre-allocated to the same shape as u
#     # p is your packed parameters tuple (grid, newGrid, IC, FBCL, FBCR)
#     RHS(du, u, p)
#     return nothing
# end

# para = (grid, [ax, at])

# res = nlsolve(
#   f_nl!, u0_vec;
#   method    = :trust_region,
#   linsolve  = :dogleg,      # :cg
#   autodiff  = :central,   
#   xtol      = 1e-6,
#   ftol      = 1e-6,
#   autoscale = true,
#   show_trace= true,
#   iterations = 150,
# )

# num_u = reshape(res.zero, size(U0_for_solver))
# using Plots
# # plot_u_2D(grid, num_u)
# include(joinpath(@__DIR__, "../plot_helper_grid.jl"))
# plot_u_interactive(grid, reshape(u0_vec, size(U0_for_solver)))
# plot_u_interactive(grid, num_u)
# plot_mesh(grid)
# oscillating_u = num_u
# @save joinpath(@__DIR__, "lin_adv_oscillating_solution_sin.jld2") oscillating_u grid


using JuMP, MadNLP, Random, MadNLPHSL
@load joinpath(@__DIR__, "lin_adv_oscillating_solution_sin.jld2") oscillating_u grid

include(joinpath(@__DIR__, "../../src_simpleGrid/SimpleGrid.jl"))

function RHS_for_solution(du, u, para)

    # 这里的 grid 是 SimpleGrid 类型
    grid, a = para
    ref = grid.ref

    for cell_id in 1:length(grid.cells)
        J_k = Diagonal(grid.geometric_terms.J_q[cell_id])
        Dx = grid.VOL[cell_id][1]
        Dt = grid.VOL[cell_id][2]
        local_sol = @view u[cell_id, :]
        ux = ax * J_k * Dx * local_sol
        ut = at * J_k * Dt * local_sol

        du[cell_id, :] = -ux - ut
    end

    for interface in grid.face_interfaces
        c1, lf1 = interface.face_1
        c2, lf2 = interface.face_2
        P1, P2 = interface.P1, interface.P2
        R_1, R_2 = ref.R[lf1], ref.R[lf2]
        mask1, mask2 = @views ref.f_mask[lf1], ref.f_mask[lf2]


        normal_1, normal_2 = @views grid.geometric_terms.N_f[c1][:, mask1], grid.geometric_terms.N_f[c2][:, mask2]

        an1, an2 = normal_1 .* a, normal_2 .* a
        lambda_1, lambda_2 = an1[1, :] + an1[2, :], an2[1, :] + an2[2, :]

        u1_face, u2_face = R_1 * u[c1, :], R_2 * u[c2, :]
        u1_adj, u2_adj = u2_face[P2], u1_face[P1]
        flux_1 = 0.5 * ((lambda_1 .+ abs.(lambda_1)) .* u1_face .+ (lambda_1 .- abs.(lambda_1)) .* u1_adj)
        flux_2 = 0.5 * ((lambda_2 .+ abs.(lambda_2)) .* u2_face .+ (lambda_2 .- abs.(lambda_2)) .* u2_adj)
        du[c1, :] += ref.H_inv * Matrix(R_1)' * ref.H_face * (lambda_1 .* u1_face .- flux_1)
        du[c2, :] += ref.H_inv * Matrix(R_2)' * ref.H_face * (lambda_2 .* u2_face .- flux_2)

    end

    for (cell_id, face_id) in union(grid.face_sets["BOTTOM_INFLOW_1"], grid.face_sets["BOTTOM_INFLOW_2"])
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        an = normal .* a
        gamma = an[1, :] + an[2, :]

        u_face = R * u[cell_id, :]
        u_adj = bottomBC.(R * grid.xyz_SBP[cell_id])
        flux = gamma .* u_adj
        du[cell_id, :] += ref.H_inv * Matrix(R)' * ref.H_face * (gamma .* u_face .- flux)
    end

    for (cell_id, face_id) in grid.face_sets["LEFT_INFLOW"]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        an = normal .* a
        gamma = an[1, :] + an[2, :]

        u_face = R * u[cell_id, :]
        u_adj = leftBC.(R * grid.xyz_SBP[cell_id])
        flux = gamma .* u_adj
        du[cell_id, :] += ref.H_inv * Matrix(R)' * ref.H_face * (gamma .* u_face .- flux)
    end

    return nothing
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
        
        # 更宽松的收敛条件
        # tol=1e-8,                          # 放松主要容差
        # acceptable_tol=1e-2,               # 更宽松的可接受容差
        # acceptable_iter=5,                 # 更早接受解
        
        # 步长控制
        # alpha_min=1e-8,                    # 更大的最小步长
        
        # 发散控制
        # diverging_iterates_tol=1e8,        # 更严格的发散检测
        
        # 正则化
        # jacobian_regularization_value=1e-6, # 更强的正则化
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

buffer = SpaceTimeBuffer(grid, [ax, at], oscillating_u)
buffer.indices_moving_coords = [10]
buffer.indices_moving_coords
model1 = pde_model_with_solver_struct(buffer, αmesh = 0.1, αshk = 100, max_iter = 100, lin_solver = MadNLPHSL.Ma57Solver)

# using Plots
include(joinpath(@__DIR__, "../plot_helper_simple_grid.jl"))
final_simple_grid, final_u = wrap_up_results(buffer)
plot_u_interactive(final_simple_grid, final_u)
plot_mesh(final_simple_grid)

@save joinpath(@__DIR__, "../optimization_results//buffer_two_coordinates_sin_problem.jld2") buffer final_simple_grid final_u




@load joinpath(@__DIR__, "../optimization_results//buffer_two_coordinates_sin_problem.jld2") buffer final_simple_grid final_u




plot_u_interactive(grid, oscillating_u)

plot_mesh(grid)
# -------------------------------------------------------------------
# debug here
# 计算 参考单元的 失真度

area_of_ref_elem = sum(ref.H)
G_ref = I(2) # ∂ ξ / ∂ ξ = I_2
δ_ref = norm(G_ref) / det(G_ref)^0.5
δref2 = δ_ref^2
r_msh_ref = area_of_ref_elem * δ_ref^2

for cell_id in 1:4
    xK = [coord[1] for coord in grid.xyz[cell_id]]
    tK = [coord[2] for coord in grid.xyz[cell_id]]
    xt = [xK, tK]
    GK = zeros(Float64, 2,2)
    for a in 1:2, b in 1:2
        GK[a,b] = mean(ref.D[a] * xt[b])  # 累加第 b 方向导数 * 物理坐标分量 a
    end
    # 修复：使用绝对值避免负数开方
    # 更robust的处理方式
    det_GK = abs(det(GK))
    norm_GK = norm(GK)
    
    if abs(det_GK) < 1e-12
        # 接近奇异的情况，给一个很大的惩罚
        δK = Float64(1e6)
    else
        # 正常情况
        δK = norm_GK / sqrt(det_GK)
    end
    r_msh_K = area_of_ref_elem * δK^2
    # r_mesh[cell_id] = r_msh_K - r_msh_ref  # 减去参考单元的失真度
    print("cell_id: ", cell_id, " r_msh_K: ", r_msh_K, " r_msh_ref: ", r_msh_ref, " δK: ", δK, "\n")
    r_mesh[cell_id] = max(r_msh_K - r_msh_ref, 0)
end

cell_id = 3
xK = [coord[1] for coord in grid.xyz[cell_id]]
tK = [coord[2] for coord in grid.xyz[cell_id]]
xt = [xK, tK]
GK = zeros(Float64, 2,2)
for a in 1:2, b in 1:2
    GK[a,b] = mean(ref.D[a] * xt[b])  # 累加第 b 方向导数 * 物理坐标分量 a
end
# 修复：使用绝对值避免负数开方
# 更robust的处理方式
det_GK = det(GK)
norm_GK = norm(GK)

if abs(det_GK) < 1e-12 || det_GK < 0
    # 接近奇异的情况，给一个很大的惩罚
    δK = Float64(1e6)
else
    # 正常情况
    δK = norm_GK / sqrt(det_GK)
end
r_msh_K = area_of_ref_elem * δK^2
# r_mesh[cell_id] = r_msh_K - r_msh_ref  # 减去参考单元的失真度
print("cell_id: ", cell_id, " r_msh_K: ", r_msh_K, " r_msh_ref: ", r_msh_ref, " δK: ", δK, "\n")
r_mesh[cell_id] = max(r_msh_K - r_msh_ref, 0)



# PDE 残差作为等式约束
function pde_residual_debug(buffer)
    u_vec = buffer.initial_guess[buffer.index_mesh+1:end]  # 使用最终解作为输入
    num_of_element = length(grid.VOL) # 因为可能要用自己写的 SimpleGrid， 这里先用 VOL 来算 element/node 数量
    num_of_nodes = size(grid.VOL[1][1])[1]

    xyz_gmsh = buffer.initial_guess[1:buffer.index_mesh]
    xyz_gmsh_coords = [Coord{2, Float64}((xyz_gmsh[i], xyz_gmsh[buffer.index_mesh ÷ 2 + i])) for i in 1:buffer.index_mesh÷2]
    current_grid = construct_simpleGrid(
        grid.cells, buffer.ref, xyz_gmsh_coords, 
        grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
    )


    Umatrix = reshape(u_vec, (num_of_element, num_of_nodes))
    dU = similar(Umatrix) 
    fill!(dU, 0.0)  # 初始化 dU
    new_para = (current_grid, buffer.a)
    RHS_for_solution(dU, Umatrix, new_para)  # 计算残差
    return vec(dU)

end

maximum(abs.(pde_residual_debug(buffer_debug)))


function mesh_shock_objective_debug(buffer_debug; αmesh = 1.0, αshk = 1.0, area_of_ref_elem = 2, r_msh_ref = 4.0) where {T<:Real}

    ref = buffer_debug.ref
    area_of_ref_elem = sum(ref.H)
    G_ref = I(2) # ∂ ξ / ∂ ξ = I_2
    δ_ref = norm(G_ref) / det(G_ref)^0.5
    δref2 = δ_ref^2
    r_msh_ref = area_of_ref_elem * δ_ref^2


    u_vec = buffer_debug.initial_guess[buffer_debug.index_mesh+1:end]  # 使用最终解作为输入
    num_of_element = length(grid.VOL) # 因为可能要用自己写的 SimpleGrid， 这里先用 VOL 来算 element/node 数量
    num_of_nodes = size(grid.VOL[1][1])[1]

    xyz_gmsh = buffer_debug.initial_guess[1:buffer_debug.index_mesh]
    xyz_gmsh_coords = [Coord{2, Float64}((xyz_gmsh[i], xyz_gmsh[buffer_debug.index_mesh ÷ 2 + i])) for i in 1:buffer_debug.index_mesh÷2]
    current_grid = construct_simpleGrid(
        grid.cells, buffer_debug.ref, xyz_gmsh_coords, 
        grid.face_interfaces, grid.face_sets, buffer_debug.indices_moving_coords
    )
    
    # 计算每个单元的 mesh distortion
    r_mesh = zeros(Float64, num_of_element)
    for cell_id in 1:num_of_element
        xK = [coord[1] for coord in current_grid.xyz_SBP[cell_id]]
        tK = [coord[2] for coord in current_grid.xyz_SBP[cell_id]]
        xt = [xK, tK]
        GK = zeros(Float64, 2,2)
        for a in 1:2, b in 1:2
            GK[a,b] = mean(ref.D[a] * xt[b])  # 累加第 b 方向导数 * 物理坐标分量 a
        end
        # 修复：使用绝对值避免负数开方
        # 更robust的处理方式
        det_GK = det(GK)
        norm_GK = norm(GK)
        
        if abs(det_GK) < 1e-12
            # 接近奇异的情况，给一个很大的惩罚
            δK = T(1e6)
        elseif det_GK < 0
            # 网格翻转的情况，给一个很大的惩罚，但仍然可微
            δK = norm_GK / sqrt(-det_GK) + T(1e3)  # 添加惩罚项
        else
            # 正常情况
            δK = norm_GK / sqrt(det_GK)
        end
        r_msh_K = area_of_ref_elem * δK^2
        # r_mesh[cell_id] = r_msh_K - r_msh_ref  # 减去参考单元的失真度
        r_mesh[cell_id] = max(δK^2 - δref2, 0)
        print("cell_id: ", cell_id, " r_msh_K: ", r_msh_K, " r_msh_ref: ", r_msh_ref, " δK: ", δK, "\n")
    end
    
    # 计算每个单元的 shock tracking residual
    r_shk = zeros(Float64, num_of_element)
    Umatrix = reshape(u_vec, (num_of_element, num_of_nodes))
    for cell_id in 1:num_of_element
        num_uk = Umatrix[cell_id, :]
        quadrature_k = current_grid.geometric_terms.J_q[cell_id] .* diag(ref.H)
        area_K = area_of_ref_elem * current_grid.geometric_terms.J_q[cell_id][1]
        num_̄uk = sum(quadrature_k .* num_uk) / area_K
        r_shk[cell_id] = sum((num_uk .- num_̄uk).^2 .* quadrature_k)
    end

    total_mesh_distortion = αmesh * sum(r_mesh.^2)
    total_shock_tracking_residual = αshk * sum(r_shk.^2)

    print(
          "total mesh distortion: ", total_mesh_distortion, "\n",
          "total shock tracking residual: ", total_shock_tracking_residual, "\n"
           )
    # return αmesh * sum(r_mesh.^2) + αshk * sum(r_shk.^2)
    
end


mesh_shock_objective_debug(buffer_analytic,  αmesh = 0.001, αshk = 10000)



mesh_shock_objective_debug(buffer_debug; αmesh = 0.0001, αshk = 30000)


plot_u_interactive(grid, oscillating_u)

buffer_analytic = SpaceTimeBuffer(grid, [ax, at], oscillating_u)
buffer_analytic.initial_guess[10] = 0.1
xyz_gmsh_ana = buffer_analytic.initial_guess[1:buffer_analytic.index_mesh]
xyz_gmsh_coords_ana = [Coord{2, Float64}((xyz_gmsh_ana[i], xyz_gmsh_ana[buffer_analytic.index_mesh ÷ 2 + i])) for i in 1:buffer_analytic.index_mesh÷2]
simpleGrid_ana = construct_simpleGrid(
    grid.cells, buffer_analytic.ref, xyz_gmsh_coords_ana, 
    grid.face_interfaces, grid.face_sets, buffer_analytic.indices_moving_coords
)

plot_mesh(simpleGrid_ana)
analytic_U = zeros(Float64, (13, 12))
analytic_U[[3,4,6, 9, 11, 13], :] .= -1.0
analytic_U[[1,2,5, 7, 8, 10, 12], :] .= 1.0

buffer_analytic.initial_guess[27:end] = vec(analytic_U)

maximum(abs.(pde_residual_debug(buffer_analytic)))
plot_u_interactive(simpleGrid_ana, reshape(buffer_analytic.initial_guess[buffer_analytic.index_mesh+1:end], (13, 12)))


Umatrix = reshape(vec(buffer_analytic.initial_guess[buffer_analytic.index_mesh+1:end]), (13, 12))
dU = similar(Umatrix) 
fill!(dU, 0.0)  # 初始化 dU
new_para = (simpleGrid_ana, buffer_analytic.a)
RHS_for_solution(dU, Umatrix, new_para)  # 计算残差
maximum(abs.(dU))



u_vec_debug = buffer_analytic.initial_guess[buffer_analytic.index_mesh+1:end]  # 使用最终解作为输入

xyz_gmsh_debug = buffer_analytic.initial_guess[1:buffer_analytic.index_mesh]
xyz_gmsh_debug_coords = [Coord{2, Float64}((xyz_gmsh_debug[i], xyz_gmsh_debug[buffer_analytic.index_mesh ÷ 2 + i])) for i in 1:buffer_analytic.index_mesh÷2]
current_grid = construct_simpleGrid(
    grid.cells, buffer_analytic.ref, xyz_gmsh_debug_coords, 
    grid.face_interfaces, grid.face_sets, buffer_analytic.indices_moving_coords
)

plot_mesh(simpleGrid_ana)




buffer.indices_moving_coords = [10]
num_of_element = length(grid.VOL) # 因为可能要用自己写的 SimpleGrid， 这里先用 VOL 来算 element/node 数量
num_of_nodes = size(grid.VOL[1][1])[1]

num_of_xyz_gmsh = length(grid.xyz_gmsh)

# 计算 参考单元的 失真度
ref = buffer.ref
area_of_ref_elem = sum(ref.H)
G_ref = I(2) # ∂ ξ / ∂ ξ = I_2
δ_ref = norm(G_ref) / det(G_ref)^0.5
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

#-----------------------------------------------------------------------
# some plotting here
function bottomBC(x::Float64)
    if x <= -0.1
        return -cospi(x) - 0.5
    else
        return cospi(x) + 0.5
    end
end

ax, at = 1.0, 1.0

# get x coords of a grid at initial/final time for plotting
# for the grid : lin_adv_grid_nonalign_75_half, the cell_id has final time is 1 and 4, face_id are 1 and 3
TOP_INFLOW = [FaceIndex(1, 1), FaceIndex(4, 3)]
x_coords_final_before_opt = Dict()
for (cell_id, face_id) in TOP_INFLOW
    x_coords_final_before_opt[cell_id] = map(x -> x[1], ref.R[face_id] * grid.xyz_q[cell_id])
end


BOTTOM_INFLOW = collect(union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2")))
x_coords_initial_before_opt = Dict()
for (cell_id, face_id) in BOTTOM_INFLOW
    x_coords_initial_before_opt[cell_id] = map(x -> x[1], ref.R[face_id] * grid.xyz_q[cell_id])
end

# plot the analytical initial condition
using Plots
analytic_initial_condition = plot(x_coords_initial_before_opt[3], bottomBC.(x_coords_initial_before_opt[3]), label = "", xlims = (-0.5, 0.5))
scatter!(analytic_initial_condition, x_coords_initial_before_opt[3], bottomBC.(x_coords_initial_before_opt[3]), label = "")
plot!(analytic_initial_condition, x_coords_initial_before_opt[2], bottomBC.(x_coords_initial_before_opt[2]), label = "")
scatter!(analytic_initial_condition, x_coords_initial_before_opt[2], bottomBC.(x_coords_initial_before_opt[2]), label = "")
title!(analytic_initial_condition, "Analytic Initial Condition")

analytic_final_solution = plot(x_coords_final_before_opt[1], analytic_sol.(x_coords_final_before_opt[1]), label = "", xlims = (-0.5, 0.5))
scatter!(analytic_final_solution, x_coords_final_before_opt[1], analytic_sol.(x_coords_final_before_opt[1]), label = "")
plot!(analytic_final_solution, x_coords_final_before_opt[4], analytic_sol.(x_coords_final_before_opt[4]), label = "")
scatter!(analytic_final_solution, x_coords_final_before_opt[4], analytic_sol.(x_coords_final_before_opt[4]), label = "")
title!(analytic_final_solution, "Analytic Final Solution")

@load joinpath(@__DIR__, "lin_adv_oscillating_solution_sin.jld2") oscillating_u grid

numerical_solution_t0_oscillating =  plot(x_coords_initial_before_opt[3], ref.R[1] * oscillating_u[3, :], label = "", xlims = (-0.5, 0.5))
scatter!(numerical_solution_t0_oscillating, x_coords_initial_before_opt[3], ref.R[1] * oscillating_u[3, :], label = "")
plot!(numerical_solution_t0_oscillating, x_coords_initial_before_opt[2], ref.R[2] * oscillating_u[2, :], label = "")
scatter!(numerical_solution_t0_oscillating, x_coords_initial_before_opt[2], ref.R[2] * oscillating_u[2, :], label = "")
title!(numerical_solution_t0_oscillating, "Numerical solution at Initial Time \n before optimization")

numerical_solution_tF_oscillating = plot(x_coords_final_before_opt[1], ref.R[1] * oscillating_u[1, :], label = "", xlims = (-0.5, 0.5))
scatter!(numerical_solution_tF_oscillating, x_coords_final_before_opt[1], ref.R[1] * oscillating_u[1, :], label = "")
plot!(numerical_solution_tF_oscillating, x_coords_final_before_opt[4], ref.R[3] * oscillating_u[4, :], label = "")
scatter!(numerical_solution_tF_oscillating, x_coords_final_before_opt[4], ref.R[3] * oscillating_u[4, :], label = "")
title!(numerical_solution_tF_oscillating, "Numerical solution at Final Time \n before optimization")


# 把解析解 和 数值解（before optimization）画在一起对比，at t = tF
final_solution_comparision_before_opt = plot(xlims = (-0.5, 0.5), grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    plot!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], ref.R[face_id] * oscillating_u[cell_id, :], 
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], ref.R[face_id] * oscillating_u[cell_id, :], 
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], analytic_sol.(x_coords_final_before_opt[cell_id]),
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], analytic_sol.(x_coords_final_before_opt[cell_id]),
           label = "", color = :red, markersize = 4)
    first_red = false
end
title!(final_solution_comparision_before_opt, "Numerical vs Analytic at t = 0.2 \n before optimization")
xlabel!(final_solution_comparision_before_opt, "x")
ylabel!(final_solution_comparision_before_opt, "u")
savefig(final_solution_comparision_before_opt, joinpath(@__DIR__, "../plots_sin/linadv_final_sol_compare_before_opt_sin_problem.png"))

# 把解析解 和 数值解（before optimization）画在一起对比，at t = t0
initial_solution_comparision_before_opt = plot(xlims = (-0.5, 0.5), grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in BOTTOM_INFLOW
    plot!(initial_solution_comparision_before_opt, x_coords_initial_before_opt[cell_id], ref.R[face_id] * oscillating_u[cell_id, :], 
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(initial_solution_comparision_before_opt, x_coords_initial_before_opt[cell_id], ref.R[face_id] * oscillating_u[cell_id, :], 
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(initial_solution_comparision_before_opt, x_coords_initial_before_opt[cell_id], analytic_sol.(x_coords_initial_before_opt[cell_id], x2 = 0.0),
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(initial_solution_comparision_before_opt, x_coords_initial_before_opt[cell_id], analytic_sol.(x_coords_initial_before_opt[cell_id], x2 = 0.0),
           label = "", color = :red, markersize = 4)
    first_red = false
end 
title!(initial_solution_comparision_before_opt, "Numerical vs Analytic at t = 0 \n before optimization")
xlabel!(initial_solution_comparision_before_opt, "x")
ylabel!(initial_solution_comparision_before_opt, "u")
savefig(initial_solution_comparision_before_opt, joinpath(@__DIR__, "../plots_sin/linadv_initial_sol_compare_before_opt_sin_problem.png"))


@load joinpath(@__DIR__, "../optimization_results//buffer_two_coordinates_sin_problem.jld2") buffer final_simple_grid final_u
include(joinpath(@__DIR__, "../../src_simpleGrid/SimpleGrid.jl"))
include(joinpath(@__DIR__, "../plot_helper_simple_grid.jl"))
final_simple_grid, final_u = wrap_up_results(buffer)
plot_u_interactive(final_simple_grid, final_u)
plot_mesh(final_simple_grid)

x_coords_final_after_opt = Dict()
for (cell_id, face_id) in TOP_INFLOW
    x_coords_final_after_opt[cell_id] = map(x -> x[1], ref.R[face_id] * final_simple_grid.xyz_SBP[cell_id])
end 

x_coords_initial_after_opt = Dict()
for (cell_id, face_id) in BOTTOM_INFLOW
    x_coords_initial_after_opt[cell_id] = map(x -> x[1], ref.R[face_id] * final_simple_grid.xyz_SBP[cell_id])
end

initial_solution_comparision_after_opt = plot(xlims =  (-0.5, 0.5), grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in BOTTOM_INFLOW
    plot!(initial_solution_comparision_after_opt, x_coords_initial_after_opt[cell_id], ref.R[face_id] * final_u[cell_id, :],
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(initial_solution_comparision_after_opt, x_coords_initial_after_opt[cell_id], ref.R[face_id] * final_u[cell_id, :],
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(initial_solution_comparision_after_opt, x_coords_initial_after_opt[cell_id], analytic_sol.(x_coords_initial_after_opt[cell_id], x2 = 0.0),
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(initial_solution_comparision_after_opt, x_coords_initial_after_opt[cell_id], analytic_sol.(x_coords_initial_after_opt[cell_id], x2 = 0.0),
           label = "", color = :red, markersize = 4)
    first_red = false
end
title!(initial_solution_comparision_after_opt, "Numerical vs Analytic at t = 0 \n after optimization")
xlabel!(initial_solution_comparision_after_opt, "x")
ylabel!(initial_solution_comparision_after_opt, "u")
savefig(initial_solution_comparision_after_opt, joinpath(@__DIR__, "../plots_sin/linadv_initial_sol_compare_after_opt_sin_problem.png"))


final_solution_comparision_after_opt = plot(xlims = (-0.5, 0.5), grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    plot!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], ref.R[face_id] * final_u[cell_id, :],
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], ref.R[face_id] * final_u[cell_id, :],
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], analytic_sol.(x_coords_final_after_opt[cell_id]),
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], analytic_sol.(x_coords_final_after_opt[cell_id]),
           label = "", color = :red, markersize = 4)
    first_red = false
end
title!(final_solution_comparision_after_opt, "Numerical vs Analytic at t = 0.2 \n after optimization")
xlabel!(final_solution_comparision_after_opt, "x")
ylabel!(final_solution_comparision_after_opt, "u")
savefig(final_solution_comparision_after_opt, joinpath(@__DIR__, "../plots_sin/linadv_final_sol_compare_after_opt_sin_problem.png"))


plot_mesh(grid)

# plot the comparison of the mesh before and after optimization
using Plots.Measures
gr()
num_of_cells = length(grid.VOL)
num_of_nodes = size(grid.VOL[1][1], 1)
plot_size = (1200, 400)
xlimit = (-0.51, 0.51)
ylimit = (-0.01, 0.21)

mesh_comparision = scatter(xlabel = "x", ylabel = "t", title = "Mesh comparison", 
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
plot!(mesh_comparision, [-0.1, 0.1], [0.0, 0.2], label = "True shock", color = :red, markersize = 5)
mesh_comparision
savefig(mesh_comparision, joinpath(@__DIR__, "../plots_sin/linadv_mesh_comparision_sin_problem.png"))

mesh_nonaligned = scatter(xlabel = "x", ylabel = "t",
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
plot!(mesh_nonaligned, [-0.1, 0.1], [0.0, 0.2], label = "True shock", color = :red)
mesh_nonaligned
savefig(mesh_nonaligned, joinpath(@__DIR__, "../plots_sin/linadv_mesh_nonaligned_sin_problem.png")) 

mesh_after_opt = scatter(xlabel = "x", ylabel = "t",
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
plot!(mesh_after_opt, [-0.1, 0.1], [0.0, 0.2], label = "True shock", color = :red, markersize = 5)
mesh_after_opt
savefig(mesh_after_opt, joinpath(@__DIR__, "../plots_sin/linadv_mesh_after_opt_sin_problem.png"))

# plot error at final time
gr()
error_comparison_final = scatter(grid = false, title = "Error comparison at t = 0.2", xlabel = "x", ylabel = "Error")
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    scatter!(error_comparison_final, x_coords_final_before_opt[cell_id], abs.(analytic_sol.(x_coords_final_before_opt[cell_id]) - ref.R[face_id] * oscillating_u[cell_id, :]), color = :blue, label = first_blue ? "before" : "")
    scatter!(error_comparison_final, x_coords_final_after_opt[cell_id], abs.(analytic_sol.(x_coords_final_after_opt[cell_id]) - ref.R[face_id] * final_u[cell_id, :]), color = :red, label = first_red ? "after" : "")
    first_blue = false
    first_red = false
end
error_comparison_final
savefig(error_comparison_final, joinpath(@__DIR__, "../plots_sin/linadv_error_comparision_final_sin_problem.png"))

# plot error at initial time
error_comparison_initial = scatter(grid = false, title = "Error comparison at t = 0.0", xlabel = "x", ylabel = "Error", ylimit = (-0.0001, 0.01))
first_blue = true
first_red = true
for (cell_id, face_id) in BOTTOM_INFLOW
    scatter!(error_comparison_initial, x_coords_initial_before_opt[cell_id], abs.(analytic_sol.(x_coords_initial_before_opt[cell_id], x2 = 0.0) - ref.R[face_id] * oscillating_u[cell_id, :]), color = :blue, label = first_blue ? "before" : "")
    scatter!(error_comparison_initial, x_coords_initial_after_opt[cell_id], abs.(analytic_sol.(x_coords_initial_after_opt[cell_id], x2 = 0.0) - ref.R[face_id] * final_u[cell_id, :]), color = :red, label = first_red ? "after" : "")
    first_blue = false
    first_red = false
end
error_comparison_initial
savefig(error_comparison_initial, joinpath(@__DIR__, "../plots_sin/linadv_error_comparision_initial_sin_problem.png"))

function get_final_u_for_plotting_sol(grid, u)
    TOP_INFLOW = collect(union(get_face_set(grid, "TOP_INFLOW_1"), get_face_set(grid, "TOP_INFLOW_2")))
    ref = grid.cells[TOP_INFLOW[1][1]].ref_data[]
    num_sol = Dict()
    for (cell_id, face_id) in TOP_INFLOW
        R = ref.R[face_id]
        mask = ref.f_mask[face_id]
        temp_u = map(x -> x[1], R * u[cell_id, :])
        num_sol[cell_id] = temp_u[mask]
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