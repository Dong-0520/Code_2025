
using Pkg

using LinearAlgebra, SparseArrays, Trixi, BlockArrays
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics        

include("../../src/SBPLite.jl")
using .SBPLite

# order = 2
# ref = TriangleDiagELG(order, 2 * order)
# ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)

# grid = read_mesh(joinpath(@__DIR__, "lin_adv_grid__50.msh"), ref_elems_data, Base.identity)

@load joinpath(@__DIR__, "lin_adv_grid__50.jld2") grid

ax, at = (1.0, 1.0)

bottomBC1(x::Coord) = -cospi(x[1]) - 1
bottomBC2(x::Coord) = cospi(x[1]) + 1
leftBC(x::Coord) = -cospi(-1 - ax * x[2]) - 1




# para = (grid, [ax, at])

function set_up_U0(grid; perturbation = 1)
    U0 = zeros(Float64, (n_cells(grid), length(grid.xyz_q[1])))
    for cell_id in 1:n_cells(grid)
        x = [x[1] for x in grid.xyz_q[cell_id]]
        t = [x[2] for x in grid.xyz_q[cell_id]]
        if all(t .> x .+ 0.1 .- 1e-5)
            U0[cell_id, :] = -cospi.(x.-t) .- 1
        else
            U0[cell_id, :] = cospi.(x.-t) .+ 1
        end
    end
    return U0 .+ rand(size(U0)) .* perturbation
    
end

function plot_u_interactive(grid, us)
    # 返回交互式3D图
    plotlyjs()
    p = scatter3d(markersize=1, markercolor=:blue)

    for u in us
        for cell_id in 1:n_cells(grid)
            x = [x[1] for x in grid.xyz_q[cell_id]]
            t = [x[2] for x in grid.xyz_q[cell_id]]
            u_cell = u[cell_id, :]
            scatter3d!(p, x, t, u_cell, 
                    markersize=1, 
                    markercolor=:blue, 
                    label="")
        end
    end


    xlabel!(p, "x")
    ylabel!(p, "t")
    zlabel!(p, "u")
    title!(p, "Solution u")
    
    return p
end

function plot_u_2D(grid, u)
    p = plot()
    for cell_id in 1:n_cells(grid)
        x = [x[1] for x in grid.xyz_q[cell_id]]
        t = [x[2] for x in grid.xyz_q[cell_id]]
        u_cell = u[cell_id, :]
        # 画二维颜色图
        scatter!(p, x, t, zcolor=u_cell, markersize=5, label="", color=:viridis)
    end
    p
    
end


# using NonlinearSolve, LineSearches, ADTypes, SparseConnectivityTracer, SparseDiffTools, NLsolve
# U0_for_solver = set_up_U0(grid; perturbation = 1)

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
# plot_u_2D(grid, reshape(u0_vec, size(U0_for_solver)))
# plot_u_interactive(grid, reshape(u0_vec, size(U0_for_solver)))
# plot_u_interactive(grid, num_u)


#-----------------------------------------------------
using Pkg
Pkg.activate("MadNLP_env")

using JuMP, MadNLP, Random, MadNLPHSL



function RHS(du, u, p)

    grid, a = p

    for cell_id in 1:n_cells(grid)
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
        elem1 = grid.cells[c1].ref_data[]
        elem2 = grid.cells[c2].ref_data[]
        R_1, R_2 = elem1.R[lf1], elem2.R[lf2]
        mask1, mask2 = @views elem1.f_mask[lf1], elem2.f_mask[lf2]


        normal_1, normal_2 = @views grid.geometric_terms.N_f[c1][:, mask1], grid.geometric_terms.N_f[c2][:, mask2]

        an1, an2 = normal_1 .* a, normal_2 .* a
        lambda_1, lambda_2 = an1[1, :] + an1[2, :], an2[1, :] + an2[2, :]

        u1_face, u2_face = R_1 * u[c1, :], R_2 * u[c2, :]
        u1_adj, u2_adj = u2_face[P2], u1_face[P1]
        flux_1 = 0.5 * ((lambda_1 .+ abs.(lambda_1)) .* u1_face .+ (lambda_1 .- abs.(lambda_1)) .* u1_adj)
        flux_2 = 0.5 * ((lambda_2 .+ abs.(lambda_2)) .* u2_face .+ (lambda_2 .- abs.(lambda_2)) .* u2_adj)
        du[c1, :] += elem1.H_inv * Matrix(R_1)' * elem1.H_face * (lambda_1 .* u1_face .- flux_1)
        du[c2, :] += elem2.H_inv * Matrix(R_2)' * elem2.H_face * (lambda_2 .* u2_face .- flux_2)

    end


    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_1")
        cell = get_cells(grid, cell_id)
        elem = grid.cells[cell_id].ref_data[]
        R = elem.R[face_id]
        mask = @views elem.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        an = normal .* a
        gamma = an[1, :] + an[2, :]

        u_face = R * u[cell_id, :]
        u_adj = bottomBC1.(R * grid.xyz_q[cell_id])
        flux = gamma .* u_adj
        du[cell_id, :] += elem.H_inv * Matrix(R)' * elem.H_face * (gamma .* u_face .- flux)
    end

    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_2")
        cell = get_cells(grid, cell_id)
        elem = grid.cells[cell_id].ref_data[]
        R = elem.R[face_id]
        mask = @views elem.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        an = normal .* a
        gamma = an[1, :] + an[2, :]

        u_face = R * u[cell_id, :]
        u_adj = bottomBC2.(R * grid.xyz_q[cell_id])
        flux = gamma .* u_adj
        du[cell_id, :] += elem.H_inv * Matrix(R)' * elem.H_face * (gamma .* u_face .- flux)
    end

    for (cell_id, face_id) in get_face_set(grid, "LEFT_INFLOW")
        cell = get_cells(grid, cell_id)
        elem = grid.cells[cell_id].ref_data[]
        R = elem.R[face_id]
        mask = @views elem.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        an = normal .* a
        gamma = an[1, :] + an[2, :]

        u_face = R * u[cell_id, :]
        u_adj = leftBC.(R * grid.xyz_q[cell_id])
        flux = gamma .* u_adj
        du[cell_id, :] += elem.H_inv * Matrix(R)' * elem.H_face * (gamma .* u_face .- flux)
    end

    return nothing
end


U0_for_solver = set_up_U0(grid; perturbation = 1) * 0.0
para = (grid, [ax, at])


function pde_model(para, U0_for_solver; lin_solver = MadNLPHSL.Ma27Solver)
    # 使用 HSL 的线性求解器
      model = Model(() -> MadNLP.Optimizer(linear_solver=lin_solver))
    
    n_total = length(vec(U0_for_solver))
    u0_vec = vec(U0_for_solver)
    
    @variable(model, u[i=1:n_total], start = u0_vec[i])
    
    function residual_norm_squared(x::T...) where {T<:Real}
        u_vec = collect(x)  
        U_matrix = reshape(u_vec, size(U0_for_solver))
        du = similar(U_matrix, T)
        fill!(du, zero(T))  
        RHS(du, U_matrix, para)
        residual_vec = vec(du)
        return sum(residual_vec[i]^2 for i in eachindex(residual_vec))  
    end
    
    register(model, :residual_norm_squared, n_total, residual_norm_squared; autodiff = true)
    
    @NLobjective(model, Min, residual_norm_squared(u...))
    
    optimize!(model)
    return value.(u)
    
end



@time u_solution_ma27 = pde_model(para, U0_for_solver; lin_solver = Ma27Solver)
@time u_solution_ma57 = pde_model(para, U0_for_solver; lin_solver = Ma57Solver)
@time u_solution_ma77 = pde_model(para, U0_for_solver; lin_solver = Ma77Solver)
@time u_solution_Lapack = pde_model(para, U0_for_solver; lin_solver = MadNLPHSL.LapackCPUSolver)

# Plot the initial guess and solution
plot_u_interactive(grid, [U0_for_solver, reshape(u_solution_ma27, size(U0_for_solver))])
plot_u_interactive(grid, [U0_for_solver, reshape(u_solution_ma57, size(U0_for_solver))])
plot_u_interactive(grid, [U0_for_solver, reshape(u_solution_ma77, size(U0_for_solver))])
plot_u_interactive(grid, [U0_for_solver, reshape(u_solution_Lapack, size(U0_forSolver))])

# 总结
# 1. 使用 HSL 的线性求解器（Ma27, Ma57, Ma77）在非线性求解中表现良好，速度快且稳定。
# 2. 理论上来说 Ma57 是最适合我们的求解器
# 3. 本文档只讨论了如何把之前的space_time 问题转化为一个非线性优化问题，并使用 MadNLP 来求解。
# 4. RHS function 写法简单，不像 ExaModel那样需要 转换为 symbolic 表达式，使得代码过于复杂


# ----------------------------------------------------
# 这里我们测试一下一个struct 是否能够输入进solver里

# 这部分内容非常重要，在这里我们发现，对于optimization solver, 核心代码为如下两行：
# 1. 定义问题
# register(model, :residual_norm_squared, n_total, residual_norm_squared; autodiff = true)
# 2. 解决问题  
# optimize!(model)

# 关键约束：residual_norm_squared 函数只能接受 n_total 个独立的标量参数
# - 不能直接传入数组、struct 等复合类型
# - 必须支持自动微分（ForwardDiff 可能传入 Dual 类型）
# - 函数签名必须是 (x::T...) where {T<:Real}

# 解决方案：使用 solver 结构体作为"信息容器"和"参数解释器"
# 优势：
# 1. 类型稳定性：du = similar(U_matrix, T) 确保输出类型正确
# 2. 性能优化：避免重复构造临时对象和内存分配
# 3. 扩展性：为未来复杂问题预留设计空间
# 4. 可维护性：集中管理所有求解相关的状态和参数

# 未来扩展计划：
# 当需要同时优化 mesh coordinate 时，可以通过索引分组实现：
# solver.index_u = 1:n_solution              # 解变量的索引
# solver.index_mesh = (n_solution+1):n_total # 网格坐标的索引
# 
# 在 residual_norm_squared 中：
# u_vec = collect(x)[solver.index_u]         # 提取解变量
# mesh_vec = collect(x)[solver.index_mesh]   # 提取网格坐标





U0_for_solver = set_up_U0(grid; perturbation = 1) * 0.0

# 简化的求解器结构体
mutable struct SpaceTimeSolver
    grid::Any                           # 网格
    a::Vector{Float64}                  # 参数 [ax, at]
    initial_guess::Array{Float64, 2}    # 初始猜测
    final_solution::Array{Float64, 2}   # 最终解
    initial_objective::Float64          # 初始目标函数值
    final_objective::Float64            # 最终目标函数值
    final_residual::Float64             # 最终残差
    converged::Bool                     # 是否收敛
    solve_time::Float64                 # 求解时间
    
    # 构造函数
    function SpaceTimeSolver(grid, a::Vector{Float64}, initial_guess::Array{Float64, 2})
        new(grid, a, initial_guess, 
            copy(initial_guess),  # 初始解作为默认最终解
            Inf,                  # 初始目标函数值
            Inf,                  # 最终目标函数值
            Inf,                  # 最终残差
            false,                # 未收敛
            0.0)                  # 求解时间
    end
end

function RHS_with_solver(du, u, solver::SpaceTimeSolver)
    grid = solver.grid
    a = solver.a
    ax, at = a[1], a[2]
    
    for cell_id in 1:n_cells(grid)
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
        elem1 = grid.cells[c1].ref_data[]
        elem2 = grid.cells[c2].ref_data[]
        R_1, R_2 = elem1.R[lf1], elem2.R[lf2]
        mask1, mask2 = @views elem1.f_mask[lf1], elem2.f_mask[lf2]


        normal_1, normal_2 = @views grid.geometric_terms.N_f[c1][:, mask1], grid.geometric_terms.N_f[c2][:, mask2]

        an1, an2 = normal_1 .* a, normal_2 .* a
        lambda_1, lambda_2 = an1[1, :] + an1[2, :], an2[1, :] + an2[2, :]

        u1_face, u2_face = R_1 * u[c1, :], R_2 * u[c2, :]
        u1_adj, u2_adj = u2_face[P2], u1_face[P1]
        flux_1 = 0.5 * ((lambda_1 .+ abs.(lambda_1)) .* u1_face .+ (lambda_1 .- abs.(lambda_1)) .* u1_adj)
        flux_2 = 0.5 * ((lambda_2 .+ abs.(lambda_2)) .* u2_face .+ (lambda_2 .- abs.(lambda_2)) .* u2_adj)
        du[c1, :] += elem1.H_inv * Matrix(R_1)' * elem1.H_face * (lambda_1 .* u1_face .- flux_1)
        du[c2, :] += elem2.H_inv * Matrix(R_2)' * elem2.H_face * (lambda_2 .* u2_face .- flux_2)

    end


    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_1")
        cell = get_cells(grid, cell_id)
        elem = grid.cells[cell_id].ref_data[]
        R = elem.R[face_id]
        mask = @views elem.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        an = normal .* a
        gamma = an[1, :] + an[2, :]

        u_face = R * u[cell_id, :]
        u_adj = bottomBC1.(R * grid.xyz_q[cell_id])
        flux = gamma .* u_adj
        du[cell_id, :] += elem.H_inv * Matrix(R)' * elem.H_face * (gamma .* u_face .- flux)
    end

    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_2")
        cell = get_cells(grid, cell_id)
        elem = grid.cells[cell_id].ref_data[]
        R = elem.R[face_id]
        mask = @views elem.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        an = normal .* a
        gamma = an[1, :] + an[2, :]

        u_face = R * u[cell_id, :]
        u_adj = bottomBC2.(R * grid.xyz_q[cell_id])
        flux = gamma .* u_adj
        du[cell_id, :] += elem.H_inv * Matrix(R)' * elem.H_face * (gamma .* u_face .- flux)
    end

    for (cell_id, face_id) in get_face_set(grid, "LEFT_INFLOW")
        cell = get_cells(grid, cell_id)
        elem = grid.cells[cell_id].ref_data[]
        R = elem.R[face_id]
        mask = @views elem.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        an = normal .* a
        gamma = an[1, :] + an[2, :]

        u_face = R * u[cell_id, :]
        u_adj = leftBC.(R * grid.xyz_q[cell_id])
        flux = gamma .* u_adj
        du[cell_id, :] += elem.H_inv * Matrix(R)' * elem.H_face * (gamma .* u_face .- flux)
    end

    return nothing
    
end




# 便利构造函数
SpaceTimeSolver(grid, ax::Float64, at::Float64, initial_guess) = 
    SpaceTimeSolver(grid, [ax, at], initial_guess)

# 更新求解器状态的函数
function update_solver_results!(solver::SpaceTimeSolver, 
                               final_solution::Array{Float64, 2},
                               initial_obj::Float64,
                               final_obj::Float64,
                               final_residual::Float64,
                               solve_time::Float64)
    solver.final_solution = copy(final_solution)
    solver.initial_objective = initial_obj
    solver.final_objective = final_obj
    solver.final_residual = final_residual
    solver.converged = true
    solver.solve_time = solve_time
end


function pde_model_with_solver_struct(solver::SpaceTimeSolver; lin_solver = MadNLPHSL.Ma27Solver)
    model = Model(() -> MadNLP.Optimizer(
        linear_solver=lin_solver,
        print_level=MadNLP.INFO,
        max_iter=1000
    ))
    
    n_total = length(vec(solver.initial_guess))
    u0_vec = vec(solver.initial_guess)
    
    @variable(model, u[i=1:n_total], start = u0_vec[i])
    
    function residual_norm_squared(x::T...) where {T<:Real}
        u_vec = collect(x)  
        U_matrix = reshape(u_vec, size(solver.initial_guess))
        du = similar(U_matrix, T)
        fill!(du, zero(T))  
        
        RHS_with_solver(du, U_matrix, solver)
        
        residual_vec = vec(du)
        return sum(residual_vec[i]^2 for i in eachindex(residual_vec))  
    end
    
    register(model, :residual_norm_squared, n_total, residual_norm_squared; autodiff = true)
    
    @NLobjective(model, Min, residual_norm_squared(u...))
    
    # 记录初始状态
    initial_obj = residual_norm_squared(u0_vec...)
    println("🚀 开始优化，初始目标函数值: $initial_obj")
    
    # 计时求解
    solve_time = @elapsed begin
        optimize!(model)
    end
    
    # 获取最终解
    final_u = reshape(value.(u), size(solver.initial_guess))
    final_obj = objective_value(model)
    final_residual = sqrt(final_obj)
    
    # 更新求解器状态
    update_solver_results!(solver, final_u, initial_obj, final_obj, final_residual, solve_time)
    
    # 输出总结
    println("✅ 优化完成!")
    println("  求解时间: $(solve_time) 秒")
    println("  最终目标函数值: $final_obj")
    println("  目标函数减少: $(initial_obj - final_obj)")
    println("  减少比例: $(100 * (initial_obj - final_obj) / initial_obj)%")
    println("  是否收敛: $(solver.converged)")
    
    return solver
end

solution = pde_model_with_solver_struct(
    SpaceTimeSolver(grid, ax, at, U0_for_solver);
    lin_solver = Ma27Solver
)

# ----------------------------------------------------
# polynomial exactness test
for cell_id in 1:n_cells(grid)
    coeff1 = rand(1)
    coeff2 = rand(1)
    x = [x[1] for x in grid.xyz_q[cell_id]]
    t = [x[2] for x in grid.xyz_q[cell_id]]
    p = coeff1 .* x  .* t
    dpdx = coeff1 .* t
    dpdt = coeff1 .* x
    Dx = grid.VOL[cell_id][1]
    Dt = grid.VOL[cell_id][2]
    @test all(abs.(Dx * p .- dpdx) .< 1e-10)
    @test all(abs.(Dt * p .- dpdt) .< 1e-10)
end

include("../plot_helper.jl")


order = 3
ref = TriangleDiagELGL(order, 2 * order)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)

grid1_ICOSAHOM = read_mesh(joinpath(@__DIR__, "lin_adv_ICOSAHOM_1_8.msh"), ref_elems_data, Base.identity)
grid2_ICOSAHOM = read_mesh(joinpath(@__DIR__, "lin_adv_ICOSAHOM_2_8.msh"), ref_elems_data, Base.identity)
grid3_ICOSAHOM = read_mesh(joinpath(@__DIR__, "lin_adv_ICOSAHOM_3_8.msh"), ref_elems_data, Base.identity)



plot_SBP_mesh(grid1_ICOSAHOM, show_cell_ids = false)



 
p = scatter(xlabel = "x", ylabel = "t", 
                size=(2200, 300), aspect_ratio=:equal, xlims = (-1.01, 1.01), ylims = (-0.01, 0.21))
num_of_cells = n_cells(grid1_ICOSAHOM)
num_of_nodes = length(grid1_ICOSAHOM.xyz_q[1])
for cell_id in 1:num_of_cells
    vertices_IDS = vertices(grid1_ICOSAHOM.cells[cell_id])
    vertice1 = grid1_ICOSAHOM.xyz_gmsh[vertices_IDS[1]]
    vertice2 = grid1_ICOSAHOM.xyz_gmsh[vertices_IDS[2]]
    vertice3 = grid1_ICOSAHOM.xyz_gmsh[vertices_IDS[3]]

    v1x, v1y = vertice1[1], vertice1[2]
    v2x, v2y = vertice2[1], vertice2[2]
    v3x, v3y = vertice3[1], vertice3[2]

    center_x = (v1x + v2x + v3x) / 3
    center_y = (v1y + v2y + v3y) / 3


    plot!(p, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
            label="", color=:blue, xlabel = "x", ylabel = "t")

    x = [grid1_ICOSAHOM.xyz_q[cell_id][node_id][1] for node_id in 1:num_of_nodes]
    y = [grid1_ICOSAHOM.xyz_q[cell_id][node_id][2] for node_id in 1:num_of_nodes]
    scatter!(p, x, y, markersize=1, label="", markercolor=:red)
end
# 先去掉原来的标签
plot!(p, ylabel = "", xlabel = "", left_margin = 9Plots.mm,
                                    bottom_margin = 8.5Plots.mm,)

# 手动添加标签，精确控制位置
annotate!(p, [-1.08], [0.1], [text("t", 12, :black)])    # t 标签
annotate!(p, [0], [-0.05], [text("x", 12, :black)])     # x 标签

# 添加两个红点
scatter!(p, [0.5, 0.3], [0.0, 0.2], color=:red, markersize=10, label="")

# 用红线连接
plot!(p, [0.5, 0.3], [0.0, 0.2], color=:red, linewidth=3, label="")

display(p)

save(joinpath(@__DIR__, "lin_adv_ICOSAHOM_1_8_mesh.png"), p)

p2 = scatter(size=(2200, 300), aspect_ratio=:equal, xlims = (-1.01, 1.01), ylims = (0.19, 0.41))
num_of_cells = n_cells(grid2_ICOSAHOM)
num_of_nodes = length(grid2_ICOSAHOM.xyz_q[1])
for cell_id in 1:num_of_cells
    vertices_IDS = vertices(grid2_ICOSAHOM.cells[cell_id])
    vertice1 = grid2_ICOSAHOM.xyz_gmsh[vertices_IDS[1]]
    vertice2 = grid2_ICOSAHOM.xyz_gmsh[vertices_IDS[2]]
    vertice3 = grid2_ICOSAHOM.xyz_gmsh[vertices_IDS[3]]

    v1x, v1y = vertice1[1], vertice1[2]
    v2x, v2y = vertice2[1], vertice2[2]
    v3x, v3y = vertice3[1], vertice3[2]

    center_x = (v1x + v2x + v3x) / 3
    center_y = (v1y + v2y + v3y) / 3


    plot!(p2, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
            label="", color=:blue, xlabel = "x", ylabel = "t")

    x = [grid2_ICOSAHOM.xyz_q[cell_id][node_id][1] for node_id in 1:num_of_nodes]
    y = [grid2_ICOSAHOM.xyz_q[cell_id][node_id][2] for node_id in 1:num_of_nodes]
    scatter!(p2, x, y, markersize=1, label="", markercolor=:red)
end
# 添加两个红点
scatter!(p2, [0.3, 0.1], [0.2, 0.4], color=:red, markersize=10, label="")

# 用红线连接
plot!(p2, [0.3, 0.1], [0.2, 0.4], color=:red, linewidth=3, label="")

save(joinpath(@__DIR__, "lin_adv_ICOSAHOM_2_8_mesh.png"), p2)

p3 = scatter(size=(2200, 300), aspect_ratio=:equal, xlims = (-1.01, 1.01), ylims = (0.39, 0.61))
num_of_cells = n_cells(grid3_ICOSAHOM)
num_of_nodes = length(grid3_ICOSAHOM.xyz_q[1])
for cell_id in 1:num_of_cells
    vertices_IDS = vertices(grid3_ICOSAHOM.cells[cell_id])
    vertice1 = grid3_ICOSAHOM.xyz_gmsh[vertices_IDS[1]]
    vertice2 = grid3_ICOSAHOM.xyz_gmsh[vertices_IDS[2]]
    vertice3 = grid3_ICOSAHOM.xyz_gmsh[vertices_IDS[3]]

    v1x, v1y = vertice1[1], vertice1[2]
    v2x, v2y = vertice2[1], vertice2[2]
    v3x, v3y = vertice3[1], vertice3[2]

    center_x = (v1x + v2x + v3x) / 3
    center_y = (v1y + v2y + v3y) / 3


    plot!(p3, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
            label="", color=:blue, xlabel = "x", ylabel = "t")

    x = [grid3_ICOSAHOM.xyz_q[cell_id][node_id][1] for node_id in 1:num_of_nodes]
    y = [grid3_ICOSAHOM.xyz_q[cell_id][node_id][2] for node_id in 1:num_of_nodes]
    scatter!(p3, x, y, markersize=1, label="", markercolor=:red)
end
# 添加两个红点
scatter!(p3, [0.1, -0.1], [0.4, 0.6], color=:red, markersize=10, label="")

# 用红线连接
plot!(p3, [0.1, -0.1], [0.4, 0.6], color=:red, linewidth=3, label="")

save(joinpath(@__DIR__, "lin_adv_ICOSAHOM_3_8_mesh.png"), p3)