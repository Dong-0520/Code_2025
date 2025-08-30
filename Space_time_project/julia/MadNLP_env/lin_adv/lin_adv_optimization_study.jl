"""
一些杂乱的代码总结了一下如何使用JuMP和MadNLP来建立和求解优化问题。
包括了最初我们打算如何设计mesh quality and measure of oscillation
以及建立NLconstraints的思路
成熟的代码请去子文件夹查看
"""



using Pkg

using LinearAlgebra, SparseArrays, Trixi, BlockArrays, ArraysOfArrays
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics        
using ForwardDiff

include("../../src/SBPLite.jl")
# include(joinpath(@__DIR__, "plot_helper.jl"))
using .SBPLite

order = 2
ref = TriangleDiagELG(order, 2 * order)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)

# grid = read_mesh(joinpath(@__DIR__, "lin_adv_grid_nonalign_50.msh"), ref_elems_data, Base.identity)


# @save joinpath(@__DIR__, "lin_adv_grid_nonalign_50.jld2") grid
@load joinpath(@__DIR__, "lin_adv_grid_nonalign_50.jld2") grid

ax, at = (1.0, 1.0)

# bottomBC1(x::Coord) = -cospi(x[1]) - 1
# bottomBC2(x::Coord) = cospi(x[1]) + 1
# leftBC(x::Coord) = -cospi(-1 - ax * x[2]) - 1

bottomBC1(x::Coord) = -1
bottomBC2(x::Coord) = 1
function bottomBC(x::Coord)
    if x[1] <= -0.1
        return -1
    else
        return 1
    end
end
leftBC(x::Coord) = -1

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



using NonlinearSolve, LineSearches, ADTypes, SparseConnectivityTracer, SparseDiffTools, NLsolve
U0_for_solver = zeros(Float64, (n_cells(grid), length(grid.xyz_q[1])))
para = (grid, [ax, at])
# u0_vec = vec(U0_for_solver)
# du = similar(U0_for_solver); fill!(du, 0.0)

function f_nl!(F, u)
    U = reshape(u, size(U0_for_solver))
    du = similar(U); fill!(du, 0.0)
    RHS_for_solution(du, U, para)      # 你的核心残差
    F[:] = vec(du)
end



res = nlsolve(
  f_nl!,  vec(U0_for_solver);
  method    = :trust_region,
  linsolve  = :dogleg,      # :cg
  autodiff  = :central,   
  xtol      = 1e-6,
  ftol      = 1e-6,
  autoscale = true,
  show_trace= true,
  iterations = 150,
)

oscillating_u = reshape(res.zero, size(U0_for_solver))
plot_u_2D(grid, oscillating_u)
plot_u_interactive(grid, [reshape(oscillating_u, size(U0_for_solver))])

# @save joinpath(@__DIR__, "lin_adv_oscillating_solution.jld2") oscillating_u grid
# @load joinpath(@__DIR__, "lin_adv_oscillating_solution.jld2") oscillating_u grid

# now we have a solution on a mesh that the interfaces not aligned with shock, this mesh and solution
# can be used to the initial guess for the next optimization problem

# 这是之前的笔记和代码，复制过来，接下来我们就要开始尝试如何把整个optimization problem 建立起来并解决了
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

# 解决方案：使用 solver 结构体作为"信息容器"和"参数解释器" (这里见下面，我们已经尝试要用SpaceTimeBuffer 存信息)
# 优势：
# 1. 类型稳定性：du = similar(U_matrix, T) 确保输出类型正确
# 2. 性能优化：避免重复构造临时对象和内存分配
# 3. 扩展性：为未来复杂问题预留设计空间
# 4. 可维护性：集中管理所有求解相关的状态和参数

# 未来扩展计划：
# 当需要同时优化 mesh coordinate 时，可以通过索引分组实现：
# solver.index_mesh = 1:n_solution # 网格坐标的索引 
# solver.index_u = (n_solution+1):n_total             # 解变量的索引

# 在 residual_norm_squared 中：
# u_vec = collect(x)[solver.index_u]         # 提取解变量
# mesh_vec = collect(x)[solver.index_mesh]   # 提取网格坐标

#-----------------------------------------------------
Pkg.activate("MadNLP_env")

using JuMP, MadNLP, Random, MadNLPHSL
@load joinpath(@__DIR__, "lin_adv_oscillating_solution.jld2") oscillating_u grid


# 简化的求解器结构体
# This should contain all necessary parameters and informations
# initial_guess: a vector of both all of xyz_gmsh coordinates and the oscillating solution
# index_mesh: an integer indicating the first n index of initial guess is mesh coordinate, 
# 网格上的解的index就是除了前index_mesh 的后面部分
# index_moving_coordinates: an array of indices indicating which coordinates are moving(passed to optimization solver)
# final_solution: the final solution after optimization
# initial_objective: the initial objective function value
# final_objective: the final objective function value
# final_residual: the final residual norm
# converged: whether the optimization has converged
# solve_time: the time taken to solve the optimization problem
mutable struct SpaceTimeBuffer
    grid::Any                           # 网格
    a::Vector{Float64}                  # 参数 [ax, at]
    initial_guess::Array{Float64, 1}    # 初始猜测, 包括所有的动的和不动的xyz_gms坐标，和 oscillating_u 作为最开始的解
    index_mesh::Int                      # 网格坐标的索引， initial_guess[1:index_mesh] 是所有 xyz_gmsh 坐标
    indices_moving_coords::Array{Int64, 1} # 可移动坐标的索引 initial_guess[index_moving_coordinates] 是所有动的xyz_gmsh 坐标, 这些也是要在optimization solver中更新的解
    xL::Float64                     # 左边界 xL
    xR::Float64                     # 右边界 xR
    t0::Float64                     # 初始时间 t0
    tF::Float64                     # 结束时间 tF
    final_solution::Array{Float64, 1}   # 最终解
    initial_objective::Float64          # 初始目标函数值
    final_objective::Float64            # 最终目标函数值
    final_residual::Float64             # 最终残差
    converged::Bool                     # 是否收敛
    solve_time::Float64                 # 求解时间
    ref::SBPLite.RefElemData
    
    # 构造函数
    function SpaceTimeBuffer(grid, a::Vector{Float64}, oscillating_u::Array{Float64, 2})
        indices_moving_coords, coords_xyz_gmsh, t0, tF, xL, xR = get_meshinfo_for_buffer(grid)
        initial_guess = vcat(coords_xyz_gmsh, vec(oscillating_u))
        index_mesh = length(coords_xyz_gmsh)  # 网格坐标的索引
        ref = grid.cells[1].ref_data[]  # 获取参考元素数据
        new(grid, a, 
        deepcopy(initial_guess), 
        index_mesh, 
        indices_moving_coords,  # 可移动坐标的索引
        xL, xR, t0, tF,
        zeros(Float64, length(initial_guess)),  # 最终解初始化为零
        Inf, Inf, Inf, false, Inf, ref)  # 初始目标函数值、最终目标函数值、最终残差、是否收敛、求解时间
    end
end


# 这个struct 方便我们每次更新了 mesh坐标之后，能够更新 SBP operators
mutable struct SimpleGrid{dim, C <: AbstractCell, T <: Real}
    cells::Vector{C}
    ref::RefElemData
    xyz_gmsh::Vector{Coord{dim, T}}
    xyz_SBP::VectorOfArrays{Coord{dim, T}, 1}
    face_interfaces::Vector{FaceInterface}
    face_sets::Dict{String, Set{FaceIndex}}
    geometric_terms::GeometricTerms{T}
    VOL::Vector{NTuple{dim, Matrix{T}}}

    xyz_gmsh_vec::Vector{T}              # 展平的坐标向量
    indices_moving_coords::Vector{Int64}       # 可动坐标的索引
end

"""
Designed for one-dimensional space-time slab,
return t0, tF, xL, xR
"""
function find_boundaries(simple_grid::Grid)
    t0 = minimum([coord[2] for coord in simple_grid.xyz_gmsh])
    tF = maximum([coord[2] for coord in simple_grid.xyz_gmsh])
    xL = minimum([coord[1] for coord in simple_grid.xyz_gmsh])
    xR = maximum([coord[1] for coord in simple_grid.xyz_gmsh])
    return t0, tF, xL, xR
end


"""
This function returns the total number of coordinates in the xyz_gmsh
    so the first 
"""
function get_meshinfo_for_buffer(simple_grid::Grid; dim = 2, tol = 1e-8)
    t0, tF, xL, xR = find_boundaries(simple_grid)
    num_of_points = length(simple_grid.xyz_gmsh)
    num_of_coords = dim * num_of_points
    x = [coord[1] for coord in simple_grid.xyz_gmsh]
    t = [coord[2] for coord in simple_grid.xyz_gmsh]
    coords = vcat(x, t)
    indices = Vector{Int64}()
    for i in 1:num_of_points
        curr_point = simple_grid.xyz_gmsh[i]
        xi = curr_point[1]
        ti = curr_point[2]
        # 如果点在 t = t0, tF 边界上，那么只有x可以动
        # 如果点在 x= xL, xR 边界上，那么只有t可以动
        # 如果点在四个corner上，那么x t 都不可以动都不能放在indices里
        if (isapprox(xi, xL, atol=tol) && isapprox(ti, t0, atol=tol)) || (isapprox(xi, xR, atol=tol) && isapprox(ti, tF, atol=tol)) || (isapprox(xi, xL, atol=tol) && isapprox(ti, tF, atol=tol)) || (isapprox(xi, xR, atol=tol) && isapprox(ti, t0, atol=tol))
            # 左下角
            continue  # 不能动
        elseif isapprox(ti, t0, atol=tol) || isapprox(ti, tF, atol=tol)
            # 在上边界或下边界
            push!(indices, i)  # 只能动x
        elseif isapprox(xi, xL, atol=tol) || isapprox(xi, xR, atol=tol)
            # 在左边界或右边界
            push!(indices, i + num_of_points)  # 只能动t
        else
            # 在内部
            push!(indices, i)  # x可以动
            push!(indices, i + num_of_points)  # t可以动
        end
    end
    return indices, coords, t0, tF, xL, xR
end



function construct_simpleGrid(cells::Vector{C},
                            ref::RefElemData, 
                            new_xyz_gmsh::Vector{Coord{dim, T}}, 
                            face_interfaces::Vector{FaceInterface},
                            face_sets::Dict{String, Set{FaceIndex}}, 
                            indices_moving_coords::Vector{Int64},; 
                            comp_coords::Vector{SVector{dim, T}} = 
                                        [SVector{dim, T}(-1.0, -1.0),
                                        SVector{dim, T}(1.0, -1.0),
                                        SVector{dim, T}(-1.0, 1.0)],
                            basis = MonomialType, 
                            gmsh_order = 1) where {dim, C, T<: Real} 

    # 重新计算映射
    n_cells = length(cells)
    mapping_ref_phys = Vector{PolynomialCurvilinearMapping}()
    
    for i in 1:n_cells
        phys_coords = new_xyz_gmsh[collect(cells[i].nodes)]
        feats = reduce(hcat, SBPLite.basis_functions(basis, comp_coords, gmsh_order))' |> Matrix
        coeffs = Tuple(feats \ get_i_coordinates(phys_coords, i) for i in 1:dim)
        push!(mapping_ref_phys, PolynomialCurvilinearMapping{dim, T, basis}(coeffs))
    end
    
    # 初始化数据结构
    xyz_SBP = Vector{Vector{Coord{dim, T}}}(undef, n_cells)
    Λ_q = Vector{Array{T, 3}}(undef, n_cells)
    Λ_f = Vector{Array{T, 3}}(undef, n_cells)
    J_f = Vector{Vector{T}}(undef, n_cells)
    J_q = Vector{Vector{T}}(undef, n_cells)
    N_f = Vector{Array{T, 2}}(undef, n_cells)
    VOL = Vector{NTuple{dim, Matrix{T}}}(undef, n_cells)
    # FAC = Vector(undef, n_cells)

    # 转换参考坐标为正确的泛型类型
    rst_q_coords = [SVector{dim, T}(ref.rst_q[:, i]) for i in 1:size(ref.rst_q, 2)]
    rst_f_coords = [SVector{dim, T}(ref.rst_f[:, i]) for i in 1:size(ref.rst_f, 2)]
    
    @inbounds for i in 1:n_cells
                # 物理坐标 - 使用转换后的坐标
        xyz_SBP[i] = comp_to_phys(mapping_ref_phys[i], rst_q_coords)
        
        # 面几何项 - 使用转换后的坐标
        Λ_f[i], J_f[i] = metric_terms_exact(mapping_ref_phys[i], rst_f_coords)
        J_f[i] = abs.(J_f[i]) # make sure the Jacobian is positive
        N_f[i] = zeros(T, dim, size(ref.rst_f, 2))
        for m in 1:dim
            N_f[i][m, :] .= sum([Λ_f[i][:, n, m] .* ref.n_rst[n, :] for n in 1:dim])
        end
        
        E = compute_E_phys(ref, N_f[i])
            # 修复：确保 E 是正确的类型，并使用正确的函数签名
        if isa(E, Tuple)
            # 使用元组版本的函数
            Λ_q[i], J_q[i] = metric_terms_optimised(mapping_ref_phys[i], rst_q_coords, E, ref)
        else
            # 确保 E 是 Array{T} 类型
            E_array = Array{T}(E)
            # 确保其他参数也是正确的类型
            Qt_T = Matrix{T}(ref.Qt)
            Qt_inv_T = Matrix{T}(ref.Qt_inv)
            Λ_q[i], J_q[i] = metric_terms_optimised(mapping_ref_phys[i], rst_q_coords, E_array, Qt_T, Qt_inv_T)
        end
        
        J_q[i] = abs.(J_q[i]) # make sure the Jacobian is positive
        VOL[i] = compute_VOL_phys(ref, Λ_q[i], J_q[i], E)
    end
    
    # 创建几何项结构
    geometric_terms = GeometricTerms(VectorOfArrays.((J_q, Λ_q, J_f, Λ_f, N_f))...)
    
    x = [coord[1] for coord in new_xyz_gmsh]
    t = [coord[2] for coord in new_xyz_gmsh]
    xyz_gmsh_vec = vcat(x, t)

    # 构造并返回新的 SimpleGrid
    return SimpleGrid{dim, C, T}(
        cells,
        ref,
        new_xyz_gmsh,
        VectorOfArrays{Coord{dim, T}, 1}(xyz_SBP),
        face_interfaces,
        face_sets,
        geometric_terms,
        VOL,
        xyz_gmsh_vec,
        indices_moving_coords
    )
    
end

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
        # elem1 = grid.cells[c1].ref_data[]
        # elem2 = grid.cells[c2].ref_data[]
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


    # # for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_1")
    # for (cell_id, face_id) in grid.face_sets["BOTTOM_INFLOW_1"]

    #     R = ref.R[face_id]
    #     mask = @views ref.f_mask[face_id]
    #     normal = grid.geometric_terms.N_f[cell_id][:, mask]
    #     an = normal .* a
    #     gamma = an[1, :] + an[2, :]

    #     u_face = R * u[cell_id, :]
    #     u_adj = bottomBC1.(R * grid.xyz_SBP[cell_id])
    #     flux = gamma .* u_adj
    #     du[cell_id, :] += ref.H_inv * Matrix(R)' * ref.H_face * (gamma .* u_face .- flux)
    # end

    # # for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_2")
    # for (cell_id, face_id) in grid.face_sets["BOTTOM_INFLOW_2"]

    #     R = ref.R[face_id]
    #     mask = @views ref.f_mask[face_id]
    #     normal = grid.geometric_terms.N_f[cell_id][:, mask]
    #     an = normal .* a
    #     gamma = an[1, :] + an[2, :]

    #     u_face = R * u[cell_id, :]
    #     u_adj = bottomBC2.(R * grid.xyz_SBP[cell_id])
    #     flux = gamma .* u_adj
    #     du[cell_id, :] += ref.H_inv * Matrix(R)' * ref.H_face * (gamma .* u_face .- flux)
    # end

    # for (cell_id, face_id) in get_face_set(grid, "LEFT_INFLOW")
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



# t0, tF, xL, xR = find_boundaries(grid)
# indices_moving_coords, coords_xyz_gmsh, t0, tF, xL, xR = get_meshinfo_for_buffer(grid)

buffer = SpaceTimeBuffer(grid, [ax, at], oscillating_u)


# 之后再想办法 优化网格更新
function pde_model_with_solver_struct_1(buffer::SpaceTimeBuffer; lin_solver = MadNLPHSL.Ma57Solver, 
                                                                αmesh = 1.0, αshk = 1.0, max_iter = 100)
    model = Model(() -> MadNLP.Optimizer(
        linear_solver=lin_solver,
        print_level=MadNLP.INFO,
        max_iter=max_iter
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
    r_msh_ref = area_of_ref_elem * δ_ref^2
    
    n_total = length(buffer.indices_moving_coords) + length(buffer.initial_guess) - buffer.index_mesh

    # 储存最开始的 xyz_gmsh 坐标, 在 solver 迭代更新的过程中，可以保证只有 能动的在动，不能动的一直没变
    # 用 Ref 包装，使其可以在函数内部修改
    current_grid = Ref{SimpleGrid}()
    current_xyz_gmsh = Ref{Vector{Float64}}()
    initial_xyz_gmsh = buffer.initial_guess[1:buffer.index_mesh]
    new_xyz_gmsh = deepcopy(initial_xyz_gmsh) # 初始化 new_xyz_gmsh 为初始值
    new_xyz_gmsh_coords = [Coord{2, Float64}((new_xyz_gmsh[i], new_xyz_gmsh[num_of_xyz_gmsh + i])) for i in 1:num_of_xyz_gmsh]
    current_grid[] = construct_simpleGrid(
        grid.cells, buffer.ref, new_xyz_gmsh_coords, 
        grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
    )
            

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

        if T <: ForwardDiff.Dual
            # 🔧 对于Dual类型，始终重建以避免缓存冲突
            mesh_vec_float = [ForwardDiff.value(i) for i in mesh_vec]
            println("🔄 Dual: 强制重建网格 (避免缓存冲突)")
            
            new_xyz_gmsh = Vector{Float64}(initial_xyz_gmsh)
            new_xyz_gmsh[buffer.indices_moving_coords] = mesh_vec_float
            new_xyz_gmsh_coords = [Coord{2, Float64}((new_xyz_gmsh[i], new_xyz_gmsh[num_of_xyz_gmsh + i])) for i in 1:num_of_xyz_gmsh]
            
            current_grid_local = construct_simpleGrid(
                grid.cells, buffer.ref, new_xyz_gmsh_coords,
                grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
            )
            
        elseif T == Float64
            # Float64：仍可使用缓存优化
            coord_diff = maximum(abs.(current_grid[].xyz_gmsh_vec[current_grid[].indices_moving_coords] .- mesh_vec))
            if coord_diff > 1e-12
                println("🔄 Float64: 更新网格 (差异: $coord_diff)")
                current_xyz_gmsh[][current_grid[].indices_moving_coords] .= mesh_vec
                new_xyz_gmsh_coords = [Coord{2, Float64}((current_xyz_gmsh[][i], current_xyz_gmsh[][num_of_xyz_gmsh + i])) for i in 1:num_of_xyz_gmsh]
                current_grid[] = construct_simpleGrid(
                    grid.cells, buffer.ref, new_xyz_gmsh_coords,
                    grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
                )
            end
            current_grid_local = current_grid[]
        end


        Umatrix = reshape(u_vec, (num_of_element, num_of_nodes))
        dU = similar(Umatrix, T) 
        fill!(dU, zero(T))  # 初始化 dU
        new_para = (current_grid_local, buffer.a)
        RHS_for_solution(dU, Umatrix, new_para)  # 计算残差
        return vec(dU)

    end

    function mesh_shock_objectivex(x::T...) where {T<:Real}
        mesh_u_vec = collect(x)  
        mesh_vec = mesh_u_vec[1: length(buffer.indices_moving_coords)] # 前几个放的是可以动的坐标
        u_vec = mesh_u_vec[(length(buffer.indices_moving_coords) + 1):end]  # 后面放的是解变量

        if T <: ForwardDiff.Dual
            # 🔧 对于Dual类型，始终重建以避免缓存冲突
            mesh_vec_float = [ForwardDiff.value(i) for i in mesh_vec]
            println("🔄 Dual: 强制重建网格 (避免缓存冲突)")
            
            new_xyz_gmsh = Vector{Float64}(initial_xyz_gmsh)
            new_xyz_gmsh[buffer.indices_moving_coords] = mesh_vec_float
            new_xyz_gmsh_coords = [Coord{2, Float64}((new_xyz_gmsh[i], new_xyz_gmsh[num_of_xyz_gmsh + i])) for i in 1:num_of_xyz_gmsh]
            
            current_grid_local = construct_simpleGrid(
                grid.cells, buffer.ref, new_xyz_gmsh_coords,
                grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
            )
            
        elseif T == Float64
            # Float64：仍可使用缓存优化
            coord_diff = maximum(abs.(current_grid[].xyz_gmsh_vec[current_grid[].indices_moving_coords] .- mesh_vec))
            if coord_diff > 1e-12
                println("🔄 Float64: 更新网格 (差异: $coord_diff)")
                current_xyz_gmsh[][current_grid[].indices_moving_coords] .= mesh_vec
                new_xyz_gmsh_coords = [Coord{2, Float64}((current_xyz_gmsh[][i], current_xyz_gmsh[][num_of_xyz_gmsh + i])) for i in 1:num_of_xyz_gmsh]
                current_grid[] = construct_simpleGrid(
                    grid.cells, buffer.ref, new_xyz_gmsh_coords,
                    grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
                )
            end
            current_grid_local = current_grid[]
        end
        
        # 计算每个单元的 mesh distortion
        r_mesh = zeros(T, num_of_element)
        for cell_id in 1:num_of_element
            xK = [coord[1] for coord in current_grid_local.xyz_SBP[cell_id]]
            tK = [coord[2] for coord in current_grid_local.xyz_SBP[cell_id]]
            xt = [xK, tK]
            GK = zeros(T, 2,2)
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
            r_mesh[cell_id] = r_msh_K - r_msh_ref  # 减去参考单元的失真度
        end
        
        # 计算每个单元的 shock tracking residual
        r_shk = zeros(T, num_of_element)
        Umatrix = reshape(u_vec, (num_of_element, num_of_nodes))
        for cell_id in 1:num_of_element
            num_uk = Umatrix[cell_id, :]
            quadrature_k = current_grid_local.geometric_terms.J_q[cell_id] .* diag(ref.H)
            area_K = area_of_ref_elem * current_grid_local.geometric_terms.J_q[cell_id][1]
            num_̄uk = sum(quadrature_k .* num_uk) / area_K
            r_shk[cell_id] = sum((num_uk .- num_̄uk).^2 .* quadrature_k)
        end

        return αmesh * sum(r_mesh.^2) + αshk * sum(r_shk.^2)
        
    end

    # 注册函数
    register(model, :pde_residual, n_total, pde_residual; autodiff = true)
    register(model, :mesh_shock_objectivex, n_total, mesh_shock_objectivex; autodiff = true)

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
    @NLobjective(model, Min, mesh_shock_objectivex(u...))

    # ---------------------------------------------------
    # 记录初始状态
    initial_obj = mesh_shock_objectivex(u0_vec...)
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

    return buffer  # 返回更新后的 buffer
end


function pde_model_with_solver_struct_2(buffer::SpaceTimeBuffer; lin_solver = MadNLPHSL.Ma57Solver, 
                                                                αmesh = 1.0, αshk = 1.0, max_iter = 100)
    model = Model(() -> MadNLP.Optimizer(
        linear_solver=lin_solver,
        print_level=MadNLP.INFO,
        max_iter=max_iter
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
            r_mesh[cell_id] = r_msh_K - r_msh_ref  # 减去参考单元的失真度
        end
        
        # 计算每个单元的 shock tracking residual
        r_shk = zeros(T, num_of_element)
        Umatrix = reshape(u_vec, (num_of_element, num_of_nodes))
        for cell_id in 1:num_of_element
            num_uk = Umatrix[cell_id, :]
            quadrature_k = current_grid.geometric_terms.J_q[cell_id] .* diag(ref.H)
            area_K = area_of_ref_elem * current_grid.geometric_terms.J_q[cell_id][1]
            num_̄uk = sum(quadrature_k .* num_uk) / area_K
            r_shk[cell_id] = sum((num_uk .- num_̄uk).^2 .* quadrature_k)
        end

        return αmesh * sum(r_mesh.^2) + αshk * sum(r_shk.^2)
        
    end

    # 注册函数
    register(model, :pde_residual, n_total, pde_residual; autodiff = true)
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

    # return buffer  # 返回更新后的 buffer
end

buffer = SpaceTimeBuffer(grid, [ax, at], oscillating_u)
pde_model_with_solver_struct_2(buffer, αmesh = 0.01, αshk = 10000)



buffer.final_objective
buffer.final_solution

initial_guess = buffer.initial_guess
initial_guess[buffer.indices_moving_coords] = buffer.final_solution[1:length(buffer.indices_moving_coords)]

new_xyz_gmsh = initial_guess[1:buffer.index_mesh]
new_xyz_gmsh_coords = [Coord{2, Float64}((new_xyz_gmsh[i], new_xyz_gmsh[buffer.index_mesh ÷ 2 + i])) for i in 1:13]
final_simple_grid = construct_simpleGrid(
    grid.cells, buffer.ref, new_xyz_gmsh_coords, 
    grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
)
final_u = reshape(buffer.final_solution[length(buffer.indices_moving_coords)+1:end], (length(grid.VOL), size(grid.VOL[1][1])[1]))


plot_mesh(final_simple_grid)
plot_u_interactive(final_simple_grid, final_u)

plot_u_interactive(grid, oscillating_u)






# 这部分代码介绍了如何计算某个单元的 mesh distortion
# G = ∂X / ∂ξ = [ ∂x/∂ξ ∂t/∂ξ; ∂x/∂η ∂t/∂η ] is the mapping Jacobian Matrix
# such that det(G) 应该与 grid.geometric_terms.J_q[cell_id] 储存的值相等
# 并且因为我们暂时用的是 线性 映射 从 ref element to physical element, 所以 J_q 值都相等
# 又因为我们的 ref element 的面积是 2， 所以对于某个单元的 mesh distortion 定义为
# r_mesh[cell_id] = 2 * (norm(G) / det(G)^0.5)^2
# 几何意义上来说，分子代表 衡量所有方向的总伸缩 量，就是不要在某个方向过于拉伸
# 而分母代表 是体积（或二维下的面积）放缩因子，即 physical element 的面积是 ref element面积的 det(G) 倍
# 要让所有方向都均匀放大到这个面积，单个方向就要放大 det(G)^dim 倍
# 所以 r_mesh[cell_id] 接近 1 时说明 时表示“只有均匀放大／缩小，没有扭曲，物理单元与参考单元形状较为一致，失真度小
# 反之 r_mesh[cell_id] 越大，说明物理单元的形状与参考单元的形状差异越大，失真度越大
cell_id = 1
area_of_ref_elem = sum(ref.H)
x = [coord[1] for coord in grid.xyz_q[cell_id]]
t = [coord[2] for coord in grid.xyz_q[cell_id]]
xt = [x, t]
G = zeros(2,2)
for a in 1:2, b in 1:2
  # 累加第 b 方向导数 * 物理坐标分量 a
  G[a,b] = mean(ref.D[a] * xt[b])
end
δ = norm(G) / det(G)^0.5
norm(G) / (grid.geometric_terms.J_q[cell_id][1])^0.5
r_msh1 = area_of_ref_elem * δ^2

# 但是参考单元也有失真度，所以如果不把每个物理单元的失真度减去参考单元的失真度，solver会倾向于把
# 每个物理单元全都扭会参考单元的形状
# 所以这里我们再计算一下参考单元的失真度
r = ref.rst_q[1, :]
s = ref.rst_q[2, :]
rs = [r, s]
G_ref = zeros(2,2)
for a in 1:2, b in 1:2
  # 累加第 b 方向导数 * 参考坐标分量 a
  G_ref[a,b] = mean(ref.D[a] * rs[b])
end
δ_ref = norm(G_ref) / det(G_ref)^0.5
r_msh_ref = area_of_ref_elem * δ_ref^2

# 这部分代码介绍了如何计算某个单元的 基于解与单元平均值偏差”的追踪指标
# 这在zahr早期的文章中有提到并大量使用，但是在2020年之后的文章中变成了enriched residual,
# 这个enriched residual 对于SBP来说可能稍微难以实现一点，我们先用 这个追踪指标来代替
# 追踪指标应该是, 对于element K
# int_K || num_u - num_̄u ||^2 dV where num_̄u = 1/area(K) * int_K num_u dV
# 代码来说， num_̄u = sum(quadrature_K .* num_u) / (area(K))
# quadrature_K = grid.geometric_terms.J_q[cell_id] .* diag(ref.H)
# area(K) = area_of_ref_elem * J_q[cell_id][1]
# then f_shk_K = int_K || num_u - num_̄u ||^2 dV 
# 但权重并不平均，对于element上每个点，都要单独减去 num_̄u 然后得到新的 vector，再平方后计算这个vector在当前
# 物理单元上的数值积分

# 一个 shock附近的element，发现 f_shk 的值较大，虽然0.018 没那么大， 但是
cell_id = 12
u12 = oscillating_u[cell_id, :]
quadrature12 = grid.geometric_terms.J_q[cell_id] .* diag(ref.H)
area_12 = area_of_ref_elem * grid.geometric_terms.J_q[cell_id][1]
num_̄u12 = sum(quadrature12 .* u12) / area_12
f_shk = sum((u12 .- num_̄u12).^2 .* quadrature12)

# 另一个 shock附近的element，发现 f_shk 的值较小，基本为0， 所以这个方法应该也算可行，后续可能也许要给这部分的residual加个系数
cell_id = 4
u4 = oscillating_u[cell_id, :]
quadrature4 = grid.geometric_terms.J_q[cell_id] .* diag(ref.H)
area_4 = area_of_ref_elem * grid.geometric_terms.J_q[cell_id][1]
num_̄u4 = sum(quadrature4 .* u4) / area_4
f_shk4 = sum((u4 .- num_̄u4).^2 .* quadrature4)







@save joinpath(@__DIR__, "space_time_buffer.jld2") buffer

length(buffer.indices_moving_coords)  # 检查最终解的长度
for i in 1:156
    print(buffer.final_solution[12:end][i], "\n")
end

reshape(buffer.final_solution[12:end], (13, 12))  # 检查最终解的形状
plot_mesh(grid)


#----------------------------------------------


dual1 = ForwardDiff.Dual(1.5, (1.0, 0.0, 0.0))  # 值为1.5，对第1个变量的导数为1.0
dual2 = ForwardDiff.Dual(2.3, (0.0, 1.0, 0.0))  # 值为2.3，对第2个变量的导数为1.0  
dual3 = ForwardDiff.Dual(-0.8, (0.0, 0.0, 1.0)) # 值为-0.8，对第3个变量的导数为1.0

println("dual1: ", dual1)
println("dual2: ", dual2) 
println("dual3: ", dual3)

# 创建包含3个Dual数的数组
dual_array = [dual1, dual2, dual3]
println("Dual数组: ", dual_array)
println("数组类型: ", typeof(dual_array))
println("元素类型: ", eltype(dual_array))

println("dual1的值: ", ForwardDiff.value(dual1))
println("dual2的值: ", ForwardDiff.value(dual2))
println("dual3的值: ", ForwardDiff.value(dual3))

values = [ForwardDiff.value(x) for x in dual_array]

# ——--------------------------------------------------------

buffer_debug = SpaceTimeBuffer(grid, [ax, at], oscillating_u)


ref = buffer_debug.ref
area_of_ref_elem = sum(ref.H)
G_ref = I(2) # ∂ ξ / ∂ ξ = I_2
δ_ref = norm(G_ref) / det(G_ref)^0.5
r_msh_ref = area_of_ref_elem * δ_ref^2

# PDE 残差作为等式约束
function pde_residual_debug(buffer_debug)
    u_vec = buffer_debug.initial_guess[buffer_debug.index_mesh+1:end]  # 使用最终解作为输入
    num_of_element = length(grid.VOL) # 因为可能要用自己写的 SimpleGrid， 这里先用 VOL 来算 element/node 数量
    num_of_nodes = size(grid.VOL[1][1])[1]

    xyz_gmsh = buffer_debug.initial_guess[1:buffer_debug.index_mesh]
    xyz_gmsh_coords = [Coord{2, Float64}((xyz_gmsh[i], xyz_gmsh[buffer_debug.index_mesh ÷ 2 + i])) for i in 1:buffer_debug.index_mesh÷2]
    current_grid = construct_simpleGrid(
        grid.cells, buffer_debug.ref, xyz_gmsh_coords, 
        grid.face_interfaces, grid.face_sets, buffer_debug.indices_moving_coords
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
        r_mesh[cell_id] = r_msh_K - r_msh_ref  # 减去参考单元的失真度
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
mesh_shock_objective_debug(buffer_debug; αmesh = 0.01, αshk = 10000)


plot_u_interactive(grid, oscillating_u)