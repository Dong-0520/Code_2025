print("sss")

include(joinpath(@__DIR__, "local_parameter.jl"))
include(joinpath(@__DIR__, "../src/parameter.jl"))
include(joinpath(@__DIR__, "../../src_simpleGrid/SimpleGrid_Euler.jl"))
include("../src/nonlinear_solver_contribution.jl")
include("../src/dissipation.jl")
include(joinpath(@__DIR__, "../src/plot_helper_grid.jl"))
include(joinpath(@__DIR__, "../src/plot_helper_simple_grid.jl"))

# solve the problem on the nonaligned grid by pseudo transient with dissipation term
using NonlinearSolve

function RHS_for_solver(du, U, p; WL = [3, 2, 10], WR = [3, -1, 5], vol_diss_factor = 10, num_of_variables = 3, interface_diss_factor = 1.0)
    """
    U is a three dimensional array
    each U[k, :, :] is a 2D array containing the values of the variables at each nodes of node on k element
    say there are m variables and N nodes per element
                 ρ1, ρu1, E1
                 ρ2, ρu2, E2
                 .   .    .
    U[k, :, :] = .   .    .
                 .   .    .
                 ρN, ρuN, EN
    """
    newGrid, smooth_interior_interfaces, interfaces_aligning_shock, interfaces_aligning_contact_wave, 
                                bottom_faceIndex, boundary_faceIndex, IC, FBC = p
    W = evaluate_W_forSolver(U)
    V = evaluate_entropy_variables_forSolver(W) # entropy variables
    One = ones(size(newGrid.ref.rst,2) * num_of_variables)

    for cell_id in 1:n_cells(grid)
        # VOL_contribution!(du, newGrid, W, cell_id, One)
        # print(vol_diss_factor, "\n")
        VOL_contribution_with_dissipation_forSolver!(du, newGrid, W, V, cell_id, One, vol_diss_factor = vol_diss_factor)
    end


    for interface in smooth_interior_interfaces
        ck, _ = interface.face_1
        cv, _ = interface.face_2

        Fxkk = two_point_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
        Fxkv = two_point_flux_function_forSolver(W[ck, :, :], W[cv, :, :])
        Ukk = two_point_state_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
        Ukv = two_point_state_flux_function_forSolver(W[ck, :, :], W[cv, :, :])

        # for cell v
        Fxvv = two_point_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
        Fxvk = two_point_flux_function_forSolver(W[cv, :, :], W[ck, :, :])
        Uvv = two_point_state_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
        Uvk = two_point_state_flux_function_forSolver(W[cv, :, :], W[ck, :, :])
    
        Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
        Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv
        
        Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
        Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk

        duk_smooth, duv_smooth = smooth_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxkv, Fxvv, Fxvk, Ukk, Ukv, Uvv, Uvk, One)

        du[ck, :, 1] += duk_smooth[1:3:end]
        du[ck, :, 2] += duk_smooth[2:3:end]
        du[ck, :, 3] += duk_smooth[3:3:end]
        du[cv, :, 1] += duv_smooth[1:3:end]
        du[cv, :, 2] += duv_smooth[2:3:end]
        du[cv, :, 3] += duv_smooth[3:3:end]
    end

    for interface in union(interfaces_aligning_shock, interfaces_aligning_contact_wave)
        ck, _ = interface.face_1
        cv, _ = interface.face_2
    
        # for cell k
        Fxkk = two_point_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
        Ukk = two_point_state_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
        uk = vec(U[ck,:,:]')
        Fk = evaluate_Fx(W[ck, :, :])

        # for cell v
        Fxvv = two_point_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
        Uvv = two_point_state_flux_function_forSolver(W[cv, :, :], W[cv, :, :])

        uv = vec(U[cv,:,:]')
        Fv = evaluate_Fx(W[cv, :, :])
    
        Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
        Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv
        
        Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
        Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk

        duk_shock, duv_shock = shock_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxvv, Ukk, Uvv, uk, uv, Fk, Fv, One)

        du[ck, :, 1] += duk_shock[1:3:end]
        du[ck, :, 2] += duk_shock[2:3:end]
        du[ck, :, 3] += duk_shock[3:3:end]
        du[cv, :, 1] += duv_shock[1:3:end]
        du[cv, :, 2] += duv_shock[2:3:end]
        du[cv, :, 3] += duv_shock[3:3:end]
    end

    for faceindex in bottom_faceIndex
        cell_id, _ = faceindex

        Ētγ = newGrid.Ēγ_for_bottom[faceindex]

        Ukk = two_point_state_flux_function_forSolver(W[cell_id, :, :], W[cell_id, :, :])
        # U0 = vec(IC[faceindex]')
        
        temp_du = Ētγ .* Ukk * One - Ētγ * IC[faceindex]
        du[cell_id, :, 1] += temp_du[1:3:end]
        du[cell_id, :, 2] += temp_du[2:3:end]
        du[cell_id, :, 3] += temp_du[3:3:end]
    end
    
    for faceindex in boundary_faceIndex
        cell_id, _ = faceindex

        Ēxγ = newGrid.Ēγ_for_boundary[faceindex]

        Fxkk = two_point_flux_function_forSolver(W[cell_id, :, :], W[cell_id, :, :])
        temp_du = Ēxγ .* Fxkk * One - Ēxγ * FBC[faceindex]
        du[cell_id, :, 1] += temp_du[1:3:end]
        du[cell_id, :, 2] += temp_du[2:3:end]
        du[cell_id, :, 3] += temp_du[3:3:end]

    end

    return nothing
end



buffer = SpaceTimeBuffer(grid, [S1, S2, S3], analytic_U)


num_of_xyz_gmsh = buffer.index_mesh ÷ 2
new_xyz_gmsh_coords = [Coord{2, Float64}((buffer.initial_guess[1:buffer.index_mesh][i], buffer.initial_guess[1:buffer.index_mesh][num_of_xyz_gmsh + i])) for i in 1:num_of_xyz_gmsh]

newGrid = construct_simpleGrid(
    grid.cells, buffer.ref, new_xyz_gmsh_coords, grid.face_interfaces, 
    grid.face_sets, buffer.indices_moving_coords, 
    buffer.bottom_faceIndex,
    buffer.boundary_faceIndex
)


# para = (current_grid, buffer.smooth_interior_interfaces, buffer.interfaces_aligning_shock, buffer.interfaces_aligning_contact_wave, 
#                                 buffer.bottom_faceIndex, buffer.boundary_faceIndex, IC, FBC)


# du = similar(analytic_U); fill!(du, 0.0)

# RHS_for_solver(du, analytic_U, para)

function space_time_RK4(U, pseudo_dt, p; vol_diss_factor = 0.1, interface_diss_factor = 1.0)
    c1, c2, c3, c4 = 0.0, 0.40128709, 0.56449983, 0.87678807
    b1, b2, b3, b4 = 0.20334721, 0.19932974, 0.28339585, 0.31392720
    a21 = 0.40128709
    a31, a32 = 0.28224991, 0.28224991
    a41, a42, a43 = 0.25972925, 0.25479937, 0.36225945

    # 🔥 适配新的RHS函数
    function compute_RHS(U_input)
        du = similar(U_input)
        fill!(du, 0.0)
        RHS_for_solver(du, U_input, p, 
                      vol_diss_factor=vol_diss_factor, 
                      interface_diss_factor=interface_diss_factor)
        return -du  # 🔥 注意：伪瞬态是 dU/dt = -R(U)，所以取负号
    end

    k1 = compute_RHS(U)
    k2 = compute_RHS(U + a21 * pseudo_dt * k1)
    k3 = compute_RHS(U + a31 * pseudo_dt * k1 + a32 * pseudo_dt * k2)
    k4 = compute_RHS(U + a41 * pseudo_dt * k1 + a42 * pseudo_dt * k2 + a43 * pseudo_dt * k3)

    result = U + pseudo_dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4)

    # return positive_filtering(result)
    return result
end

function space_time_solve(U0::Array{Float64, 3}, p; 
                         pseudo_dt = 1.0, 
                         num_of_pseudo_time_step = 1000, 
                         num_of_variable = 3,  
                         vol_diss_factor = 10.0, 
                         interface_diss_factor = 1.0,
                         tolerance = 1e-2,
                         adaptive_dt = true,
                         print_interval = 10)

    num_of_cells = size(U0, 1)
    num_of_nodes = size(U0, 2)
    
    # 🔥 初始化存储
    all_U = Vector{Array{Float64, 3}}()
    diffs = Vector{Float64}()
    residual_norms = Vector{Float64}()
    
    push!(all_U, deepcopy(U0))
    
    println("🚀 开始伪瞬态求解...")
    println("初始伪时间步长: $pseudo_dt")
    println("目标容差: $tolerance")

    for pseudo_step in 1:num_of_pseudo_time_step
        curr_U = all_U[pseudo_step]
        
        # 🔥 计算当前步的残差范数
        du_residual = similar(curr_U)
        fill!(du_residual, 0.0)
        RHS_for_solver(du_residual, curr_U, p, 
                      vol_diss_factor=vol_diss_factor, 
                      interface_diss_factor=interface_diss_factor)
        residual_norm = norm(du_residual)
        push!(residual_norms, residual_norm)
        
        # 🔥 RK4步进
        next_U = space_time_RK4(curr_U, pseudo_dt, p, 
                               vol_diss_factor=vol_diss_factor, 
                               interface_diss_factor=interface_diss_factor)
        
        # 🔥 计算变化量
        max_diff = maximum(abs.(next_U - curr_U))
        push!(diffs, max_diff)
        push!(all_U, next_U)

        # 🔥 自适应时间步长
        if adaptive_dt && pseudo_step > 5
            recent_diffs = diffs[max(1, end-4):end]
            if length(recent_diffs) > 1
                diff_trend = (recent_diffs[end] - recent_diffs[1]) / length(recent_diffs)
                
                if diff_trend < -1e-8 && max_diff < 1e-6  # 收敛良好
                    pseudo_dt = min(pseudo_dt * 1.2, 1e-2)
                elseif diff_trend > 1e-8 || max_diff > 10  # 收敛困难
                    pseudo_dt = max(pseudo_dt * 0.7, 1e-6)
                end
            end
        end

        # 🔥 输出进度
        if pseudo_step % print_interval == 0
            println("步数 $pseudo_step: 残差范数 = $(round(residual_norm, digits=8)), 最大变化 = $(round(max_diff, digits=8)), dt = $(round(pseudo_dt, digits=6))")
        end

        # 🔥 检查发散
        if max_diff > 100 || isnan(max_diff) || any(isnan, next_U)
            println("❌ 不稳定! 步数: $pseudo_step, pseudo_dt = $pseudo_dt")
            return all_U, diffs, residual_norms, false
        end
        
        # 🔥 检查收敛 (使用残差范数更准确)
        if residual_norm < tolerance && max_diff < tolerance * 10
            println("✅ 收敛成功! 残差范数: $(round(residual_norm, digits=12))")
            println("总步数: $pseudo_step, 最终dt: $(round(pseudo_dt, digits=6))")
            return all_U, diffs, residual_norms, true
        end
    end
    
    println("⚠️  未收敛，达到最大步数")
    return all_U, diffs, residual_norms, false
end


# 🔥 设置参数 (适配你的RHS_for_solver)
p = (newGrid, buffer.smooth_interior_interfaces, buffer.interfaces_aligning_shock, buffer.interfaces_aligning_contact_wave, 
                                buffer.bottom_faceIndex, buffer.boundary_faceIndex, IC, FBC)


analytic_U[[10, 11], :, 1] .= ρstarR
analytic_U[[12], :, 1] .= ρstarL
analytic_U[[10,11,12], :, 2] .= ustar
analytic_U[[10,11,12], :, 3] .= pstar

# 🔥 调用伪瞬态求解
all_U, diffs, residual_norms, converged = space_time_solve(
    analytic_U,  # 初始解
    p,
    pseudo_dt = 1e-2,
    num_of_pseudo_time_step = 5000,
    vol_diss_factor = 20.0,
    interface_diss_factor = 1.0,
    tolerance = 1e-8,
    adaptive_dt = true,
    print_interval = 50
)

final_U = all_U[end]

# plot_one_variable_by_U(grid, final_U)
plot_interactive_solution_by_U(grid, final_U)



using SparseConnectivityTracer, ADTypes, NonlinearSolve, NLsolve

Pkg.activate("MadNLP_env")

using JuMP, MadNLP, Random, MadNLPHSL

function RHS_for_solution(du, U, para; WL = [3, 2, 10], WR = [3, -1, 5])
    """
    U is a three dimensional array
    each U[k, :, :] is a 2D array containing the values of the variables at each nodes of node on k element
    say there are m variables and N nodes per element
                 ρ1, ρu1, E1
                 ρ2, ρu2, E2
                 .   .    .
    U[k, :, :] = .   .    .
                 .   .    .
                 ρN, ρuN, EN
    """
    newGrid, smooth_interior_interfaces, interfaces_aligning_shock, interfaces_aligning_contact_wave, bottom_faceIndex, boundary_faceIndex, IC, FBC = para
    W = evaluate_W_forSolver(U)
    One = ones(length(grid.xyz[1]) * 3)

    for cell_id in 1:n_cells(grid)
        VOL_contribution!(du, newGrid, W, cell_id, One)
    end


    for interface in smooth_interior_interfaces
        ck, _ = interface.face_1
        cv, _ = interface.face_2

        Fxkk = two_point_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
        Fxkv = two_point_flux_function_forSolver(W[ck, :, :], W[cv, :, :])
        Ukk = two_point_state_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
        Ukv = two_point_state_flux_function_forSolver(W[ck, :, :], W[cv, :, :])

        # for cell v
        Fxvv = two_point_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
        Fxvk = two_point_flux_function_forSolver(W[cv, :, :], W[ck, :, :])
        Uvv = two_point_state_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
        Uvk = two_point_state_flux_function_forSolver(W[cv, :, :], W[ck, :, :])
    
        Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
        Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv
        
        Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
        Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk

        duk_smooth, duv_smooth = smooth_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxkv, Fxvv, Fxvk, Ukk, Ukv, Uvv, Uvk, One)

        du[ck, :, 1] += duk_smooth[1:3:end]
        du[ck, :, 2] += duk_smooth[2:3:end]
        du[ck, :, 3] += duk_smooth[3:3:end]
        du[cv, :, 1] += duv_smooth[1:3:end]
        du[cv, :, 2] += duv_smooth[2:3:end]
        du[cv, :, 3] += duv_smooth[3:3:end]
    end

    for interface in union(interfaces_aligning_shock, interfaces_aligning_contact_wave)
        ck, _ = interface.face_1
        cv, _ = interface.face_2
    
        # for cell k
        Fxkk = two_point_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
        Ukk = two_point_state_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
        uk = vec(U[ck,:,:]')
        Fk = evaluate_Fx(W[ck, :, :])

        # for cell v
        Fxvv = two_point_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
        Uvv = two_point_state_flux_function_forSolver(W[cv, :, :], W[cv, :, :])

        uv = vec(U[cv,:,:]')
        Fv = evaluate_Fx(W[cv, :, :])
    
        Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
        Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv
        
        Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
        Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk

        duk_shock, duv_shock = shock_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxvv, Ukk, Uvv, uk, uv, Fk, Fv, One)

        du[ck, :, 1] += duk_shock[1:3:end]
        du[ck, :, 2] += duk_shock[2:3:end]
        du[ck, :, 3] += duk_shock[3:3:end]
        du[cv, :, 1] += duv_shock[1:3:end]
        du[cv, :, 2] += duv_shock[2:3:end]
        du[cv, :, 3] += duv_shock[3:3:end]
    end

    for faceindex in bottom_faceIndex
        cell_id, _ = faceindex

        Ētγ = newGrid.Ēγ_for_bottom[faceindex]

        Ukk = two_point_state_flux_function_forSolver(W[cell_id, :, :], W[cell_id, :, :])
        # U0 = vec(IC[faceindex]')
        
        temp_du = Ētγ .* Ukk * One - Ētγ * IC[faceindex]
        du[cell_id, :, 1] += temp_du[1:3:end]
        du[cell_id, :, 2] += temp_du[2:3:end]
        du[cell_id, :, 3] += temp_du[3:3:end]
    end
    
    for faceindex in boundary_faceIndex
        cell_id, _ = faceindex

        Ēxγ = newGrid.Ēγ_for_boundary[faceindex]

        Fxkk = two_point_flux_function_forSolver(W[cell_id, :, :], W[cell_id, :, :])
        temp_du = Ēxγ .* Fxkk * One - Ēxγ * FBC[faceindex]
        du[cell_id, :, 1] += temp_du[1:3:end]
        du[cell_id, :, 2] += temp_du[2:3:end]
        du[cell_id, :, 3] += temp_du[3:3:end]

    end

    return nothing
end


function wrap_up_results(buffer; num_of_xyz_gmsh = length(grid.xyz_gmsh))
    initial_guess = buffer.initial_guess
    initial_guess[buffer.indices_moving_coords] = buffer.final_solution[1:length(buffer.indices_moving_coords)]

    new_xyz_gmsh = initial_guess[1:buffer.index_mesh]
    new_xyz_gmsh_coords = [Coord{2, Float64}((new_xyz_gmsh[i], new_xyz_gmsh[buffer.index_mesh ÷ 2 + i])) for i in 1:num_of_xyz_gmsh]
    final_simple_grid = construct_simpleGrid(
        grid.cells, buffer.ref, new_xyz_gmsh_coords, 
        grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords,            
        buffer.bottom_faceIndex,
        buffer.boundary_faceIndex
    )

    final_u = reshape(buffer.final_solution[length(buffer.indices_moving_coords)+1:end], (length(grid.VOL), size(grid.VOL[1][1])[1], 3))

    return final_simple_grid, final_u

end

# t0, tF, xL, xR = find_boundaries(grid)
# indices_moving_coords, coords_xyz_gmsh, t0, tF, xL, xR = get_meshinfo_for_buffer(grid)

function pde_model_with_solver_struct(buffer::SpaceTimeBuffer; lin_solver = MadNLPHSL.Ma57Solver, 
                                                                αmesh = 1.0, αshk = 1.0, max_iter = 100, num_of_variables = 3)
    # model = Model(() -> MadNLP.Optimizer(
    #     linear_solver=lin_solver,
    #     print_level=MadNLP.INFO,
    #     max_iter=max_iter
    # ))  
    model = Model(() -> MadNLP.Optimizer(
        linear_solver=lin_solver,
        print_level=MadNLP.INFO,
        max_iter=max_iter,                       # 减少最大迭代次数
        
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
    #         grid.cells, buffer.ref, new_xyz_gmsh_coords, grid.face_interfaces, 
    #         grid.face_sets, buffer.indices_moving_coords, 
    #         buffer.bottom_faceIndex,
    #         buffer.boundary_faceIndex
    #     )
            
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
            grid.cells, buffer.ref, new_xyz_gmsh_coords, grid.face_interfaces, 
            grid.face_sets, buffer.indices_moving_coords, 
            buffer.bottom_faceIndex,
            buffer.boundary_faceIndex
        )

        Umatrix = reshape(u_vec, (num_of_element, num_of_nodes, num_of_variables))
        dU = similar(Umatrix, T) 
        fill!(dU, zero(T))  # 初始化 dU
        para = (current_grid, buffer.smooth_interior_interfaces, buffer.interfaces_aligning_shock, buffer.interfaces_aligning_contact_wave, 
                                buffer.bottom_faceIndex, buffer.boundary_faceIndex, IC, FBC)
        RHS_for_solution(dU, Umatrix, para)  # 计算残差
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
            grid.cells, buffer.ref, new_xyz_gmsh_coords, grid.face_interfaces, 
            grid.face_sets, buffer.indices_moving_coords, 
            buffer.bottom_faceIndex,
            buffer.boundary_faceIndex
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
            
            if abs(det_GK) < 1e-12 || det_GK < 0
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
        Umatrix = reshape(u_vec, (num_of_element, num_of_nodes, num_of_variables))
        for cell_id in 1:num_of_element
            num_ρk = Umatrix[cell_id, :, 1]
            num_ρuk = Umatrix[cell_id, :, 2]
            num_Ek = Umatrix[cell_id, :, 3]
            quadrature_k = current_grid.geometric_terms.J_q[cell_id] .* diag(ref.H)
            area_K = area_of_ref_elem * current_grid.geometric_terms.J_q[cell_id][1]
            # num_̄uk = sum(quadrature_k .* num_uk) / area_K
            num_̄ρk  = sum(quadrature_k .* num_ρk) / area_K
            num_̄ρ̄uk = sum(quadrature_k .* num_ρuk) / area_K
            num_̄Ek = sum(quadrature_k .* num_Ek) / area_K

            ρ_shk = sum((num_ρk .- num_̄ρk).^2 .* quadrature_k) / area_K
            ρu_shk = sum((num_ρuk .- num_̄ρ̄uk).^2 .* quadrature_k) / area_K
            E_shk = sum((num_Ek .- num_̄Ek).^2 .* quadrature_k) / area_K

            r_shk[cell_id] = ρ_shk + ρu_shk + E_shk  # 计算 shock tracking residual
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
# current_grid = nothing
buffer = SpaceTimeBuffer(grid, [S1, S2, S3], final_U)
buffer.indices_moving_coords = [5, 7]
buffer.indices_moving_coords
model1 = pde_model_with_solver_struct(buffer, αmesh = 0, αshk = 10, max_iter = 400, lin_solver = MadNLPHSL.Ma57Solver)

@save joinpath(@__DIR__, "Euler_optimization_result_order1.jld2") buffer final_U



using Plots

final_simple_grid, final_u = wrap_up_results(buffer, num_of_xyz_gmsh = length(grid.xyz_gmsh))
final_W = evaluate_W_forSolver(final_u)
plot_u_interactive(final_simple_grid, final_W[:, :, 1])                                                                                                                                                                                                                              
plot_mesh(final_simple_grid)

gr()
plot_one_variable_by_W(final_simple_grid, final_W[:, :, 1])


plot_mesh(grid)

@save joinpath(@__DIR__, "Euler_optimization_result.jld2") buffer final_simple_grid final_u


@load joinpath(@__DIR__, "Euler_optimization_result.jld2") buffer final_simple_grid final_u
# include(joinpath(@__DIR__, "plot_helper.jl"))
# buffer.final_objective
# buffer.final_solution


plot_u_interactive(grid, oscillating_u)

plot_mesh(grid)
# -------------------------------------------------------------------
# plot for presentation here
function analytic_sol(x,variable; t = 0.0)
    if t < (1/S1) * x
        return (ρL, uL, pL)[variable]
    end
    # if all(yk .> (1/S1) .* xk .- 1e-5) && all(yk .> (1/S2) .* xk .- 1e-5)
    if t > (1/S1) * x && t > (1/S2) * x                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        return (ρstarL, ustar, pstar)[variable]
    end
    # if all(yk .< (1/S2) .* xk .+ 1e-5) && all(yk .> (1/S3) .* xk .- 1e-5)
    if t < (1/S2) * x && t > (1/S3) * x
        # analytic_W[cell_id, :,1] .= ρstarR
        # analytic_W[cell_id, :,2] .= ustar
        # analytic_W[cell_id, :,3] .= pstar
        return (ρstarR, ustar, pstar)[variable]
    end
    # if all(yk .< (1/S3) .* xk .+ 1e-5)
    if t < (1/S3) * x
        # analytic_W[cell_id, :,1] .= ρR
        # analytic_W[cell_id, :,2] .= uR
        # analytic_W[cell_id, :,3] .= pR
        return (ρR, uR, pR)[variable]
    end

end

BOTTOM_INFLOW = collect(union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2")))
x_coords_initial_before_opt = Dict()
for (cell_id, face_id) in BOTTOM_INFLOW
    x_coords_initial_before_opt[cell_id] = map(x -> x[1], ref.R[face_id] * grid.xyz_q[cell_id])
end

# elements having final time are 9, 13, 10, 8
# cooresponding face indices are 1, 1, 1, 1
TOP_INFLOW = [FaceIndex((1, 3)), FaceIndex((6, 2)), FaceIndex((5, 2)), FaceIndex((3, 1))]
x_coords_final_before_opt = Dict()
for (cell_id, face_id) in TOP_INFLOW
    x_coords_final_before_opt[cell_id] = map(x -> x[1], ref.R[face_id] * grid.xyz_q[cell_id])
end
x_coords_final_before_opt
@load joinpath(@__DIR__, "Euler_optimization_result.jld2") buffer final_simple_grid final_u

gr()

oscillating_W = analytic_W
oscillating_ρ = oscillating_W[:, :, 1]
oscillating_u = oscillating_W[:, :, 2]
oscillating_p = oscillating_W[:, :, 3]
final_solution_comparision_before_opt = plot(xlims = (-0.251, 0.251), grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    plot!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], ref.R[face_id] * oscillating_ρ[cell_id, :], 
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], ref.R[face_id] * oscillating_ρ[cell_id, :], 
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], Vector(analytic_sol.(x_coords_final_before_opt[cell_id], 1, t = 0.075)),
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], Vector(analytic_sol.(x_coords_final_before_opt[cell_id], 1, t = 0.075)),
           label = "", color = :red, markersize = 4)
    first_red = false
end
title!(final_solution_comparision_before_opt, "Numerical vs Analytic at t = 0.075 \n before optimization")
xlabel!(final_solution_comparision_before_opt, "x")
ylabel!(final_solution_comparision_before_opt, "ρ")
# savefig(final_solution_comparision_before_opt, joinpath(@__DIR__, "../plots_const/linadv_final_sol_compare_before_opt_const_problem.png"))
savefig(final_solution_comparision_before_opt, joinpath(@__DIR__, "plots_Euler/Euler_final_sol_compare_before_opt_density.png"))

final_solution_comparision_before_opt = plot(xlims = (-0.251, 0.251), grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    plot!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], ref.R[face_id] * oscillating_u[cell_id, :], 
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], ref.R[face_id] * oscillating_u[cell_id, :], 
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], Vector(analytic_sol.(x_coords_final_before_opt[cell_id], 2, t = 0.075)),
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], Vector(analytic_sol.(x_coords_final_before_opt[cell_id], 2, t = 0.075)),
           label = "", color = :red, markersize = 4)
    first_red = false
end
title!(final_solution_comparision_before_opt, "Numerical vs Analytic at t = 0.075 \n before optimization")
xlabel!(final_solution_comparision_before_opt, "x")
ylabel!(final_solution_comparision_before_opt, "u")
savefig(final_solution_comparision_before_opt, joinpath(@__DIR__, "plots_Euler/Euler_final_sol_compare_before_opt_velocity.png"))

final_solution_comparision_before_opt = plot(xlims = (-0.251, 0.251), grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    plot!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], ref.R[face_id] * oscillating_p[cell_id, :], 
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], ref.R[face_id] * oscillating_p[cell_id, :], 
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], Vector(analytic_sol.(x_coords_final_before_opt[cell_id], 3, t = 0.075)),
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(final_solution_comparision_before_opt, x_coords_final_before_opt[cell_id], Vector(analytic_sol.(x_coords_final_before_opt[cell_id], 3, t = 0.075)),
           label = "", color = :red, markersize = 4)
    first_red = false
end
title!(final_solution_comparision_before_opt, "Numerical vs Analytic at t = 0.075 \n before optimization")
xlabel!(final_solution_comparision_before_opt, "x")
ylabel!(final_solution_comparision_before_opt, "p")
savefig(final_solution_comparision_before_opt, joinpath(@__DIR__, "plots_Euler/Euler_final_sol_compare_before_opt_pressure.png"))




initial_solution_comparision_before_opt = plot(xlims = (-1.01, 1.01), grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in BOTTOM_INFLOW
    plot!(initial_solution_comparision_before_opt, x_coords_initial_before_opt[cell_id], ref.R[face_id] * oscillating_u[cell_id, :], 
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(initial_solution_comparision_before_opt, x_coords_initial_before_opt[cell_id], ref.R[face_id] * oscillating_u[cell_id, :], 
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(initial_solution_comparision_before_opt, x_coords_initial_before_opt[cell_id], analytic_sol.(x_coords_initial_before_opt[cell_id]),
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(initial_solution_comparision_before_opt, x_coords_initial_before_opt[cell_id], analytic_sol.(x_coords_initial_before_opt[cell_id]),
           label = "", color = :red, markersize = 4)
    first_red = false
end 
title!(initial_solution_comparision_before_opt, "Numerical vs Analytic at t = 0 \n before optimization")
xlabel!(initial_solution_comparision_before_opt, "x")
ylabel!(initial_solution_comparision_before_opt, "u")
savefig(initial_solution_comparision_before_opt, joinpath(@__DIR__, "../plots_const/linadv_initial_sol_compare_before_opt_const_problem.png"))


@load joinpath(@__DIR__, "..//optimization_results//buffer_two_coordinates.jld2") buffer final_simple_grid final_u
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

gr()
# initial_solution_comparision_after_opt = plot(xlims = (-1.01, 1.01), grid = false)
# first_blue = true
# first_red = true
# for (cell_id, face_id) in BOTTOM_INFLOW
#     plot!(initial_solution_comparision_after_opt, x_coords_initial_after_opt[cell_id], ref.R[face_id] * final_u[cell_id, :],
#           label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
#     scatter!(initial_solution_comparision_after_opt, x_coords_initial_after_opt[cell_id], ref.R[face_id] * final_u[cell_id, :],
#            label = "", color = :blue, markersize = 4)
#     first_blue = false

#     plot!(initial_solution_comparision_after_opt, x_coords_initial_after_opt[cell_id], analytic_sol.(x_coords_initial_after_opt[cell_id], x2 = 0.0),
#           label = first_red ? "analytic" : "", color = :red, linewidth = 2)
#     scatter!(initial_solution_comparision_after_opt, x_coords_initial_after_opt[cell_id], analytic_sol.(x_coords_initial_after_opt[cell_id], x2 = 0.0),
#            label = "", color = :red, markersize = 4)
#     first_red = false
# end
# title!(initial_solution_comparision_after_opt, "Numerical vs Analytic at t = 0 \n after optimization")
# xlabel!(initial_solution_comparision_after_opt, "x")
# ylabel!(initial_solution_comparision_after_opt, "u")
# savefig(initial_solution_comparision_after_opt, joinpath(@__DIR__, "../plots_const/linadv_initial_sol_compare_after_opt_const_problem.png"))

final_solution_comparision_after_opt = plot(xlims = (-0.251, 0.251), grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    plot!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], ref.R[face_id] * final_W[cell_id, :, 1],
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], ref.R[face_id] * final_W[cell_id, :, 1],
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], analytic_sol.(x_coords_final_after_opt[cell_id], 1, t = 0.075),
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], analytic_sol.(x_coords_final_after_opt[cell_id], 1, t = 0.075),
           label = "", color = :red, markersize = 4)
    first_red = false
end
title!(final_solution_comparision_after_opt, "Numerical vs Analytic at t = 0.0.075 \n after optimization")
xlabel!(final_solution_comparision_after_opt, "x")
ylabel!(final_solution_comparision_after_opt, "ρ")
savefig(final_solution_comparision_after_opt, joinpath(@__DIR__, "plots_Euler/Euler_final_sol_compare_after_opt_density.png"))

final_solution_comparision_after_opt = plot(xlims = (-0.251, 0.251), grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    plot!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], ref.R[face_id] * final_W[cell_id, :, 2],
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], ref.R[face_id] * final_W[cell_id, :, 2],
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], analytic_sol.(x_coords_final_after_opt[cell_id], 2, t = 0.075),
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], analytic_sol.(x_coords_final_after_opt[cell_id], 2, t = 0.075),
           label = "", color = :red, markersize = 4)
    first_red = false
end
title!(final_solution_comparision_after_opt, "Numerical vs Analytic at t = 0.0.075 \n after optimization")
xlabel!(final_solution_comparision_after_opt, "x")
ylabel!(final_solution_comparision_after_opt, "u")
savefig(final_solution_comparision_after_opt, joinpath(@__DIR__, "plots_Euler/Euler_final_sol_compare_after_opt_velocity.png"))

final_solution_comparision_after_opt = plot(xlims = (-0.251, 0.251), grid = false)
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    plot!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], ref.R[face_id] * final_W[cell_id, :, 3],
          label = first_blue ? "numerical" : "", color = :blue, linewidth = 2)
    scatter!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], ref.R[face_id] * final_W[cell_id, :, 3],
           label = "", color = :blue, markersize = 4)
    first_blue = false

    plot!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], analytic_sol.(x_coords_final_after_opt[cell_id], 3, t = 0.075),
          label = first_red ? "analytic" : "", color = :red, linewidth = 2)
    scatter!(final_solution_comparision_after_opt, x_coords_final_after_opt[cell_id], analytic_sol.(x_coords_final_after_opt[cell_id], 3, t = 0.075),
           label = "", color = :red, markersize = 4)
    first_red = false
end
title!(final_solution_comparision_after_opt, "Numerical vs Analytic at t = 0.0.075 \n after optimization")
xlabel!(final_solution_comparision_after_opt, "x")
ylabel!(final_solution_comparision_after_opt, "p")
savefig(final_solution_comparision_after_opt, joinpath(@__DIR__, "plots_Euler/Euler_final_sol_compare_after_opt_pressure.png"))



using Plots.Measures
gr()
num_of_cells = length(grid.VOL)
num_of_nodes = size(grid.VOL[1][1], 1)
plot_size = (1200, 400)
xlimit = (-0.251, 0.251)
ylimit = (-0.01, 0.08)
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
          [0.0, S1 * 0.075], [0.0, 0.075], 
          label = "", color = :red, markersize = 5)  # 添加蓝色标签的占位符
plot!(mesh_comparision, [0.0, S1 * 0.075], [0.0, 0.075], label = "True Shock", color = :red, markersize = 5)
scatter!(mesh_comparision, 
          [0.0, S3 * 0.075], [0.0, 0.075], 
          label = "", color = :red, markersize = 5)  # 添加蓝色标签的占位符
plot!(mesh_comparision, [0.0, S3 * 0.075], [0.0, 0.075], label = "", color = :red, markersize = 5)
mesh_comparision
savefig(mesh_comparision, joinpath(@__DIR__, "plots_Euler/Euler_mesh_comparision.png"))

mesh_nonaligned = scatter(xlabel = "x", ylabel = "t", title = "Mesh comparison (before optimization)", 
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
          [0.0, S1 * 0.075], [0.0, 0.075], 
          label = "", color = :red, markersize = 5)  # 添加蓝色标签的占位符
plot!(mesh_nonaligned, [0.0, S1 * 0.075], [0.0, 0.075], label = "True Shock", color = :red, markersize = 5)
scatter!(mesh_nonaligned, 
          [0.0, S3 * 0.075], [0.0, 0.075], 
          label = "", color = :red, markersize = 5)  # 添加蓝色标签的占位符
plot!(mesh_nonaligned, [0.0, S3 * 0.075], [0.0, 0.075], label = "", color = :red, markersize = 5)
savefig(mesh_nonaligned, joinpath(@__DIR__, "plots_Euler/Euler_mesh_comparison_before_opt.png")) 

mesh_after_opt = scatter(xlabel = "x", ylabel = "t", title = "Mesh comparison (after optimization)", 
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
          [0.0, S1 * 0.075], [0.0, 0.075], 
          label = "", color = :red, markersize = 5)  # 添加蓝色标签的占位符
plot!(mesh_after_opt, [0.0, S1 * 0.075], [0.0, 0.075], label = "True Shock", color = :red, markersize = 5)
scatter!(mesh_after_opt, 
          [0.0, S3 * 0.075], [0.0, 0.075], 
          label = "", color = :red, markersize = 5)  # 添加蓝色标签的占位符
plot!(mesh_after_opt, [0.0, S3 * 0.075], [0.0, 0.075], label = "", color = :red, markersize = 5)
mesh_after_opt
savefig(mesh_after_opt, joinpath(@__DIR__, "plots_Euler/Euler_mesh_comparison_after_opt.png"))

riemann_problem_solution_before_opt = plot_one_variable_by_W(grid, analytic_W[:, :, 1])
savefig(riemann_problem_solution_before_opt, joinpath(@__DIR__, "plots_Euler/Euler_riemann_problem_solution_before_opt_density.png"))

riemann_problem_solution_after_opt = plot_one_variable_by_W(final_simple_grid, final_W[:, :, 1])
savefig(riemann_problem_solution_after_opt, joinpath(@__DIR__, "plots_Euler/Euler_riemann_problem_solution_after_opt_density.png"))

riemann_problem_solution_before_opt = plot_one_variable_by_W(grid, analytic_W[:, :, 2])
savefig(riemann_problem_solution_before_opt, joinpath(@__DIR__, "plots_Euler/Euler_riemann_problem_solution_before_opt_velocity.png"))

riemann_problem_solution_after_opt = plot_one_variable_by_W(final_simple_grid, final_W[:, :, 2])
savefig(riemann_problem_solution_after_opt, joinpath(@__DIR__, "plots_Euler/Euler_riemann_problem_solution_after_opt_velocity.png"))

riemann_problem_solution_before_opt = plot_one_variable_by_W(grid, analytic_W[:, :, 3])
savefig(riemann_problem_solution_before_opt, joinpath(@__DIR__, "plots_Euler/Euler_riemann_problem_solution_before_opt_pressure.png")) 

riemann_problem_solution_after_opt = plot_one_variable_by_W(final_simple_grid, final_W[:, :, 3])
savefig(riemann_problem_solution_after_opt, joinpath(@__DIR__, "plots_Euler/Euler_riemann_problem_solution_after_opt_pressure.png"))

gr()
error_comparison_final = scatter(grid = false, xlabel = "x", ylabel = "Error", title = "Error Comparison at t = 0.2")
first_blue = true
first_red = true
for (cell_id, face_id) in TOP_INFLOW
    scatter!(error_comparison_final, x_coords_final_before_opt[cell_id], abs.(analytic_sol.(x_coords_final_before_opt[cell_id]) - ref.R[face_id] * oscillating_u[cell_id, :]), color = :blue, label = first_blue ? "before" : "")
    scatter!(error_comparison_final, x_coords_final_after_opt[cell_id], abs.(analytic_sol.(x_coords_final_after_opt[cell_id]) - ref.R[face_id] * final_u[cell_id, :]), color = :red, label = first_red ? "after" : "")
    first_blue = false
    first_red = false
end
error_comparison_final
savefig(error_comparison_final, joinpath(@__DIR__, "../plots_const/linadv_error_comparison_const_problem.png"))

error_comparison_initial = scatter(grid = false, xlabel = "x", ylabel = "Error", title = "Error Comparison at t = 0.0")
first_blue = true
first_red = true
for (cell_id, face_id) in BOTTOM_INFLOW
    scatter!(error_comparison_initial, x_coords_initial_before_opt[cell_id], abs.(analytic_sol.(x_coords_initial_before_opt[cell_id], x2 = 0.0) - ref.R[face_id] * oscillating_u[cell_id, :]), color = :blue, label = first_blue ? "before" : "")
    scatter!(error_comparison_initial, x_coords_initial_after_opt[cell_id], abs.(analytic_sol.(x_coords_initial_after_opt[cell_id], x2 = 0.0) - ref.R[face_id] * final_u[cell_id, :]), color = :red, label = first_red ? "after" : "")
    first_blue = false
    first_red = false
end
error_comparison_initial
savefig(error_comparison_initial, joinpath(@__DIR__, "../plots_const/linadv_error_comparison_initial_const_problem.png"))
