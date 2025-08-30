
include("parameters.jl")

para = grid, newGrid, IC
U0 =  deepcopy(analytic_U) .- 0.1 * rand(size(analytic_U)...)



using SparseConnectivityTracer, ADTypes
using NonlinearSolve, NLsolve
# using SciMLBase          # NonlinearAlias 类型在这里
# using LineSearches       # Hager–Zhang 线搜索

include("src/nonlinear_solver_contribution.jl")


function RHS_for_solver(du, U, p; WL = [3, 2, 10], WR = [3, -1, 5])
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
    grid, newGrid, IC = p
    W = evaluate_W_forSolver(U)
    One = ones(length(grid.xyz[1]) * 3)

    for cell_id in 1:n_cells(grid)
        VOL_contribution!(du, newGrid, W, cell_id, One)
    end


    for interface in newGrid.interior_interfaces
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

    for interface in union(newGrid.interfaces_aligning_shock, newGrid.interfaces_aligning_contact_wave)
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

    bottom_faces = collect(union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2")))
    for i in 1:length(bottom_faces)
        cell_id, face_id = bottom_faces[i]
        ref = grid.cells[cell_id].ref_data[]
        Rγ = Matrix(ref.R[face_id])
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Ntγ = normal[2, :]

        Ētγ = kron(Rγ' * ref.H_face * Diagonal(Ntγ) * Rγ, I(3))

        Ukk = two_point_state_flux_function_forSolver(W[cell_id, :, :], W[cell_id, :, :])
        U0 = vec(IC[(cell_id,face_id)]')
        

        # du[cell_id, :] += newGrid.H̄inv[cell_id, :, :] * ( Ētγ .* Ukk * One - Ētγ * U0 )
        # du[cell_id, :] += Ētγ .* Ukk * One - Ētγ * U0
        temp_du = Ētγ .* Ukk * One - Ētγ * U0
        du[cell_id, :, 1] += temp_du[1:3:end]
        du[cell_id, :, 2] += temp_du[2:3:end]
        du[cell_id, :, 3] += temp_du[3:3:end]
    end
    
    for faceindex in collect(get_face_set(grid, "LEFT_INFLOW"))
        cell_id, face_id = faceindex
        ref = grid.cells[cell_id].ref_data[]
        Rγ = Matrix(ref.R[face_id])
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Nxγ = normal[1, :]
    
        Ēxγ = kron(Rγ' * ref.H_face * Diagonal(Nxγ) * Rγ, I(3))
    
        Wk = W[cell_id, :, :]
        Fxkk = two_point_flux_function_forSolver(Wk, Wk)
        temp_du = Ēxγ .* Fxkk * One - Ēxγ * FBCL
        du[cell_id, :, 1] += temp_du[1:3:end]
        du[cell_id, :, 2] += temp_du[2:3:end]
        du[cell_id, :, 3] += temp_du[3:3:end]

    end

    for faceindex in collect(get_face_set(grid, "RIGHT_INFLOW"))
        cell_id, face_id = faceindex
        ref = grid.cells[cell_id].ref_data[]
        Rγ = Matrix(ref.R[face_id])
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Nxγ = normal[1, :]

        Ēxγ = kron(Rγ' * ref.H_face * Diagonal(Nxγ) * Rγ, I(3))

        ukBC = (kron(Rγ, I(3)) * vec(W[cell_id, :, :]'))[2:3:end]

        Wk = W[cell_id, :, :]
        WBC = zeros(eltype(ukBC), length(ukBC), 3)
        WBC[:, 1] .= WR[1]
        WBC[:, 2] .= ukBC
        WBC[:, 3] .= WR[3]

        Fxkk = two_point_flux_function_forSolver(Wk, Wk)
        temp_du = Ēxγ .* Fxkk * One - kron(Rγ' * ref.H_face * Diagonal(Nxγ), I(3)) * evaluate_Fx(WBC)
        du[cell_id, :, 1] += temp_du[1:3:end]
        du[cell_id, :, 2] += temp_du[2:3:end]
        du[cell_id, :, 3] += temp_du[3:3:end]
    end

    return nothing
end

para = grid, newGrid, IC



# 初值 = 真解 + 微扰
U0_for_solver = deepcopy(analytic_U) .- 0.5*rand(size(analytic_U)...)


using NonlinearSolve, LineSearches, ADTypes, SparseConnectivityTracer, SparseDiffTools, NLsolve

# —— 初始猜测 —— #
u0_vec = vec(U0_for_solver)
du = similar(U0_for_solver); fill!(du, 0.0)

# ρ_bar = 3.814806918969338
# ρu_bar= 2.5733237805370788
# E_bar   = 31.105708483930357
# scale_vec = repeat([ρ_bar ,ρu_bar, E_bar], n_cells(grid)*length(grid.xyz_q[1]))   # 和 u0_vec 同长度
# u_scaled  = u0_vec ./ scale_vec


# --- 构造 in-place 残差 ---
# 用 NLsolve 来解

function f_nl!(F, u)
    U = reshape(u, size(U0_for_solver))  # 将 1D 向量重塑为 3D 数组
    du = similar(U); fill!(du, 0.0)
    RHS_for_solver(du, U, para)          # 🔥 直接调用 RHS_for_solver
    F[:] = vec(du)
end

res = nlsolve(
  f_nl!, u0_vec;
  method    = :trust_region,
  linsolve  = :dogleg,      # :cg
  autodiff  = :central,   
  xtol      = 1e-8,
  ftol      = 1e-8,
  autoscale = true,
  show_trace= true,
  iterations = 200,
)

@save joinpath(@__DIR__, "res_ICOSAHOM.jld2") res
@load joinpath(@__DIR__, "res_ICOSAHOM.jld2") res




include("src/plot_helper.jl")

reshaped_res = reshape(res.zero, size(U0_for_solver))

reshaped_W = zeros(size(U0_for_solver))
reshaped_W[:, :, 1] = deepcopy(reshaped_res[:,:,1])
reshaped_W[:, :, 2] = deepcopy(reshaped_res[:,:,2]) ./ deepcopy(reshaped_res[:,:,1])
reshaped_W[:, :, 3] = (deepcopy(reshaped_res[:,:,3]) .- 0.5 .* deepcopy(reshaped_res[:,:,1]) .* reshaped_W[:, :, 2].^2) * 0.4

plot_interactive_solution_by_W(grid, reshaped_W, variable = 2)

plot_density = plot_one_variable_by_W(grid, reshaped_W, variable = 1, size = (2200, 400), show_cell_id = false)
savefig(plot_density, joinpath(@__DIR__, "density_ICOSAHOM.png"))

plot_velocity = plot_one_variable_by_W(grid, reshaped_W, variable = 2, size = (2200, 400), show_cell_id = false)
savefig(plot_velocity, joinpath(@__DIR__, "velocity_ICOSAHOM.png"))

plot_pressure = plot_one_variable_by_W(grid, reshaped_W, variable = 3, size = (2200, 400), show_cell_id = false)
savefig(plot_pressure, joinpath(@__DIR__, "pressure_ICOSAHOM.png"))





plot_all_variables_by_W(grid, reshaped_W, size = (1200, 400))

plot_density = plot_one_variable_by_W(grid, reshaped_W, variable = 1, size = (2200, 400), show_cell_id = false)
savefig(plot_density, joinpath(@__DIR__, "density_ICOSAHOM.png"))

plot_velocity = plot_one_variable_by_W(grid, reshaped_W, variable = 2, size = (2200, 400), show_cell_id = false)
savefig(plot_velocity, joinpath(@__DIR__, "velocity_ICOSAHOM.png"))

plot_pressure = plot_one_variable_by_W(grid, reshaped_W, variable = 3, size = (2200, 400), show_cell_id = false)
savefig(plot_pressure, joinpath(@__DIR__, "pressure_ICOSAHOM.png"))





plot_all_variables_by_W(grid, reshaped_W, size = (1200, 400))


savefig(joinpath(@__DIR__, "density_ICOSAHOM.png"), plot_density)
plot_one_variable_by_U(grid, reshaped_res, variable = 1,  size = (1200, 400))


plot_at_final = plot_solution_at_final_time_by_U(grid, reshaped_res, variable = 1)
plot_interactive_solution_by_U(grid, reshaped_res, variable = 2)


plot_interactive_solution_by_W(grid, U0_for_solver, variable = 1)
W_initial = zeros(size(U0_for_solver))
W_initial[:, :, 1] = deepcopy(U0_for_solver[:,:,1])
W_initial[:, :, 2] = deepcopy(U0_for_solver[:,:,2]) ./ deepcopy(U0_for_solver[:,:,1])
W_initial[:, :, 3] = (deepcopy(U0_for_solver[:,:,3]) .- 0.5 .* deepcopy(U0_for_solver[:,:,1]) .* U0_for_solver[:, :, 2].^2) * 0.4


plot_interactive_solution_by_W(grid, reshaped_W, variable = 3)
p11 = plot_solution_at_final_time_by_U(grid, reshaped_res, variable = 1, threshold = 0.2)
gr()
savefig(p11, joinpath(@__DIR__, "density_ICOSAHOM_final.png"))
savefig(plot_solution_at_final_time_by_U(grid, reshaped_res, variable = 2, threshold = 0.2), joinpath(@__DIR__, "velocity_ICOSAHOM_final.png"))
savefig(plot_solution_at_final_time_by_U(grid, reshaped_res, variable = 3, threshold = 0.2), joinpath(@__DIR__, "pressure_ICOSAHOM_final.png"))

# ---------------------------------------------
# load 之前算过的解并画图









