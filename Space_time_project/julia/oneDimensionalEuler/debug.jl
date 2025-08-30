# ------------------------------------------------------------------
# some tests


S1 * (ρL - ρstarL) - (ρL * uL - ρstarL * ustar)
S1 * (ρL * uL - ρstarL * ustar) - ((pL + ρL * uL^2) - (pstar + ρstarL * ustar^2))

S1 * (EL - EstarL) - ((EL + pL)*uL - (EstarL + pstar)*ustar)

S3 * (ρR - ρstarR) - (ρR * uR - ρstarR * ustar)
S3 * (ρR * uR - ρstarR * ustar) - ((pR + ρR * uR^2) - (pstar + ρstarR * ustar^2))

S3 * (ER - EstarR) - ((ER + pR)*uR - (EstarR + pstar)*ustar)

# check n ∘ FL = n ∘ FR
cell_id, face_id = 95, 2
ref = grid.cells[cell_id].ref_data[]
Rγ = Matrix(ref.R[face_id])
mask = @views ref.f_mask[face_id]
normal = grid.geometric_terms.N_f[cell_id][:, mask]
Nxγ = normal[1, :]
Ntγ = normal[2, :]

# three wave, FaceIndex((37, 1))
Nx = Nxγ[1]
Nt = Ntγ[1]
(Nx * ρR * uR + Nt * ρR) - (Nx * ρstarR * ustar + Nt * ρstarR)
Nx * (ρR * uR^2 + pR) + Nt * (ρR * uR) - (Nx * (ρstarR * ustar^2 + pstar) + Nt * ρstarR * ustar)
Nx * uR * (ER + pR) + Nt * ER - (Nx * ustar * (EstarR + pstar) + Nt * EstarR)

# ------------------------ Debug ------------------------------
U = analytic_U
num_of_nodes = length(grid.xyz[1])
du = zeros(n_cells(grid), num_of_nodes * 3)
W = evaluate_W(U)
V = evaluate_entropy_variables(W) # entropy variables
One = ones(length(grid.xyz[1]) * 3)
interface_diss_factor = 1.0
Threads.@threads for cell_id in 1:n_cells(grid)
    # VOL_contribution!(du, newGrid, W, cell_id, One)
    # print(vol_diss_factor, "\n")
    VOL_contribution_with_dissipation!(du, newGrid, W, V, cell_id, One, vol_diss_factor = 0)
end

Threads.@threads for interface in newGrid.interior_interfaces
    ck, _ = interface.face_1
    cv, _ = interface.face_2
    Pk, Pv = interface.P1, interface.P2
    Pk_W, Pv_W = permutation_for_W(Pk), permutation_for_W(Pv)

    Rγk, Rγv = newGrid.Rγ[interface.face_1], newGrid.Rγ[interface.face_2]
    R̄γk, R̄γv = newGrid.R̄γ[interface.face_1], newGrid.R̄γ[interface.face_2]
    H̄γk, H̄γv = newGrid.H̄γ[interface.face_1], newGrid.H̄γ[interface.face_2]
    N̄xγk, N̄xγv = newGrid.N̄xγ[interface.face_1], newGrid.N̄xγ[interface.face_2]

    ρk_face, uk_face, pk_face = Rγk * W[ck, :, 1], Rγk * W[ck, :, 2], Rγk * W[ck, :, 3]
    U1k_face, U2k_face, U3k_face = Rγk * U[ck, :, 1], Rγk * U[ck, :, 2], Rγk * U[ck, :, 3]

    ρv_face, uv_face, pv_face = Rγv * W[cv, :, 1], Rγv * W[cv, :, 2], Rγv * W[cv, :, 3]
    U1v_face, U2v_face, U3v_face = Rγv * U[cv, :, 1], Rγv * U[cv, :, 2], Rγv * U[cv, :, 3]

    # for cell k
    Fxkk = two_point_flux_function(W[ck, :, :], W[ck, :, :])
    Fxkv = two_point_flux_function(W[ck, :, :], W[cv, :, :])
    Ukk = two_point_state_flux_function(W[ck, :, :], W[ck, :, :])
    Ukv = two_point_state_flux_function(W[ck, :, :], W[cv, :, :])
    vk = V[ck, :]

    # for cell v
    Fxvv = two_point_flux_function(W[cv, :, :], W[cv, :, :])
    Fxvk = two_point_flux_function(W[cv, :, :], W[ck, :, :])
    Uvv = two_point_state_flux_function(W[cv, :, :], W[cv, :, :])
    Uvk = two_point_state_flux_function(W[cv, :, :], W[ck, :, :])
    vv = V[cv, :]

    Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
    Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv
    
    Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
    Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk

    duk_smooth, duv_smooth = smooth_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxkv, Fxvv, Fxvk, Ukk, Ukv, Uvv, Uvk, One)
    Λk, Λv = Λ_matrix(U1k_face, U2k_face, U3k_face, ρk_face, uk_face, pk_face, Pk, U1v_face, U2v_face, U3v_face, ρv_face, uv_face, pv_face, Pv)
    duk_dissipation, duv_dissipation = dissipation_term(R̄γk, R̄γv, H̄γk, H̄γv, N̄xγk, N̄xγv, Λk, Λv, vk, vv, Pk_W, Pv_W)
    
    # du[ck, :] += duk_smooth - interface_diss_factor * duk_dissipation
    # du[cv, :] += duv_smooth - interface_diss_factor * duv_dissipation

    du[ck, :] += duk_smooth
    du[cv, :] += duv_smooth
end

Threads.@threads for interface in newGrid.interfaces_aligning_shock
    ck, _ = interface.face_1
    cv, _ = interface.face_2
    Pk, Pv = interface.P1, interface.P2
    Pk_W, Pv_W = permutation_for_W(Pk), permutation_for_W(Pv)

    Rγk, Rγv = newGrid.Rγ[interface.face_1], newGrid.Rγ[interface.face_2]
    R̄γk, R̄γv = newGrid.R̄γ[interface.face_1], newGrid.R̄γ[interface.face_2]
    H̄γk, H̄γv = newGrid.H̄γ[interface.face_1], newGrid.H̄γ[interface.face_2]
    N̄xγk, N̄xγv = newGrid.N̄xγ[interface.face_1], newGrid.N̄xγ[interface.face_2]

    ρk_face, uk_face, pk_face = Rγk * W[ck, :, 1], Rγk * W[ck, :, 2], Rγk * W[ck, :, 3]
    U1k_face, U2k_face, U3k_face = Rγk * U[ck, :, 1], Rγk * U[ck, :, 2], Rγk * U[ck, :, 3]

    ρv_face, uv_face, pv_face = Rγv * W[cv, :, 1], Rγv * W[cv, :, 2], Rγv * W[cv, :, 3]
    U1v_face, U2v_face, U3v_face = Rγv * U[cv, :, 1], Rγv * U[cv, :, 2], Rγv * U[cv, :, 3]

    # for cell k
    Fxkk = two_point_flux_function(W[ck, :, :], W[ck, :, :])
    Fxkv = two_point_flux_function(W[ck, :, :], W[cv, :, :])
    Ukk = two_point_state_flux_function(W[ck, :, :], W[ck, :, :])
    Ukv = two_point_state_flux_function(W[ck, :, :], W[cv, :, :])
    uk = vec(U[ck,:,:]')
    # wk = vec(W[ck,:,:]')
    vk = V[ck, :]
    Fk = evaluate_Fx(W[ck, :, :])

    # for cell v
    Fxvv = two_point_flux_function(W[cv, :, :], W[cv, :, :])
    Fxvk = two_point_flux_function(W[cv, :, :], W[ck, :, :])
    Uvv = two_point_state_flux_function(W[cv, :, :], W[cv, :, :])
    Uvk = two_point_state_flux_function(W[cv, :, :], W[ck, :, :])

    uv = vec(U[cv,:,:]')
    # wv = vec(W[cv,:,:]')
    vv = V[cv, :]
    Fv = evaluate_Fx(W[cv, :, :])

    Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
    Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv
    
    Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
    Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk


    Λk, Λv = Λ_matrix(U1k_face, U2k_face, U3k_face, ρk_face, uk_face, pk_face, Pk, U1v_face, U2v_face, U3v_face, ρv_face, uv_face, pv_face, Pv)
    indicator = Rankine_Hugoniot_condition_indicator(U1k_face, U2k_face, U3k_face, ρk_face, uk_face, pk_face, Pk, U1v_face, U2v_face, U3v_face, ρv_face, uv_face, pv_face, Pv, newGrid.Nxγ[interface.face_1], newGrid.Ntγ[interface.face_1])
    duk_smooth, duv_smooth = smooth_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxkv, Fxvv, Fxvk, Ukk, Ukv, Uvv, Uvk, One)
    duk_shock, duv_shock = shock_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxvv, Ukk, Uvv, uk, uv, Fk, Fv, One)
    duk_dissipation, duv_dissipation = dissipation_term(R̄γk, R̄γv, H̄γk, H̄γv, N̄xγk, N̄xγv, Λk, Λv, vk, vv, Pk_W, Pv_W)
    print("Indicator: ", indicator, "\n")
    du[ck, :] += indicator * duk_shock + (1 - indicator) * (duk_smooth - interface_diss_factor * duk_dissipation)
    du[cv, :] += indicator * duv_shock + (1 - indicator) * (duv_smooth - interface_diss_factor * duv_dissipation)


end

bottom_faces = collect(union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2")))
Threads.@threads for i in 1:length(bottom_faces)
    cell_id, face_id = bottom_faces[i]
    ref = grid.cells[cell_id].ref_data[]
    Rγ = Matrix(ref.R[face_id])
    mask = @views ref.f_mask[face_id]
    normal = grid.geometric_terms.N_f[cell_id][:, mask]
    Ntγ = normal[2, :]

    Ētγ = kron(Rγ' * ref.H_face * Diagonal(Ntγ) * Rγ, I(3))

    Ukk = two_point_state_flux_function(W[cell_id, :, :], W[cell_id, :, :])
    U0 = vec(IC[(cell_id,face_id)]')
    

    # du[cell_id, :] += newGrid.H̄inv[cell_id, :, :] * ( Ētγ .* Ukk * One - Ētγ * U0 )
    du[cell_id, :] += Ētγ .* Ukk * One - Ētγ * U0
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
    # WBC = zeros(Float64, size(Wk))
    # WBC[:, 1] .= WL[1]
    # WBC[:, 2] .= WL[2]
    # WBC[:, 3] .= WL[3]
    Fxkk = two_point_flux_function(Wk, Wk)
    # FxkBC = two_point_flux_function(Wk, WBC)

    # du[cell_id, :] += newGrid.H̄inv[cell_id, :, :] * (Ēxγ .* Fxkk - Ēxγ .* FxkBC) * One
    # du[cell_id, :] += (Ēxγ .* Fxkk - Ēxγ .* FxkBC) * One
    du[cell_id, :] += Ēxγ .* Fxkk * One - Ēxγ * FBCL
end
WR = [3, -1, 5]
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
    WBC = zeros(Float64, length(ukBC), 3)
    WBC[:, 1] .= WR[1]
    WBC[:, 2] .= ukBC
    WBC[:, 3] .= WR[3]

    Fxkk = two_point_flux_function(Wk, Wk)
    # FxkBC = two_point_flux_function(Wk, WBC)

    # du[cell_id, :] += newGrid.H̄inv[cell_id, :, :] * (Ēxγ .* Fxkk - Ēxγ .* FxkBC) * One
    # du[cell_id, :] += (Ēxγ .* Fxkk - Ēxγ .* FxkBC) * One


    # du[cell_id, :] += Ēxγ .* Fxkk * One - Ēxγ * evaluate_Fx(WBC)
    du[cell_id, :] += Ēxγ .* Fxkk * One - kron(Rγ' * ref.H_face * Diagonal(Nxγ), I(3)) * evaluate_Fx(WBC)


end

filtered_du = positive_filtering(du, n_cells(grid); num_of_variables = 3)
result = zeros(size(U))
Threads.@threads for cell_id in 1:n_cells(grid)
    result[cell_id, :, :] = reshape(filtered_du[cell_id,:], (3, num_of_nodes))'
end






interface = newGrid.interfaces_aligning_shock[1]
ck, lfγk = interface.face_1
cv, lfγv = interface.face_2
Pk, Pv = interface.P1, interface.P2

Rγk, Rγv = newGrid.Rγ[interface.face_1], newGrid.Rγ[interface.face_2]
R̄γk, R̄γv = newGrid.R̄γ[interface.face_1], newGrid.R̄γ[interface.face_2]
H̄γk, H̄γv = newGrid.H̄γ[interface.face_1], newGrid.H̄γ[interface.face_2]
N̄xγk, N̄xγv = newGrid.N̄xγ[interface.face_1], newGrid.N̄xγ[interface.face_2]

ρk_face, uk_face, pk_face = Rγk * W[ck, :, 1], Rγk * W[ck, :, 2], Rγk * W[ck, :, 3]
U1k_face, U2k_face, U3k_face = Rγk * U[ck, :, 1], Rγk * U[ck, :, 2], Rγk * U[ck, :, 3]
ρv_face, uv_face, pv_face = Rγv * W[cv, :, 1], Rγv * W[cv, :, 2], Rγv * W[cv, :, 3]
U1v_face, U2v_face, U3v_face = Rγv * U[cv, :, 1], Rγv * U[cv, :, 2], Rγv * U[cv, :, 3]

F2k_face = ρk_face .* uk_face.^2 + pk_face
F2v_face = ρv_face .* uv_face.^2 + pv_face
F3k_face = uk_face .* (pk_face + U3k_face)
F3v_face = uv_face .* (pv_face + U3v_face)

mask = @views ref.f_mask[lfγk]
normal = grid.geometric_terms.N_f[ck][:, mask]
Nxγk = normal[1, :]
Ntγk = normal[2, :]


diag(newGrid.Nxγ[interface.face_1])
Nxγk




all(isapprox.(Ntγ .* ργk + Nxγ .* ρuγk - Ntγ .* ργv - Nxγ .* ρuγv, 0, atol=1e-8))
all(isapprox.(Ntγ .* ρuγk + Nxγ .* ργk - Ntγ .* ρuγv - Nxγ .* ργv, 0, atol=1e-8))



U = analytic_U
du = similar(U0_for_solver)    
fill!(du, 0.0)                                
grid, newGrid, IC, _, _ = para
W = evaluate_W_forSolver(U)
V = evaluate_entropy_variables_forSolver(W) # entropy variables
One = ones(length(grid.xyz[1]) * 3)


for cell_id in 1:n_cells(grid)
    # VOL_contribution!(du, newGrid, W, cell_id, One)
    # print(vol_diss_factor, "\n")
    VOL_contribution_with_dissipation_forSolver!(du, newGrid, W, V, cell_id, One, vol_diss_factor = vol_diss_factor)
end


for interface in newGrid.interior_interfaces
    ck, _ = interface.face_1
    cv, _ = interface.face_2
    Pk, Pv = interface.P1, interface.P2
    Pk_W, Pv_W = permutation_for_W(Pk), permutation_for_W(Pv)

    Rγk, Rγv = newGrid.Rγ[interface.face_1], newGrid.Rγ[interface.face_2]
    R̄γk, R̄γv = newGrid.R̄γ[interface.face_1], newGrid.R̄γ[interface.face_2]
    H̄γk, H̄γv = newGrid.H̄γ[interface.face_1], newGrid.H̄γ[interface.face_2]
    N̄xγk, N̄xγv = newGrid.N̄xγ[interface.face_1], newGrid.N̄xγ[interface.face_2]

    ρk_face, uk_face, pk_face = Rγk * W[ck, :, 1], Rγk * W[ck, :, 2], Rγk * W[ck, :, 3]
    U1k_face, U2k_face, U3k_face = Rγk * U[ck, :, 1], Rγk * U[ck, :, 2], Rγk * U[ck, :, 3]

    ρv_face, uv_face, pv_face = Rγv * W[cv, :, 1], Rγv * W[cv, :, 2], Rγv * W[cv, :, 3]
    U1v_face, U2v_face, U3v_face = Rγv * U[cv, :, 1], Rγv * U[cv, :, 2], Rγv * U[cv, :, 3]

    # for cell k
    Fxkk = two_point_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
    Fxkv = two_point_flux_function_forSolver(W[ck, :, :], W[cv, :, :])
    Ukk = two_point_state_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
    Ukv = two_point_state_flux_function_forSolver(W[ck, :, :], W[cv, :, :])
    vk = V[ck, :]

    # for cell v
    Fxvv = two_point_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
    Fxvk = two_point_flux_function_forSolver(W[cv, :, :], W[ck, :, :])
    Uvv = two_point_state_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
    Uvk = two_point_state_flux_function_forSolver(W[cv, :, :], W[ck, :, :])
    # vv = V[cv, :]

    Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
    Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv
    
    Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
    Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk

    duk_smooth, duv_smooth = smooth_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxkv, Fxvv, Fxvk, Ukk, Ukv, Uvv, Uvk, One)

    # du[ck, :] += duk_smooth
    # du[cv, :] += duv_smooth
    du[ck, :, 1] += duk_smooth[1:3:end]
    du[ck, :, 2] += duk_smooth[2:3:end]
    du[ck, :, 3] += duk_smooth[3:3:end]
    du[cv, :, 1] += duv_smooth[1:3:end]
    du[cv, :, 2] += duv_smooth[2:3:end]
    du[cv, :, 3] += duv_smooth[3:3:end]
end

for interface in newGrid.interfaces_aligning_shock
    ck, _ = interface.face_1
    cv, _ = interface.face_2
    Pk, Pv = interface.P1, interface.P2
    # Pk_W, Pv_W = permutation_for_W(Pk), permutation_for_W(Pv)

    Rγk, Rγv = newGrid.Rγ[interface.face_1], newGrid.Rγ[interface.face_2]
    R̄γk, R̄γv = newGrid.R̄γ[interface.face_1], newGrid.R̄γ[interface.face_2]
    H̄γk, H̄γv = newGrid.H̄γ[interface.face_1], newGrid.H̄γ[interface.face_2]
    N̄xγk, N̄xγv = newGrid.N̄xγ[interface.face_1], newGrid.N̄xγ[interface.face_2]

    ρk_face, uk_face, pk_face = Rγk * W[ck, :, 1], Rγk * W[ck, :, 2], Rγk * W[ck, :, 3]
    U1k_face, U2k_face, U3k_face = Rγk * U[ck, :, 1], Rγk * U[ck, :, 2], Rγk * U[ck, :, 3]

    ρv_face, uv_face, pv_face = Rγv * W[cv, :, 1], Rγv * W[cv, :, 2], Rγv * W[cv, :, 3]
    U1v_face, U2v_face, U3v_face = Rγv * U[cv, :, 1], Rγv * U[cv, :, 2], Rγv * U[cv, :, 3]

    # for cell k
    Fxkk = two_point_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
    # Fxkv = two_point_flux_function(W[ck, :, :], W[cv, :, :])
    Ukk = two_point_state_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
    # Ukv = two_point_state_flux_function(W[ck, :, :], W[cv, :, :])
    uk = vec(U[ck,:,:]')
    # wk = vec(W[ck,:,:]')
    # vk = V[ck, :]
    Fk = evaluate_Fx(W[ck, :, :])

    # for cell v
    Fxvv = two_point_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
    # Fxvk = two_point_flux_function(W[cv, :, :], W[ck, :, :])
    Uvv = two_point_state_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
    # Uvk = two_point_state_flux_function(W[cv, :, :], W[ck, :, :])

    uv = vec(U[cv,:,:]')
    # wv = vec(W[cv,:,:]')
    # vv = V[cv, :]
    Fv = evaluate_Fx(W[cv, :, :])

    Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
    Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv
    
    Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
    Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk


    # Λk, Λv = Λ_matrix(U1k_face, U2k_face, U3k_face, ρk_face, uk_face, pk_face, Pk, U1v_face, U2v_face, U3v_face, ρv_face, uv_face, pv_face, Pv)
    # indicator = Rankine_Hugoniot_condition_indicator(U1k_face, U2k_face, U3k_face, ρk_face, uk_face, pk_face, Pk, U1v_face, U2v_face, U3v_face, ρv_face, uv_face, pv_face, Pv, newGrid.Nxγ[interface.face_1], newGrid.Ntγ[interface.face_1])
    # duk_smooth, duv_smooth = smooth_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxkv, Fxvv, Fxvk, Ukk, Ukv, Uvv, Uvk, One)
    duk_shock, duv_shock = shock_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxvv, Ukk, Uvv, uk, uv, Fk, Fv, One)
    # duk_dissipation, duv_dissipation = dissipation_term(R̄γk, R̄γv, H̄γk, H̄γv, N̄xγk, N̄xγv, Λk, Λv, vk, vv, Pk_W, Pv_W)
    # print("Indicator: ", indicator, "\n")
    # du[ck, :] += duk_shock
    # du[cv, :] += duv_shock

    du[ck, :, 1] += duk_shock[1:3:end]
    du[ck, :, 2] += duk_shock[2:3:end]
    du[ck, :, 3] += duk_shock[3:3:end]
    du[cv, :, 1] += duv_shock[1:3:end]
    du[cv, :, 2] += duv_shock[2:3:end]
    du[cv, :, 3] += duv_shock[3:3:end]
end




# -------------------------- Debug ------------------------------
du = similar(U0_for_solver)
fill!(du, 0.0)
grid, newGrid, IC, _, _ = para
W = evaluate_W_forSolver(U)
V = evaluate_entropy_variables_forSolver(W) # entropy variables
One = ones(length(grid.xyz[1]) * 3)

for cell_id in 1:n_cells(grid)
    # VOL_contribution!(du, newGrid, W, cell_id, One)
    # print(vol_diss_factor, "\n")
    VOL_contribution_with_dissipation_forSolver!(du, newGrid, W, V, cell_id, One, vol_diss_factor = 0.0)
end


for interface in newGrid.interior_interfaces
    ck, _ = interface.face_1
    cv, _ = interface.face_2
    Pk, Pv = interface.P1, interface.P2
    Pk_W, Pv_W = permutation_for_W(Pk), permutation_for_W(Pv)

    Rγk, Rγv = newGrid.Rγ[interface.face_1], newGrid.Rγ[interface.face_2]
    R̄γk, R̄γv = newGrid.R̄γ[interface.face_1], newGrid.R̄γ[interface.face_2]
    H̄γk, H̄γv = newGrid.H̄γ[interface.face_1], newGrid.H̄γ[interface.face_2]
    N̄xγk, N̄xγv = newGrid.N̄xγ[interface.face_1], newGrid.N̄xγ[interface.face_2]

    ρk_face, uk_face, pk_face = Rγk * W[ck, :, 1], Rγk * W[ck, :, 2], Rγk * W[ck, :, 3]
    U1k_face, U2k_face, U3k_face = Rγk * U[ck, :, 1], Rγk * U[ck, :, 2], Rγk * U[ck, :, 3]

    ρv_face, uv_face, pv_face = Rγv * W[cv, :, 1], Rγv * W[cv, :, 2], Rγv * W[cv, :, 3]
    U1v_face, U2v_face, U3v_face = Rγv * U[cv, :, 1], Rγv * U[cv, :, 2], Rγv * U[cv, :, 3]

    # for cell k
    Fxkk = two_point_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
    Fxkv = two_point_flux_function_forSolver(W[ck, :, :], W[cv, :, :])
    Ukk = two_point_state_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
    Ukv = two_point_state_flux_function_forSolver(W[ck, :, :], W[cv, :, :])
    vk = V[ck, :]

    # for cell v
    Fxvv = two_point_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
    Fxvk = two_point_flux_function_forSolver(W[cv, :, :], W[ck, :, :])
    Uvv = two_point_state_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
    Uvk = two_point_state_flux_function_forSolver(W[cv, :, :], W[ck, :, :])
    # vv = V[cv, :]

    Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
    Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv
    
    Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
    Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk

    duk_smooth, duv_smooth = smooth_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxkv, Fxvv, Fxvk, Ukk, Ukv, Uvv, Uvk, One)

    # du[ck, :] += duk_smooth
    # du[cv, :] += duv_smooth
    du[ck, :, 1] += duk_smooth[1:3:end]
    du[ck, :, 2] += duk_smooth[2:3:end]
    du[ck, :, 3] += duk_smooth[3:3:end]
    du[cv, :, 1] += duv_smooth[1:3:end]
    du[cv, :, 2] += duv_smooth[2:3:end]
    du[cv, :, 3] += duv_smooth[3:3:end]
end

for interface in newGrid.interfaces_aligning_shock
    ck, _ = interface.face_1
    cv, _ = interface.face_2
    Pk, Pv = interface.P1, interface.P2
    # Pk_W, Pv_W = permutation_for_W(Pk), permutation_for_W(Pv)

    Rγk, Rγv = newGrid.Rγ[interface.face_1], newGrid.Rγ[interface.face_2]
    R̄γk, R̄γv = newGrid.R̄γ[interface.face_1], newGrid.R̄γ[interface.face_2]
    H̄γk, H̄γv = newGrid.H̄γ[interface.face_1], newGrid.H̄γ[interface.face_2]
    N̄xγk, N̄xγv = newGrid.N̄xγ[interface.face_1], newGrid.N̄xγ[interface.face_2]

    ρk_face, uk_face, pk_face = Rγk * W[ck, :, 1], Rγk * W[ck, :, 2], Rγk * W[ck, :, 3]
    U1k_face, U2k_face, U3k_face = Rγk * U[ck, :, 1], Rγk * U[ck, :, 2], Rγk * U[ck, :, 3]

    ρv_face, uv_face, pv_face = Rγv * W[cv, :, 1], Rγv * W[cv, :, 2], Rγv * W[cv, :, 3]
    U1v_face, U2v_face, U3v_face = Rγv * U[cv, :, 1], Rγv * U[cv, :, 2], Rγv * U[cv, :, 3]

    # for cell k
    Fxkk = two_point_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
    # Fxkv = two_point_flux_function(W[ck, :, :], W[cv, :, :])
    Ukk = two_point_state_flux_function_forSolver(W[ck, :, :], W[ck, :, :])
    # Ukv = two_point_state_flux_function(W[ck, :, :], W[cv, :, :])
    uk = vec(U[ck,:,:]')
    # wk = vec(W[ck,:,:]')
    # vk = V[ck, :]
    Fk = evaluate_Fx(W[ck, :, :])

    # for cell v
    Fxvv = two_point_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
    # Fxvk = two_point_flux_function(W[cv, :, :], W[ck, :, :])
    Uvv = two_point_state_flux_function_forSolver(W[cv, :, :], W[cv, :, :])
    # Uvk = two_point_state_flux_function(W[cv, :, :], W[ck, :, :])

    uv = vec(U[cv,:,:]')
    # wv = vec(W[cv,:,:]')
    # vv = V[cv, :]
    Fv = evaluate_Fx(W[cv, :, :])

    Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
    Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv
    
    Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
    Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk


    # Λk, Λv = Λ_matrix(U1k_face, U2k_face, U3k_face, ρk_face, uk_face, pk_face, Pk, U1v_face, U2v_face, U3v_face, ρv_face, uv_face, pv_face, Pv)
    # indicator = Rankine_Hugoniot_condition_indicator(U1k_face, U2k_face, U3k_face, ρk_face, uk_face, pk_face, Pk, U1v_face, U2v_face, U3v_face, ρv_face, uv_face, pv_face, Pv, newGrid.Nxγ[interface.face_1], newGrid.Ntγ[interface.face_1])
    # duk_smooth, duv_smooth = smooth_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxkv, Fxvv, Fxvk, Ukk, Ukv, Uvv, Uvk, One)
    duk_shock, duv_shock = shock_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxvv, Ukk, Uvv, uk, uv, Fk, Fv, One)
    # duk_dissipation, duv_dissipation = dissipation_term(R̄γk, R̄γv, H̄γk, H̄γv, N̄xγk, N̄xγv, Λk, Λv, vk, vv, Pk_W, Pv_W)
    # print("Indicator: ", indicator, "\n")
    # du[ck, :] += duk_shock
    # du[cv, :] += duv_shock

    du[ck, :, 1] += duk_shock[1:3:end]
    du[ck, :, 2] += duk_shock[2:3:end]
    du[ck, :, 3] += duk_shock[3:3:end]
    du[cv, :, 1] += duv_shock[1:3:end]
    du[cv, :, 2] += duv_shock[2:3:end]
    du[cv, :, 3] += duv_shock[3:3:end]
end

maximum(abs.(du))


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
    # du[cell_id, :] += Ēxγ .* Fxkk * One - Ēxγ * FBCL
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
    # du[cell_id, :] += Ēxγ .* Fxkk * One - kron(Rγ' * ref.H_face * Diagonal(Nxγ), I(3)) * evaluate_Fx(WBC)
    temp_du = Ēxγ .* Fxkk * One - kron(Rγ' * ref.H_face * Diagonal(Nxγ), I(3)) * evaluate_Fx(WBC)
    du[cell_id, :, 1] += temp_du[1:3:end]
    du[cell_id, :, 2] += temp_du[2:3:end]
    du[cell_id, :, 3] += temp_du[3:3:end]
end