@doc raw"""
Compute the Jacobian matrix for the transformation between conserved variables ``U`` and primitive variables ``W``.
    i.e. ∂U/∂W

# Arguments are precomputed averaged (arithmetic or Roe average) in the RHS function
- `U1::Float64`: First conserved variable ρ (density)
- `U2::Float64`: Second conserved variable ρu (momentum) 
- `U3::Float64`: Third conserved variable E (total energy)

# Returns
A 3×3 Jacobian matrix:
    |   ρ                       ρu                                                   E                   |
    |   ρu       (2γρE - γρ^2 u^2 - 2ρE + 3 ρ^2 u^2) / 2ρ         ρu(2γρE - γρ^2u^2 + ρ^2u^2) / (2ρ^2)   |
    |   E           (2γρE - γρ^2u^2 + ρ^2 u^2)ρu / (2ρ^2)        (4γρ^2E^2 - γρ^4u^4 + ρ^4u^4) / (4 ρ^3) |
or
    |   U1                  U2                                 U3                   |
    |   U2     (2(γ-1)U1U3 - U2^2(γ-3)) /(2 U1)    ((1-γ)U2^3 + 2γU1U2U3)/(2U1^2)   |
    |   U3     ((1-γ)U2^3 + 2γU1U2U3) / (2 U1^2)   (4γU1^2U3^2 + (1-γ)U2^4)/(4U1^3) |
evaluated at the averaged state.
"""
function ∂U∂W(U1::Float64, U2::Float64, U3::Float64; γ = 1.4)
    return [ U1 U2 U3;
            U2 (2 * (γ - 1) * U1 * U3 - (γ-3) * U2^2)/(2 * U1) ((1-γ) * U2^3 + 2 * γ * U1 * U2 * U3)/(2 * U1^2);
            U3 ((1-γ) * U2^3 + 2*γ*U1*U2*U3)/(2 * U1^2) (4 * γ * U1^2 * U3^2 + (1-γ) * U2^4) / (4 * U1^3)]
end


# function max_eigenvalues_on_face(ρkγ, ukγ, pkγ, ρvγ, uvγ, pvγ)
#     """
#     inputs are variables on the face, not the element
#     of size num_of_nodes * 1
#     output the maximum of eigenvalues of the flux Jacobian matrix on the face
#     """
#     max_eigen = 0
#     for i in eachindex(ρkγ)
#         # a =  √γp/ρ
#         aki = (γ * pkγ[i] / ρkγ[i])^(1/2)
#         λ1ki = abs(ukγ[i] - aki)
#         λ2ki = abs(ukγ[i])
#         λ3ki = abs(ukγ[i] + aki)

#         avi = (γ * pvγ[i] / ρvγ[i])^(1/2)
#         λ1vi = abs(uvγ[i] - avi)
#         λ2vi = abs(uvγ[i])
#         λ3vi = abs(uvγ[i] + avi)

#         max_eigen = maximum([max_eigen, λ1ki, λ2ki, λ3ki, λ1vi, λ2vi, λ3vi])
#         # print("max_eigen: ", max_eigen, "\n")
#     end
#     return max_eigen

# end

@doc raw"""
    Given the ρ, u, p of one node on the face, there are 6 values from element k and v
    Compute the eigenvalues on k and v respectively, and return the maximum absolute of them
    the eigenvalues are:
        λ1 = u - a, λ2 = u, λ3 = u + a
    where a = √(γp/ρ) is the speed of sound
"""
function max_eigenvalue(ρk::Float64, uk::Float64, pk::Float64, ρv::Float64, uv::Float64, pv::Float64; γ = 1.4)
    # a =  √γp/ρ
    ak = (γ * pk / ρk)^(1/2)
    λ1k = abs(uk - ak)
    λ2k = abs(uk)
    λ3k = abs(uk + ak)

    av = (γ * pv / ρv)^(1/2)
    λ1v = abs(uv - av)
    λ2v = abs(uv)
    λ3v = abs(uv + av)

    return max(λ1k, λ2k, λ3k, λ1v, λ2v, λ3v)
end

@doc raw"""
    given the ρ, u, p, ρ, ρu, E on face of element k and v respectively, compute the Λ matrix for k and v
    see the function ∂U∂W for the definition of ∂U/∂W
    and the function max_eigenvalue for the definition of max(eigenvalues)
    
#Returns    

        Λk: dissipation block diagonal matrix for k 
        Λv: dissipation block diagonal matrix for v
        Λ is a block diagonal matrix whose blocks are max(eigenvalues) * ∂U/∂W:
           Λ_ii = [ max(λ_i) * ∂U/∂W_ii ]
"""
function Λ_matrix(U1k_face::Vector{Float64}, U2k_face::Vector{Float64}, U3k_face::Vector{Float64}, ρk_face::Vector{Float64}, uk_face::Vector{Float64}, pk_face::Vector{Float64}, Pk::Vector{Int64},
                U1v_face::Vector{Float64}, U2v_face::Vector{Float64}, U3v_face::Vector{Float64}, ρv_face::Vector{Float64}, uv_face::Vector{Float64}, pv_face::Vector{Float64}, Pv::Vector{Int64}; 
                γ = 1.4, num_of_variables = 3)
    num_of_nodes_face = length(U1k_face)
    block_size = Int.(num_of_variables * ones(num_of_nodes_face))
    Λk = zero(BlockArray{Float64}(undef, block_size, block_size))
    Λv = zero(BlockArray{Float64}(undef, block_size, block_size))
    ρk_adj = ρv_face[Pv]
    uk_adj = uv_face[Pv]
    pk_adj = pv_face[Pv]
    U1k_adj = U1v_face[Pv]
    U2k_adj = U2v_face[Pv]
    U3k_adj = U3v_face[Pv]

    ρv_adj = ρk_face[Pk]
    uv_adj = uk_face[Pk]
    pv_adj = pk_face[Pk]
    U1v_adj = U1k_face[Pk]
    U2v_adj = U2k_face[Pk]
    U3v_adj = U3k_face[Pk]
    for i in 1:num_of_nodes_face
        Λk[Block(i, i)] = max_eigenvalue(ρk_face[i], uk_face[i], pk_face[i], ρk_adj[i], uk_adj[i], pk_adj[i]) * ∂U∂W( 0.5 * (U1k_face[i] + U1k_adj[i]),
                                                                                                                            0.5 * (U2k_face[i] + U2k_adj[i]),
                                                                                                                            0.5 * (U3k_face[i] + U3k_adj[i]))
    end

    for i in 1:num_of_nodes_face
        # Λv[Block(i,i)] = Λk[Block(Pk[i], Pk[i])]
        Λv[Block(i,i)] = max_eigenvalue(ρv_face[i], uv_face[i], pv_face[i], ρv_adj[i], uv_adj[i], pv_adj[i]) * ∂U∂W( 0.5 * (U1v_face[i] + U1v_adj[i]),
                                                                                                                            0.5 * (U2v_face[i] + U2v_adj[i]),
                                                                                                                            0.5 * (U3v_face[i] + U3v_adj[i]))
    end

    return Matrix(Λk), Matrix(Λv)
end

@doc raw"""
    Given the entropy variables on the element k and v, and the Λ matrix for k and v on the face
#Returns
    dissipation_k: the dissipation term for k, which is a vector of size num_of_nodes_face * 1
    dissipation_v: the dissipation term for v, which is a vector of size num_of_nodes_face * 1
    the dissipation term is computed as:
        dissipation_k = R̄γk' * H̄γk * N̄xγk * Λk * (R̄γk * wk - (R̄γv * wv)[Pv_W])
        dissipation_v = R̄γv' * H̄γv * N̄xγv * Λv * (R̄γv * wv - (R̄γk * wk)[Pk_W])
"""
function dissipation_term(R̄γk, R̄γv, H̄γk, H̄γv, N̄xγk, N̄xγv, Λk, Λv, vk, vv, Pk_W, Pv_W)

    dissipation_k = R̄γk' * H̄γk * abs.(N̄xγk) * Λk * (R̄γk * vk - (R̄γv * vv)[Pv_W])
    dissipation_v = R̄γv' * H̄γv * abs.(N̄xγv) * Λv * (R̄γv * vv - (R̄γk * vk)[Pk_W])
    # dissipation_k = R̄γk' * H̄γk * Λk * (R̄γk * vk - (R̄γv * vv)[Pv_W])
    # dissipation_v = R̄γv' * H̄γv * Λv * (R̄γv * vv - (R̄γk * vk)[Pk_W])
    return dissipation_k, dissipation_v
    
end



#----------------------- Debug ----------------------------------
# num_of_variables = 3
# test_W = zeros(n_cells(grid), length(grid.xyz[1]), 3) # ρ = 3, u = 2, p = 10
# test_U = zeros(n_cells(grid), length(grid.xyz[1]), 3) # ρ = 3, u = 2, p = 10
# for cell_id in 1:n_cells(grid)
#     xyzk = grid.xyz[cell_id]
#     xk = [x[1] for x in xyzk]
#     yk = [x[2] for x in xyzk]
#     test_W[cell_id, :,1] .= [xk[i]^2 * yk[i]^2 for i in 1:12]
#     test_W[cell_id, :,2] .= [xk[i] * yk[i]^2 for i in 1:12]
#     test_W[cell_id, :,3] .= [xk[i]^2 * yk[i]^2 + 5 for i in 1:12]
# end

# test_U[:, :, 1] = deepcopy(test_W[:, :, 1])
# test_U[:, :, 2] = test_W[:, :, 2] .* test_W[:, :, 1]
# test_U[:, :, 3] = test_W[:, :, 3] ./ 0.4 + 0.5 * test_W[:, :, 1] .* test_W[:, :, 2].^2


# interface = newGrid.interfaces_aligning_shock[1]
# ck, lfk = interface.face_1
# cv, lfv = interface.face_2
# Rγk, Rγv = newGrid.Rγ[interface.face_1], newGrid.Rγ[interface.face_2]
# R̄γk, R̄γv = newGrid.R̄γ[interface.face_1], newGrid.R̄γ[interface.face_2]
# Pk, Pv = interface.P1, interface.P2
# Pk_W, Pv_W = permutation_for_W(Pk), permutation_for_W(Pv)

# H̄γk = newGrid.H̄γ[interface.face_1]
# N̄xγk = newGrid.N̄xγ[interface.face_1]

# H̄γv = newGrid.H̄γ[interface.face_2]
# N̄xγv = newGrid.N̄xγ[interface.face_2]

# ρk_face, uk_face, pk_face = Rγk * test_W[ck, :, 1], Rγk * test_W[ck, :, 2], Rγk * test_W[ck, :, 3]
# U1k_face, U2k_face, U3k_face = Rγk * test_U[ck, :, 1], Rγk * test_U[ck, :, 2], Rγk * test_U[ck, :, 3]

# ρv_face, uv_face, pv_face = Rγv * test_W[cv, :, 1], Rγv * test_W[cv, :, 2], Rγv * test_W[cv, :, 3]
# U1v_face, U2v_face, U3v_face = Rγv * test_U[cv, :, 1], Rγv * test_U[cv, :, 2], Rγv * test_U[cv, :, 3]

# num_of_nodes_face = length(U1k_face)
# block_size = Int.(num_of_variables * ones(num_of_nodes_face))
# block_matrix_Λk = zero(BlockArray{Float64}(undef, block_size, block_size))
# for i in 1:num_of_nodes_face
#     block_matrix_Λk[Block(i, i)] = max_eigenvalue(ρk_face[i], uk_face[i], pk_face[i], ρk_adj[i], uk_adj[i], pk_face[i])* ∂U∂W( 0.5 * (U1k_face[i] + U1k_adj[i]),
#                                                                                                                             0.5 * (U2k_face[i] + U2k_adj[i]),
#                                                                                                                             0.5 * (U3k_face[i] + U3k_adj[i]))
# end
# Matrix(block_matrix_Λk)

# ρv_face, ρv_adj = Rγv * test_W[cv, :, 1], (Rγk * test_W[ck, :, 1])[Pk]
# uv_face, uv_adj = Rγv * test_W[cv, :, 2], (Rγk * test_W[ck, :, 2])[Pk]
# pv_face, pv_adj = Rγv * test_W[cv, :, 3], (Rγk * test_W[ck, :, 3])[Pk]

# U1v_face, U1v_adj = Rγv * test_U[cv, :, 1], (Rγk * test_U[ck, :, 1])[Pk]
# U2v_face, U2v_adj = Rγv * test_U[cv, :, 2], (Rγk * test_U[ck, :, 2])[Pk]
# U3v_face, U3v_adj = Rγv * test_U[cv, :, 3], (Rγk * test_U[ck, :, 3])[Pk]
# block_matrix_Λv = zero(BlockArray{Float64}(undef, block_size, block_size))
# for i in 1:num_of_nodes_face
#     block_matrix_Λv[Block(i, i)] = max_eigenvalue(ρv_face[i], uv_face[i], pv_face[i], ρv_adj[i], uv_adj[i], pv_face[i])* ∂U∂W( 0.5 * (U1v_face[i] + U1v_adj[i]),
#                                                                                                                             0.5 * (U2v_face[i] + U2v_adj[i]),
#                                                                                                                             0.5 * (U3v_face[i] + U3v_adj[i]))
# end
# Matrix(block_matrix_Λv)

# for i in 1:52
#     interface = newGrid.interior_interfaces[i]
#     ck, lfk = interface.face_1
#     cv, lfv = interface.face_2

#     Pk, Pv = interface.P1, interface.P2
#     Pk_W, Pv_W = permutation_for_W(Pk), permutation_for_W(Pv)

#     Rγk, Rγv = newGrid.Rγ[interface.face_1], newGrid.Rγ[interface.face_2]
#     R̄γk, R̄γv = newGrid.R̄γ[interface.face_1], newGrid.R̄γ[interface.face_2]
#     H̄γk, H̄γv = newGrid.H̄γ[interface.face_1], newGrid.H̄γ[interface.face_2]
#     N̄xγk, N̄xγv = newGrid.N̄xγ[interface.face_1], newGrid.N̄xγ[interface.face_2]

#     ρk_face, uk_face, pk_face = Rγk * W[ck, :, 1], Rγk * W[ck, :, 2], Rγk * W[ck, :, 3]
#     U1k_face, U2k_face, U3k_face = Rγk * U[ck, :, 1], Rγk * U[ck, :, 2], Rγk * U[ck, :, 3]

#     ρv_face, uv_face, pv_face = Rγv * W[cv, :, 1], Rγv * W[cv, :, 2], Rγv * W[cv, :, 3]
#     U1v_face, U2v_face, U3v_face = Rγv * U[cv, :, 1], Rγv * U[cv, :, 2], Rγv * U[cv, :, 3]

#     # for cell k
#     Fxkk = two_point_flux_function(W[ck, :, :], W[ck, :, :])
#     Fxkv = two_point_flux_function(W[ck, :, :], W[cv, :, :])
#     Ukk = two_point_state_flux_function(W[ck, :, :], W[ck, :, :])
#     Ukv = two_point_state_flux_function(W[ck, :, :], W[cv, :, :])
#     vk = V[ck, :]

#     # for cell v
#     Fxvv = two_point_flux_function(W[cv, :, :], W[cv, :, :])
#     Fxvk = two_point_flux_function(W[cv, :, :], W[ck, :, :])
#     Uvv = two_point_state_flux_function(W[cv, :, :], W[cv, :, :])
#     Uvk = two_point_state_flux_function(W[cv, :, :], W[ck, :, :])
#     vv = V[cv, :]

#     Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
#     Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv

#     Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
#     Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk

#     duk_smooth, duv_smooth = smooth_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxkv, Fxvv, Fxvk, Ukk, Ukv, Uvv, Uvk, One)
#     Λk, Λv = Λ_matrix(U1k_face, U2k_face, U3k_face, ρk_face, uk_face, pk_face, Pk, U1v_face, U2v_face, U3v_face, ρv_face, uv_face, pv_face, Pv)
#     duk_dissipation, duv_dissipation = dissipation_term(R̄γk, R̄γv, H̄γk, H̄γv, N̄xγk, N̄xγv, Λk, Λv, vk, vv, Pk_W, Pv_W)
#     # print(maximum(abs.(duk_dissipation)), "\n")
#     # print(maximum(abs.(duv_dissipation)), "\n")
#     if maximum(abs.(duk_dissipation)) > 1e-5 || maximum(abs.(duv_dissipation)) > 1e-5
#         println("dissipation $i term is not zero \n ")
#         print("ck is ", ck, "\n")
#         print("cv is ", cv, "\n")
#     end
# end

# interface = newGrid.interior_interfaces[4] # 4 and 37 
# ck, lfk = interface.face_1
# cv, lfv = interface.face_2
# Pk, Pv = interface.P1, interface.P2
# Pk_W, Pv_W = permutation_for_W(Pk), permutation_for_W(Pv)

# Rγk, Rγv = newGrid.Rγ[interface.face_1], newGrid.Rγ[interface.face_2]
# R̄γk, R̄γv = newGrid.R̄γ[interface.face_1], newGrid.R̄γ[interface.face_2]
# H̄γk, H̄γv = newGrid.H̄γ[interface.face_1], newGrid.H̄γ[interface.face_2]
# N̄xγk, N̄xγv = newGrid.N̄xγ[interface.face_1], newGrid.N̄xγ[interface.face_2]

# ρk_face, uk_face, pk_face = Rγk * W[ck, :, 1], Rγk * W[ck, :, 2], Rγk * W[ck, :, 3]
# U1k_face, U2k_face, U3k_face = Rγk * U[ck, :, 1], Rγk * U[ck, :, 2], Rγk * U[ck, :, 3]

# ρv_face, uv_face, pv_face = Rγv * W[cv, :, 1], Rγv * W[cv, :, 2], Rγv * W[cv, :, 3]
# U1v_face, U2v_face, U3v_face = Rγv * U[cv, :, 1], Rγv * U[cv, :, 2], Rγv * U[cv, :, 3]

# # for cell k
# Fxkk = two_point_flux_function(W[ck, :, :], W[ck, :, :])
# Fxkv = two_point_flux_function(W[ck, :, :], W[cv, :, :])
# Ukk = two_point_state_flux_function(W[ck, :, :], W[ck, :, :])
# Ukv = two_point_state_flux_function(W[ck, :, :], W[cv, :, :])
# vk = V[ck, :]

# # for cell v
# Fxvv = two_point_flux_function(W[cv, :, :], W[cv, :, :])
# Fxvk = two_point_flux_function(W[cv, :, :], W[ck, :, :])
# Uvv = two_point_state_flux_function(W[cv, :, :], W[cv, :, :])
# Uvk = two_point_state_flux_function(W[cv, :, :], W[ck, :, :])
# vv = V[cv, :]

# Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
# Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv

# Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
# Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk

# duk_smooth, duv_smooth = smooth_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, Fxkk, Fxkv, Fxvv, Fxvk, Ukk, Ukv, Uvv, Uvk, One)
# Λk, Λv = Λ_matrix(U1k_face, U2k_face, U3k_face, ρk_face, uk_face, pk_face, Pk, U1v_face, U2v_face, U3v_face, ρv_face, uv_face, pv_face, Pv)
# duk_dissipation, duv_dissipation = dissipation_term(R̄γk, R̄γv, H̄γk, H̄γv, N̄xγk, N̄xγv, Λk, Λv, vk, vv, Pk_W, Pv_W)