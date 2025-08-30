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




function VOL_contribution_with_dissipation_forSolver!(du, newGrid, W, V, cell_id, One; vol_diss_factor = 0.1)



    entropy_k = V[cell_id, :]
    Dx = newGrid.D̄x[cell_id, :, :]
    Dt = newGrid.D̄t[cell_id, :, :]
    H = newGrid.H̄[cell_id, :, :]
    dissipation = Dx' * H * Dx * entropy_k

    Ukk = two_point_state_flux_function_forSolver(W[cell_id, :, :], W[cell_id, :, :])
    Fxkk = two_point_flux_function_forSolver(W[cell_id, :, :], W[cell_id, :, :])

    # println("maximum(abs(Ukk)) = ", maximum(abs, Ukk))
    # println("maximum(abs(Fxkk)) = ", maximum(abs, Fxkk))

    temp = -2 .* (Dt .* Ukk .+ Dx .* Fxkk) * One
    temp .-= vol_diss_factor .* dissipation
    temp_du = H * temp

    # println("maximum(abs(temp)) = ", maximum(abs, temp))
    # println("maximum(abs(temp_du)) = ", maximum(abs, temp_du))  


    du[cell_id, :, 1] .+= temp_du[1:3:end]
    du[cell_id, :, 2] .+= temp_du[2:3:end]
    du[cell_id, :, 3] .+= temp_du[3:3:end]
    return nothing
end