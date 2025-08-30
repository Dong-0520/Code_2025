

function Chandrashekar1D_f1_forSolver(ρki::Float64, ρvj::Float64, uki::Float64, uvj::Float64)
    # return Trixi.ln_mean.(ρi, ρj) .* middle.(ui, uj)
    return Trixi.ln_mean(ρki, ρvj) * middle(uki, uvj)
end

function Chandrashekar1D_f2_forSolver(ρki::Float64, ρvj::Float64, uki::Float64, uvj::Float64, βki::Float64, βvj::Float64)
    # return middle.(ui, uj) .* Chandrashekar1D_f1(ρi, ρj, ui, uj) .+ (middle.(ρi, ρj) ./ (2 .* middle.(βi, βj)))
    return middle(uki, uvj) * Chandrashekar1D_f1_forSolver(ρki, ρvj, uki, uvj) + (middle(ρki, ρvj) / (2 * middle(βki, βvj)))
end

function Chandrashekar1D_f3_forSolver(ρki::Float64, ρvj::Float64, uki::Float64, uvj::Float64, βki::Float64, βvj::Float64)
    # return (1 ./ (2 * (γ - 1) .* Trixi.ln_mean.(βi, βj))) .* Chandrashekar1D_f1(ρi, ρj, ui, uj) .- (middle.(ui.^2, uj.^2) ./ 2) .* Chandrashekar1D_f1(ρi, ρj, ui, uj) .+ middle.(ui, uj) .* Chandrashekar1D_f2(ρi, ρj, ui, uj, βi, βj)
    return (1 / (2 * (γ - 1) * Trixi.ln_mean(βki, βvj))) * Chandrashekar1D_f1_forSolver(ρki, ρvj, uki, uvj) - (middle(uki^2, uvj^2) / 2) * Chandrashekar1D_f1_forSolver(ρki, ρvj, uki, uvj) + middle(uki, uvj) * Chandrashekar1D_f2_forSolver(ρki, ρvj, uki, uvj, βki, βvj)
end

function evaluate_Fx(U::Array{Float64, 3}; γ=1.4)

    result = zeros(size(U))

    ρ = U[:,:,1]
    ρu = U[:,:,2]
    E = U[:,:,3]

    ρu2 = @. ρu^2 / ρ
    p = @. (γ - 1) * (E - 0.5 * ρu2)

    result[:,:,1] = ρu
    result[:,:,2] = ρu2 + p
    result[:,:,3] = @. (ρu / ρ) * (E + p)

    return result

end

function evaluate_Fx(Wk::Matrix{Float64}; γ=1.4)
    """
    given Wk = [ρ, u, p]

          [ρ1, u1, p1]
          [ρ2, u2, p2]
          .
   Wk =       .
          .
          [ρN, uN, pN]

    return  one-dimensional vector containing
    [F1_1, F2_1, F3_1, F1_2, F2_2, F3_2. ..., F1_N, F2_N, F3_N]

    where Fi_j is the i^th spatial flux function evaluated at the j^th node of the k^th element
    """
    result = zeros(size(Wk))

    ρ = Wk[:,1]
    u = Wk[:,2]
    p = Wk[:,3]

    E = @. p/(γ-1) + 0.5 * ρ * u^2

    result[:,1] = @. ρ * u
    result[:,2] = @. p + ρ * u^2
    result[:,3] = @. u * (E + p)

    return vec(result')
end

function evaluate_W(U::Array{Float64, 3}; γ = 1.4)
    result = zeros(size(U))
    ρ = U[:,:,1]
    ρu = U[:,:,2]
    E = U[:,:,3]
    result[:,:,1] = ρ
    result[:,:,2] = @. ρu / ρ
    result[:,:,3] = @. (E - 0.5 * result[:,:,2]^2 * ρ) * (γ - 1)
    return result
end



function evaluate_entropy_variables(W::Array{Float64, 3}; γ = 1.4)
    num_of_cells = size(W)[1]
    num_of_nodes = size(W)[2]
    num_of_variables = size(W)[3]

    V = zeros(Float64, num_of_cells, num_of_nodes * num_of_variables)
    temp_V = zeros(Float64, num_of_cells, num_of_nodes, num_of_variables)
    ρ = W[:, :, 1]
    u = W[:, :, 2]
    p = W[:, :, 3]

    physical_s = log.(p) .- γ * log.(ρ)

    temp_V[:, :, 1] = (γ .- physical_s) ./ (γ - 1) .- ρ .* u.^2 ./ (2 * p)
    temp_V[:, :, 2] = ρ .* u ./ p
    temp_V[:, :, 3] = - ρ ./ p 

    for i in 1:num_of_cells
        V[i, :] = vec(temp_V[i, :, :]')
    end

    return V
end

function entropy_variables(ρ::Float64, u::Float64, p::Float64; γ = 1.4)
    """
    ρ: density
    u: velocity
    p: pressure
    γ: specific heat ratio (default is 1.4 for air)
    Returns the entropy variables 
        \frac{γ - s}{γ - 1} - \frac{ρ u^2}{2p} , \frac{ρu}{p} , \frac{- ρ}{p}
    in the form of a 1-dimensional array.
    """
    s = log(p) - γ * log(ρ)
    return (γ - s) / (γ - 1) - (ρ * u^2) / (2 * p), ρ * u / p, -ρ / p
end

function entropy_variables(ρk::Vector{Float64}, uk::Vector{Float64}, pk::Vector{Float64}; γ = 1.4)
    """
    inputs are variables on face γ
    output a vetcor of entropy variables evaluated at each node on the face

    """
    result = zeros(length(ρk) * 3)
    for i in eachindex(ρk)
        ρki = ρk[i]
        uki = uk[i]
        pki = pk[i]
        result[(i-1)*3 + 1 : 3*i] .= entropy_variables(ρki, uki, pki; γ = γ)
    end
    return result
end

function two_point_flux(Wki::Array{Float64, 1}, Wvj::Array{Float64, 1}; γ = 1.4)
    """
    Wki, Wvj: 1-dimensional arrays of primitive variables (density, velocity, pressure) (ρ, u, p)
    γ: specific heat ratio (default is 1.4 for air)
    Returns the Chandrashekar flux evaluated by the two nodes i, j from element k and element v
             ̂ρ ̄u 
        f* = ̄u f_1 + ̄ρ / (2 * ̄̄β)
             1/(2 * (γ - 1) * ̂̂β) f_1 - (1/2) * \bar{u^2} f_1 + ̄u f_2
    """          
    ρki, ρvj = Wki[1], Wvj[1]
    uki, uvj = Wki[2], Wvj[2]
    pki, pvj = Wki[3], Wvj[3]
    βki, βvj = ρki / (2 * pki), ρvj / (2 * pvj)
    return [Chandrashekar1D_f1_forSolver(ρki, ρvj, uki, uvj), Chandrashekar1D_f2_forSolver(ρki, ρvj, uki, uvj, βki, βvj), Chandrashekar1D_f3_forSolver(ρki, ρvj, uki, uvj, βki, βvj)]
end

function two_point_flux_function(Wk::Array{Float64, 2}, Wv::Array{Float64, 2}; γ = 1.4)
    """
    two argument matrix function
    Wk, Wv: 2-dimensional arrays of primitive variables (density, velocity, pressure) (ρ, u, p)
    say there are m variables and N nodes per element
    Wk, Wv: N * m
              ρ1 u1  p1
              ρ2 u2  p2
    Wk=       .  .    .
              .  .    .
              .  .    .
              ρN uN  pN
    """

    num_of_nodes, num_of_variables = size(Wk)

    block_size = Int.(num_of_variables * ones(num_of_nodes))
    block_matrix_Fx = BlockArray{Float64}(undef, block_size, block_size)
    for i in 1:num_of_nodes
        for j in 1:num_of_nodes
            Wki = Wk[i, :]
            Wvj = Wv[j, :]
            block_matrix_Fx[Block(i, j)] = diagm(two_point_flux(Wki, Wvj; γ = 1.4))
        end
    end
    return Matrix(block_matrix_Fx)

end

function two_point_state_flux(Wki::Array{Float64, 1}, Wvj::Array{Float64, 1}; γ = 1.4, R = 8.31446261815324)
    "
    This is 1D version, so u and v are m * 1 vectors containing the values of the variable on one specific nodes
                ρi,    ρj 
    ui, uj =    vi,    vj
              1/T i, 1/T j

    where, T = p / (R ρ)
           z = 1/T = R * ρ / p

    see paper by Yamaleev
    "

    ρki, ρvj = Wki[1], Wvj[1]
    uki, uvj = Wki[2], Wvj[2]
    pki, pvj = Wki[3], Wvj[3]


    zki, zvj = R * ρki / pki, R * ρvj / pvj

    ρ_logmean = Trixi.ln_mean(ρki, ρvj)
    v_mean = middle(uki, uvj)
    z_logmean = Trixi.ln_mean(zki, zvj) 
    
    return [ρ_logmean, ρ_logmean * v_mean, ρ_logmean * ( R/((γ-1)*z_logmean) + v_mean^2 - 0.5 * middle(uki^2, uvj^2) )]

end

function two_point_state_flux_function(Wk::Array{Float64, 2}, Wv::Array{Float64, 2}; γ = 1.4, R = 8.31446261815324)
    """
    two argument matrix function
    Wk, Wv: 2-dimensional arrays of primitive variables (density, velocity, pressure) (ρ, u, p)
    say there are m variables and N nodes per element
    Wk, Wv: N * m
              ρ1 u1  p1
              ρ2 u2  p2
    Wk=       .  .    .
              .  .    .
              .  .    .
              ρN uN  pN
    """

    num_of_nodes, num_of_variables = size(Wk)

    block_size = Int.(num_of_variables * ones(num_of_nodes))
    block_matrix_Fx = BlockArray{Float64}(undef, block_size, block_size)
    for i in 1:num_of_nodes
        for j in 1:num_of_nodes
            Wki = Wk[i, :]
            Wvj = Wv[j, :]
            block_matrix_Fx[Block(i, j)] = diagm(two_point_state_flux(Wki, Wvj))
        end
    end
    return Matrix(block_matrix_Fx)
    
end


function VOL_contribution!(du, newGrid, W, cell_id, One)

    # du[cell_id, :] += -2 * (newGrid.D̄t[cell_id, :, :] .* two_point_state_flux_function(W[cell_id, :, :], W[cell_id, :, :]) + newGrid.D̄x[cell_id, :, :] .* two_point_flux_function(W[cell_id, :, :], W[cell_id, :, :])) * One
    du[cell_id, :] += -2 * ((newGrid.H̄[cell_id,:,:] * newGrid.D̄t[cell_id, :, :]) .* two_point_state_flux_function(W[cell_id, :, :], W[cell_id, :, :]) + (newGrid.H̄[cell_id,:,:] * newGrid.D̄x[cell_id, :, :]) .* two_point_flux_function(W[cell_id, :, :], W[cell_id, :, :])) * One

end

function VOL_contribution_with_dissipation!(du, newGrid, W, V, cell_id, One; vol_diss_factor = 0.1)

    # entropy_k = entropy_variables(W[cell_id, :, 1], W[cell_id, :, 2], W[cell_id, :, 3])
    entropy_k = V[cell_id, :]
    Dx = newGrid.D̄x[cell_id, :, :]
    Dt = newGrid.D̄t[cell_id, :, :]
    H = newGrid.H̄[cell_id,:,:]
    dissipation = Dx' * H * Dx * entropy_k
    Ukk = two_point_state_flux_function(W[cell_id, :, :], W[cell_id, :, :])
    Fxkk = two_point_flux_function(W[cell_id, :, :], W[cell_id, :, :])
    # du[cell_id, :] += -2 * ((newGrid.H̄[cell_id,:,:] * newGrid.D̄t[cell_id, :, :]) .* two_point_state_flux_function(W[cell_id, :, :], W[cell_id, :, :]) + (newGrid.H̄[cell_id,:,:] * newGrid.D̄x[cell_id, :, :]) .* two_point_flux_function(W[cell_id, :, :], W[cell_id, :, :])) * One
    du[cell_id, :] += H * (-2 * (Dt .* Ukk + Dx .* Fxkk) * One - vol_diss_factor * dissipation)
    # print(dissipation, "\n")
end

function smooth_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, 
                                            Fxkk, Fxkv, Fxvv, Fxvk, Ukk, Ukv, Uvv, Uvk, 
                                            One)

    # du[ck, :] += (Ēxγk .* Fxkk - Ēxkv .* Fxkv + Ētγk .* Ukk - Ētkv .* Ukv) * One
    # du[cv, :] += (Ēxγv .* Fxvv - Ēxvk .* Fxvk + Ētγv .* Uvv - Ētvk .* Uvk) * One  
    duk = (Ēxγk .* Fxkk - Ēxkv .* Fxkv + Ētγk .* Ukk - Ētkv .* Ukv) * One
    duv = (Ēxγv .* Fxvv - Ēxvk .* Fxvk + Ētγv .* Uvv - Ētvk .* Uvk) * One
    return duk, duv
end

function shock_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, 
                                    Fxkk, Fxvv, Ukk, Uvv, uk, uv, Fk, Fv, One)

    # du[ck, :] += Ētγk .* Ukk * One - 0.5 * Ētγk * uk - 0.5 * Ētkv * uv + Ēxγk .* Fxkk * One - 0.5 * Ēxγk * Fk - 0.5 * Ēxkv * Fv
    # du[cv, :] += Ētγv .* Uvv * One - 0.5 * Ētγv * uv - 0.5 * Ētvk * uk + Ēxγv .* Fxvv * One - 0.5 * Ēxγv * Fv - 0.5 * Ēxvk * Fk
    duk = Ētγk .* Ukk * One - 0.5 * Ētγk * uk - 0.5 * Ētkv * uv + Ēxγk .* Fxkk * One - 0.5 * Ēxγk * Fk - 0.5 * Ēxkv * Fv
    duv = Ētγv .* Uvv * One - 0.5 * Ētγv * uv - 0.5 * Ētvk * uk + Ēxγv .* Fxvv * One - 0.5 * Ēxγv * Fv - 0.5 * Ēxvk * Fk
    return duk, duv
end

@doc raw"""
    Rankine-Hugoniot condition indicator
#Return 1 if the Rankine-Hugoniot condition is satisfied, otherwise return 0, where the condition is:
        (nx, nt) ∘ (Fxk, Uk) =  (nx, nt) ∘ (Fxv, Uv)
"""
function Rankine_Hugoniot_condition_indicator(U1k_face::Vector{Float64}, U2k_face::Vector{Float64}, U3k_face::Vector{Float64}, ρk_face::Vector{Float64}, uk_face::Vector{Float64}, pk_face::Vector{Float64}, Pk::Vector{Int64},
                                U1v_face::Vector{Float64}, U2v_face::Vector{Float64}, U3v_face::Vector{Float64}, ρv_face::Vector{Float64}, uv_face::Vector{Float64}, pv_face::Vector{Float64}, Pv::Vector{Int64},
                                Nxγk::Matrix{Float64}, Ntγk::Matrix{Float64}; tol = 1e-8)
                       
    F2k_face = ρk_face .* uk_face.^2 + pk_face
    F2v_face = ρv_face .* uv_face.^2 + pv_face
    F3k_face = uk_face .* (pk_face + U3k_face)
    F3v_face = uv_face .* (pv_face + U3v_face)

    return (all(isapprox.(Ntγk * U1k_face + Nxγk * U2k_face - Ntγk * U1v_face[Pv] - Nxγk * U2v_face[Pv], 0, atol = tol)) &
            all(isapprox.(Ntγk * U2k_face + Nxγk * F2k_face - Ntγk * U2v_face[Pv] - Nxγk * F2v_face[Pv], 0, atol = tol)) & 
                all(isapprox.(Ntγk * U3k_face + Nxγk * F3k_face - Ntγk * U3v_face[Pv] - Nxγk * F3v_face[Pv], 0, atol = tol)))
    
end

