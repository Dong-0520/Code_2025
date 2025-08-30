const POS_EPS = 1e-12   # 钳制到正区，防止分母/对数为0
const LOG_EPS = 1e-12   # 对数下限
"""
    safe_log_forSolver(x)

对 x 做下限钳制后再取 log，保证永远是有限数。
"""
function safe_log_forSolver(x; ε=LOG_EPS)
    return log(max(x, ε))
end

"""
    safe_ln_mean_forSolver(a, b)

用 (b - a)/(log b - log a) 计算对数平均，
在 a≈b 时直接退化到算术平均，绝不产生 NaN/Inf。
"""
@inline function safe_ln_mean_forSolver(a, b)

    POS_EPS = 1e-12
    LOG_EPS = 1e-12
    LNMEAN_EPS = 1e-14
    a_p = max(a, POS_EPS)
    b_p = max(b, POS_EPS)

    Δlog = log(b_p) - log(a_p)

    # ───────── 关键行 ─────────
    return ifelse(abs(Δlog) < LNMEAN_EPS,
                  0.5*(a_p + b_p),         # a≈b → 算术平均
                  (b_p - a_p)/Δlog)        # 否则 log-mean
end

function Chandrashekar1D_f1_forSolver(ρki::T, ρvj::T, uki::T, uvj::T) where T
    # return Trixi.ln_mean(ρki, ρvj) * middle(uki, uvj)
    # return (@safe_ln_mean ρki ρvj) * middle(uki, uvj) # Trixi.ln_mean(ρki, ρvj) * middle(uki, uvj)
    return safe_ln_mean_forSolver(ρki, ρvj) * middle(uki, uvj)
end

function Chandrashekar1D_f2_forSolver(ρki::T, ρvj::T, uki::T, uvj::T, βki::T, βvj::T) where T
    return middle(uki, uvj) * Chandrashekar1D_f1_forSolver(ρki, ρvj, uki, uvj) + (middle(ρki, ρvj) / (2 * middle(βki, βvj)))
end

function Chandrashekar1D_f3_forSolver(ρki::T, ρvj::T, uki::T, uvj::T, βki::T, βvj::T; γ = 1.4) where T
    # return (1 / (2 * (γ - 1) * Trixi.ln_mean(βki, βvj))) * Chandrashekar1D_f1(ρki, ρvj, uki, uvj) -
    #        (middle(uki^2, uvj^2) / 2) * Chandrashekar1D_f1(ρki, ρvj, uki, uvj) +
    #        middle(uki, uvj) * Chandrashekar1D_f2(ρki, ρvj, uki, uvj, βki, βvj)
    # return (1 / (2 * (γ - 1) * (@safe_ln_mean βki βvj))) * Chandrashekar1D_f1(ρki, ρvj, uki, uvj) -
    #        (middle(uki^2, uvj^2) / 2) * Chandrashekar1D_f1(ρki, ρvj, uki, uvj) +
    #        middle(uki, uvj) * Chandrashekar1D_f2(ρki, ρvj, uki, uvj, βki, βvj)

    return (1 / (2 * (γ - 1) * safe_ln_mean_forSolver(βki, βvj))) * Chandrashekar1D_f1_forSolver(ρki, ρvj, uki, uvj) -
           (middle(uki^2, uvj^2) / 2) * Chandrashekar1D_f1_forSolver(ρki, ρvj, uki, uvj) +
           middle(uki, uvj) * Chandrashekar1D_f2_forSolver(ρki, ρvj, uki, uvj, βki, βvj)
end

function evaluate_Fx(Wk::AbstractMatrix{T}; γ = 1.4) where T

    num_nodes, num_vars = size(Wk)
    result = zeros(eltype(Wk), num_nodes, num_vars)

    ρ = Wk[:, 1]
    u = Wk[:, 2]
    p = Wk[:, 3]

    E = @. p / (γ - 1) + 0.5 * ρ * u^2

    @. result[:, 1] = ρ * u
    @. result[:, 2] = p + ρ * u^2
    @. result[:, 3] = u * (E + p)

    return vec(permutedims(result, (2, 1)))
end

function evaluate_W_forSolver(U; γ = 1.4)
    result = similar(U)  # 注意这里
    ρ = U[:,:,1]
    ρu = U[:,:,2]
    E = U[:,:,3]
    @. result[:,:,1] = ρ
    @. result[:,:,2] = ρu / ρ
    # @. result[:,:,3] = (E - 0.5 * result[:,:,2]^2 * ρ) * (γ - 1)
    result[:,:,3] = @. max((E - 0.5 * result[:,:,2]^2 * ρ) * (γ - 1), 1e-13)
    return result
end

function evaluate_entropy_variables_forSolver(W; γ = 1.4)
    num_of_cells, num_of_nodes, num_of_variables = size(W)

    V = similar(W, eltype(W), (num_of_cells, num_of_nodes * num_of_variables))
    temp_V = similar(W)

    ρ = W[:, :, 1]
    u = W[:, :, 2]
    p = W[:, :, 3]
    physical_s = log.(max.(p, 1e-13)) .- γ * log.(max.(ρ, 1e-13))

    @. temp_V[:, :, 1] = (γ - physical_s) / (γ - 1) - ρ * u^2 / (2 * p)
    @. temp_V[:, :, 2] = ρ * u / p
    @. temp_V[:, :, 3] = - ρ / p

    for i in 1:num_of_cells
        V[i, :] = vec(permutedims(temp_V[i, :, :], (2, 1)))
    end

    return V
end

function VOL_contribution!(du, newGrid, W, cell_id, One)

    Dx = newGrid.D̄x[cell_id, :, :]
    Dt = newGrid.D̄t[cell_id, :, :]
    H = newGrid.H̄[cell_id, :, :]

    Ukk = two_point_state_flux_function_forSolver(W[cell_id, :, :], W[cell_id, :, :])
    Fxkk = two_point_flux_function_forSolver(W[cell_id, :, :], W[cell_id, :, :])

    temp = -2 .* (Dt .* Ukk .+ Dx .* Fxkk) * One
    temp_du = H * temp

    du[cell_id, :, 1] .+= temp_du[1:3:end]
    du[cell_id, :, 2] .+= temp_du[2:3:end]
    du[cell_id, :, 3] .+= temp_du[3:3:end]
    return nothing
end


function two_point_flux_forSolver(Wki::AbstractArray{T, 1}, Wvj::AbstractArray{T, 1}; γ = 1.4) where T
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

function two_point_flux_function_forSolver(Wk::AbstractArray{T, 2}, Wv::AbstractArray{T, 2}; γ = 1.4) where T
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
    block_matrix_Fx = BlockArray{promote_type(eltype(Wk), eltype(Wv))}(undef, block_size, block_size)
    for i in 1:num_of_nodes
        for j in 1:num_of_nodes
            Wki = Wk[i, :]
            Wvj = Wv[j, :]
            block_matrix_Fx[Block(i, j)] = Diagonal(two_point_flux_forSolver(Wki, Wvj; γ = 1.4))
        end
    end
    return Matrix(block_matrix_Fx)

end

function two_point_state_flux_forSolver(Wki::AbstractArray{T, 1}, Wvj::AbstractArray{T, 1}; γ = 1.4, R = 8.31446261815324) where T
    # 提取变量
    ρki, ρvj = Wki[1], Wvj[1]
    uki, uvj = Wki[2], Wvj[2]
    pki, pvj = Wki[3], Wvj[3]

    # ⚡保护 p，防止除以 0 或负数
    pki = ifelse(pki > 1e-12, pki, 1e-12)
    pvj = ifelse(pvj > 1e-12, pvj, 1e-12)

    # 计算 z
    zki, zvj = R * ρki / pki, R * ρvj / pvj

    # 计算 ρ_logmean, v_mean, z_logmean
    # ρ_logmean = Trixi.ln_mean(ρki, ρvj)
    # ρ_logmean = @safe_ln_mean ρki ρvj
    ρ_logmean = safe_ln_mean_forSolver(ρki, ρvj)
    v_mean = middle(uki, uvj)
    # z_logmean = Trixi.ln_mean(zki, zvj)
    # z_logmean = @safe_ln_mean zki zvj
    z_logmean = safe_ln_mean_forSolver(zki, zvj)

    # ⚡保护 z_logmean，防止除以接近0的数
    z_logmean = ifelse(abs(z_logmean) > 1e-12, z_logmean, 1e-12)

    # 最后输出 flux
    return [
        ρ_logmean,
        ρ_logmean * v_mean,
        ρ_logmean * (R / ((γ - 1) * z_logmean) + v_mean^2 - 0.5 * middle(uki^2, uvj^2))
    ]
end


function two_point_state_flux_function_forSolver(Wk::AbstractArray{T, 2}, Wv::AbstractArray{T, 2}; γ = 1.4, R = 8.31446261815324) where T
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
    block_matrix_Fx = BlockArray{promote_type(eltype(Wk), eltype(Wv))}(undef, block_size, block_size)
    for i in 1:num_of_nodes
        for j in 1:num_of_nodes
            Wki = Wk[i, :]
            Wvj = Wv[j, :]
            block_matrix_Fx[Block(i, j)] = Diagonal(two_point_state_flux_forSolver(Wki, Wvj))
        end
    end
    return Matrix(block_matrix_Fx)
    
end

function smooth_interface_contribution(Ēxγk, Ēxkv, Ētγk, Ētkv, Ēxγv, Ēxvk, Ētγv, Ētvk, 
                                            Fxkk, Fxkv, Fxvv, Fxvk, Ukk, Ukv, Uvv, Uvk, 
                                            One)

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