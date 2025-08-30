

using Pkg

using LinearAlgebra, SparseArrays, Trixi, BlockArrays
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics        

include("../src/SBPLite.jl")
using .SBPLite
include("../Burgers//plotting_helper.jl")
γ = 1.4

ρL = 3.0
uL = 2.0
pL = 10.0
ρR = 3.0
uR = -1.0
pR = 5.0

ρstarL = 5.013935153894693
ρstarR = 7.67671462561582
ustar = 0.7943931050164833
pstar = 20.85590303477569
@load joinpath(@__DIR__, "Euler_mesh_12.jld2") grid


function Chandrashekar1D_f1(ρki::Float64, ρvj::Float64, uki::Float64, uvj::Float64)
    # return Trixi.ln_mean.(ρi, ρj) .* middle.(ui, uj)
    return Trixi.ln_mean(ρki, ρvj) * middle(uki, uvj)
end

function Chandrashekar1D_f2(ρki::Float64, ρvj::Float64, uki::Float64, uvj::Float64, βki::Float64, βvj::Float64)
    # return middle.(ui, uj) .* Chandrashekar1D_f1(ρi, ρj, ui, uj) .+ (middle.(ρi, ρj) ./ (2 .* middle.(βi, βj)))
    return middle(uki, uvj) * Chandrashekar1D_f1(ρki, ρvj, uki, uvj) + (middle(ρki, ρvj) / (2 * middle(βki, βvj)))
end

function Chandrashekar1D_f3(ρki::Float64, ρvj::Float64, uki::Float64, uvj::Float64, βki::Float64, βvj::Float64)
    # return (1 ./ (2 * (γ - 1) .* Trixi.ln_mean.(βi, βj))) .* Chandrashekar1D_f1(ρi, ρj, ui, uj) .- (middle.(ui.^2, uj.^2) ./ 2) .* Chandrashekar1D_f1(ρi, ρj, ui, uj) .+ middle.(ui, uj) .* Chandrashekar1D_f2(ρi, ρj, ui, uj, βi, βj)
    return (1 / (2 * (γ - 1) * Trixi.ln_mean(βki, βvj))) * Chandrashekar1D_f1(ρki, ρvj, uki, uvj) - (middle(uki^2, uvj^2) / 2) * Chandrashekar1D_f1(ρki, ρvj, uki, uvj) + middle(uki, uvj) * Chandrashekar1D_f2(ρki, ρvj, uki, uvj, βki, βvj)
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
    return [Chandrashekar1D_f1(ρki, ρvj, uki, uvj), Chandrashekar1D_f2(ρki, ρvj, uki, uvj, βki, βvj), Chandrashekar1D_f3(ρki, ρvj, uki, uvj, βki, βvj)]
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

function find_interfaces_align_shock(grid)
    interfaces_set = Set{FaceInterface}()
    for (cell_id, face_id) in union(get_face_set(grid, "ONE_WAVE"), get_face_set(grid, "THREE_WAVE"))
        neighbor_id, neighbor_face_id = Tuple(findall(x -> x == FaceIndex((cell_id, face_id)), grid.topology.face_face_neighbours)[1])
        interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((neighbor_id, neighbor_face_id)))
        push!(interfaces_set, interface)
    end
    return interfaces_set
end


struct Ematrices

    # all E matrices needed for evaluating the interior interfaces contribution
    # Ex^γk, Ex^kv, Et^γk, Et^kv for ck contribution
    # Ex^γv, Ex^vk, Et^γv, Et^vk for cv contribution

    Exγk::Matrix{Float64}
    Exkv::Matrix{Float64}
    Etγk::Matrix{Float64}
    Etkv::Matrix{Float64}

    Exγv::Matrix{Float64}
    Exvk::Matrix{Float64}
    Etγv::Matrix{Float64}
    Etvk::Matrix{Float64}


end

struct myGrid

    interfaces_aligning_shock::Vector{FaceInterface} # the interfaces that are aligning with the shock
    interior_interfaces::Vector{FaceInterface} # the interior interfaces that are not aligning with the shock

    H::Array{Float64, 3}
    Hinv::Array{Float64, 3}
    Dx::Array{Float64, 3}
    Dt::Array{Float64, 3}
    E_for_interior_interfaces::Dict{FaceInterface, Ematrices} # the E matrices for the interior interfaces

    H̄::Array{Float64, 3}
    H̄inv::Array{Float64, 3}
    D̄x::Array{Float64, 3}
    D̄t::Array{Float64, 3}
    Ē_for_interior_interfaces::Dict{FaceInterface, Ematrices} # the E matrices for the interior interfaces

    # x_coord::Dict{Int, Array{Float64, 1}}()
    # y_coord::Dict{Int, Array{Float64, 1}}()
end

function MyGrid(grid::Grid; num_of_variables = 3)
    num_of_nodes = length(grid.xyz[1])
    num_of_cells = n_cells(grid)
    size_of_D = Int(num_of_nodes * num_of_variables)
    Im = I(num_of_variables)
    interfaces_aligning_shock = collect(find_interfaces_align_shock(grid))
    interior_interfaces = setdiff(grid.face_interfaces, interfaces_aligning_shock) # interfaces non-aligning with the shock

    H = zeros(num_of_cells, num_of_nodes, num_of_nodes)
    Hinv = zeros(num_of_cells, num_of_nodes, num_of_nodes)
    Dx = zeros(num_of_cells, num_of_nodes, num_of_nodes)
    Dt = zeros(num_of_cells, num_of_nodes, num_of_nodes)
    E_for_interior_interfaces = Dict{FaceInterface, Ematrices}()

    H̄ = zeros(num_of_cells, size_of_D, size_of_D)
    H̄inv = zeros(num_of_cells, size_of_D, size_of_D)
    D̂x = zeros(num_of_cells, size_of_D, size_of_D)
    D̂t = zeros(num_of_cells, size_of_D, size_of_D)
    Ē_for_interior_interfaces = Dict{FaceInterface, Ematrices}()

    Threads.@threads for cell_id in 1:num_of_cells

        Dx[cell_id, :, :] = grid.VOL[cell_id][1]
        Dt[cell_id, :, :] = grid.VOL[cell_id][2]


        D̂x[cell_id, :, :] = kron(grid.VOL[cell_id][1], Im)
        D̂t[cell_id, :, :] = kron(grid.VOL[cell_id][2], Im)
        curr_ref = grid.cells[cell_id].ref_data[]
        H_local = curr_ref.H * Diagonal(grid.geometric_terms.J_q[cell_id])
        H[cell_id, :, :] = H_local
        H̄[cell_id, :, :] = kron(H_local, Im)
        Hinv[cell_id, :, :] = inv(H_local)
        H̄inv[cell_id, :, :] = inv(H̄[cell_id, :, :])



    end

    for interface in union(interfaces_aligning_shock, interior_interfaces) 
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        refk = grid.cells[ck].ref_data[]
        refv = grid.cells[cv].ref_data[]
        Rγk, Rγv = Matrix(refk.R[lfγk]), Matrix(refv.R[lfγv])
        γk_id, γv_id = refk.R[lfγk].idxs, refv.R[lfγv].idxs

        maskγk, maskγv = @views refk.f_mask[lfγk], refv.f_mask[lfγv]
        normalγk, normalγv = @views grid.geometric_terms.N_f[ck][:, maskγk], grid.geometric_terms.N_f[cv][:, maskγv]
        Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
        Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]

        Exγk = Rγk' * refk.H_face * Diagonal(Nxγk) * Rγk
        Etγk = Rγk' * refk.H_face * Diagonal(Ntγk) * Rγk
        Exγv = Rγv' * refv.H_face * Diagonal(Nxγv) * Rγv
        Etγv = Rγv' * refv.H_face * Diagonal(Ntγv) * Rγv

        Rγkv = zeros(Float64, (length(γk_id), num_of_nodes))
        Rγkv[:, γv_id[Pv]] = I(length(γk_id))

        Rγvk = zeros(Float64, (length(γv_id), num_of_nodes))
        Rγvk[:, γk_id[Pk]] = I(length(γv_id))

        Exkv = Rγk' * refk.H_face * Diagonal(Nxγk) * Rγkv
        Etkv = Rγk' * refk.H_face * Diagonal(Ntγk) * Rγkv
        Exvk = Rγv' * refv.H_face * Diagonal(Nxγv) * Rγvk
        Etvk = Rγv' * refv.H_face * Diagonal(Ntγv) * Rγvk

        curr_Ematrics = Ematrices(Exγk, Exkv, Etγk, Etkv, Exγv, Exvk, Etγv, Etvk)
        curr_Ēmatrices = Ematrices(kron(Exγk, Im), kron(Exkv, Im), kron(Etγk, Im), kron(Etkv, Im), kron(Exγv, Im), kron(Exvk, Im), kron(Etγv, Im), kron(Etvk, Im))
        E_for_interior_interfaces[interface] = curr_Ematrics
        Ē_for_interior_interfaces[interface] = curr_Ēmatrices
    end



    return myGrid(interfaces_aligning_shock, interior_interfaces, H, Hinv, Dx, Dt, E_for_interior_interfaces, H̄, H̄inv, D̂x, D̂t, Ē_for_interior_interfaces)

end

function VOL_contribution!(du, newGrid, W, cell_id, One)

    # du[cell_id, :] += - 2 * (newGrid.D̄t[cell_id, :, :] .* two_point_state_flux_function(W[cell_id, :, :], W[cell_id, :, :]) + newGrid.D̄x[cell_id, :, :] .* two_point_flux_function(W[cell_id, :, :], W[cell_id, :, :])) * One
    du[cell_id, :] += - 2 * newGrid.H̄[cell_id,:,:] * (newGrid.D̄t[cell_id, :, :] .* two_point_state_flux_function(W[cell_id, :, :], W[cell_id, :, :]) + newGrid.D̄x[cell_id, :, :] .* two_point_flux_function(W[cell_id, :, :], W[cell_id, :, :])) * One

end


function RHS(U, p; WL = [3, 2, 10], WR = [3, -1, 5])
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
    num_of_nodes = length(grid.xyz[1])
    du = zeros(n_cells(grid), num_of_nodes * 3)
    W = evaluate_W(U)
    One = ones(length(grid.xyz[1]) * 3)

    Threads.@threads for cell_id in 1:n_cells(grid)
        VOL_contribution!(du, newGrid, W, cell_id, One)
    end

    Threads.@threads for interface in newGrid.interior_interfaces # for each interior interface( non -aligning with the shock)
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2

        # for cell k
        Fxkk = two_point_flux_function(W[ck, :, :], W[ck, :, :])
        Fxkv = two_point_flux_function(W[ck, :, :], W[cv, :, :])
        Ukk = two_point_state_flux_function(W[ck, :, :], W[ck, :, :])
        Ukv = two_point_state_flux_function(W[ck, :, :], W[cv, :, :])

        Fxvv = two_point_flux_function(W[cv, :, :], W[cv, :, :])
        Fxvk = two_point_flux_function(W[cv, :, :], W[ck, :, :])
        Uvv = two_point_state_flux_function(W[cv, :, :], W[cv, :, :])
        Uvk = two_point_state_flux_function(W[cv, :, :], W[ck, :, :])

        Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
        Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv
        
        Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
        Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk

        # du[ck, :] += newGrid.H̄inv[ck, :, :] * (Ēxγk .* Fxkk - Ēxkv .* Fxkv + Ētγk .* Ukk - Ētkv .* Ukv) * One
        # du[cv, :] += newGrid.H̄inv[cv, :, :] * (Ēxγv .* Fxvv - Ēxvk .* Fxvk + Ētγv .* Uvv - Ētvk .* Uvk) * One
        du[ck, :] += (Ēxγk .* Fxkk - Ēxkv .* Fxkv + Ētγk .* Ukk - Ētkv .* Ukv) * One
        du[cv, :] += (Ēxγv .* Fxvv - Ēxvk .* Fxvk + Ētγv .* Uvv - Ētvk .* Uvk) * One
    end

    for interface in newGrid.interfaces_aligning_shock

        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2

        # for cell k
        Fxkk = two_point_flux_function(W[ck, :, :], W[ck, :, :])
        Ukk = two_point_state_flux_function(W[ck, :, :], W[ck, :, :])
        uk = vec(U[ck,:,:]')
        Fk = evaluate_Fx(W[ck, :, :])

        Fxvv = two_point_flux_function(W[cv, :, :], W[cv, :, :])
        Uvv = two_point_state_flux_function(W[cv, :, :], W[cv, :, :])
        uv = vec(U[cv,:,:]')
        Fv = evaluate_Fx(W[cv, :, :])

        Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
        Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv
        
        Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
        Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk

        # du[ck, :] += newGrid.H̄inv[ck, :, :] * (Ētγk .* Ukk * One - 0.5 * Ētγk * uk - 0.5 * Ētkv * uv + Ēxγk .* Fxkk * One - 0.5 * Ēxγk * Fk - 0.5 * Ēxkv * Fv)
        # du[cv, :] += newGrid.H̄inv[cv, :, :] * (Ētγv .* Uvv * One - 0.5 * Ētγv * uv - 0.5 * Ētvk * uk + Ēxγv .* Fxvv * One - 0.5 * Ēxγv * Fv - 0.5 * Ēxvk * Fk)
        du[ck, :] += (Ētγk .* Ukk * One - 0.5 * Ētγk * uk - 0.5 * Ētkv * uv + Ēxγk .* Fxkk * One - 0.5 * Ēxγk * Fk - 0.5 * Ēxkv * Fv)
        du[cv, :] += (Ētγv .* Uvv * One - 0.5 * Ētγv * uv - 0.5 * Ētvk * uk + Ēxγv .* Fxvv * One - 0.5 * Ēxγv * Fv - 0.5 * Ēxvk * Fk)
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
        WBC = zeros(Float64, size(Wk))
        WBC[:, 1] .= WL[1]
        WBC[:, 2] .= WL[2]
        WBC[:, 3] .= WL[3]
        Fxkk = two_point_flux_function(Wk, Wk)
        FxkBC = two_point_flux_function(Wk, WBC)

        # du[cell_id, :] += newGrid.H̄inv[cell_id, :, :] * (Ēxγ .* Fxkk - Ēxγ .* FxkBC) * One
        du[cell_id, :] += (Ēxγ .* Fxkk - Ēxγ .* FxkBC) * One
    end

    for faceindex in collect(get_face_set(grid, "RIGHT_INFLOW"))
        cell_id, face_id = faceindex
        ref = grid.cells[cell_id].ref_data[]
        Rγ = Matrix(ref.R[face_id])
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        Nxγ = normal[1, :]

        Ēxγ = kron(Rγ' * ref.H_face * Diagonal(Nxγ) * Rγ, I(3))

        Wk = W[cell_id, :, :]
        WBC = zeros(Float64, size(Wk))
        WBC[:, 1] .= WR[1]
        WBC[:, 2] .= WR[2]
        WBC[:, 3] .= WR[3]

        Fxkk = two_point_flux_function(Wk, Wk)
        FxkBC = two_point_flux_function(Wk, WBC)

        # du[cell_id, :] += newGrid.H̄inv[cell_id, :, :] * (Ēxγ .* Fxkk - Ēxγ .* FxkBC) * One
        du[cell_id, :] += (Ēxγ .* Fxkk - Ēxγ .* FxkBC) * One
    end
    result = zeros(size(U))
    Threads.@threads for cell_id in 1:n_cells(grid)
        result[cell_id, :, :] = reshape(du[cell_id,:], (3, num_of_nodes))'
    end
    return result
end



function set_IC(grid; WL = [3, 2, 10], WR = [3, -1, 5], γ = 1.4)

    IC = Dict()

    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_1")
        temp_u = zeros(length(grid.xyz_q[cell_id]), 3)
        temp_u[:, 1] .= WL[1]
        temp_u[:, 2] .= WL[1] * WL[2]
        temp_u[:, 3] .= WL[3] / (γ - 1) + 0.5  * WL[1] * WL[2]^2
        IC[(cell_id, face_id)] = temp_u
    end
    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_2")
        temp_u = zeros(length(grid.xyz_q[cell_id]), 3)
        temp_u[:, 1] .= WR[1]
        temp_u[:, 2] .= WR[1] * WR[2]
        temp_u[:, 3] .= WR[3] / (γ - 1) + 0.5  * WR[1] * WR[2]^2
        IC[(cell_id, face_id)] = temp_u
    end

    return IC


end

function one_wave_speed(ρL, uL, pL, pstar)
    aL = sqrt(γ * pL / ρL)

    ratio1 = (γ + 1)/(2*γ)
    p_ratio = pstar / pL
    ratio2 = (γ - 1)/(2*γ)

    return uL - aL * sqrt( ratio1 * p_ratio + ratio2)
end

function three_wave_speed(ρR, uR, pR, pstar)
    aR = sqrt(γ * pR / ρR)

    ratio1 = (γ + 1)/(2*γ)
    p_ratio = pstar / pR
    ratio2 = (γ - 1)/(2*γ)

    return uR + aR * sqrt( ratio1 * p_ratio + ratio2)
end

S1 = one_wave_speed(ρL, uL, pL, pstar)
S2 = ustar
S3 = three_wave_speed(ρR, uR, pR, pstar)

analytic_W = zeros(n_cells(grid), length(grid.xyz[1]), 3) # ρ = 3, u = 2, p = 10
for cell_id in 1:n_cells(grid)


    xyzk = grid.xyz[cell_id]
    xk = [x[1] for x in xyzk]
    yk = [x[2] for x in xyzk]
    # for all element on the left of S1
    if all(yk .< (1/S1) .* xk .+ 1e-5)
        analytic_W[cell_id, :,1] .= ρL
        analytic_W[cell_id, :,2] .= uL
        analytic_W[cell_id, :,3] .= pL
    end
    if all(yk .> (1/S1) .* xk .- 1e-5) && all(yk .> (1/S2) .* xk .- 1e-5)
        analytic_W[cell_id, :,1] .= ρstarL
        analytic_W[cell_id, :,2] .= ustar
        analytic_W[cell_id, :,3] .= pstar
    end
    if all(yk .< (1/S2) .* xk .+ 1e-5) && all(yk .> (1/S3) .* xk .- 1e-5)
        analytic_W[cell_id, :,1] .= ρstarR
        analytic_W[cell_id, :,2] .= ustar
        analytic_W[cell_id, :,3] .= pstar
    end
    if all(yk .< (1/S3) .* xk .+ 1e-5)
        analytic_W[cell_id, :,1] .= ρR
        analytic_W[cell_id, :,2] .= uR
        analytic_W[cell_id, :,3] .= pR
    end

end

analytic_U = zeros(size(analytic_W))
analytic_U[:, :, 1] = deepcopy(analytic_W[:, :, 1])
analytic_U[:, :, 2] = analytic_W[:, :, 2] .* analytic_W[:, :, 1]
analytic_U[:, :, 3] = analytic_W[:, :, 3] ./ 0.4 + 0.5 * analytic_W[:, :, 1] .* analytic_W[:, :, 2].^2

# plot_one_variable(grid, analytic_W; xlimit = [-0.51, 0.51], ylimit = [-0.01, 0.151], size = (2400, 800), variable = 3)


function space_time_RK4(U, pseudo_dt, p; diss = false, test = false)
    c1, c2, c3, c4 = 0.0, 0.40128709, 0.56449983, 0.87678807
    b1, b2, b3, b4 = 0.20334721, 0.19932974, 0.28339585, 0.31392720
    a21 = 0.40128709
    a31, a32 = 0.28224991, 0.28224991
    a41, a42, a43 = 0.25972925, 0.25479937, 0.36225945


    k1 = RHS(U, p)
    k2 = RHS(U + a21 * pseudo_dt * k1, p)
    k3 = RHS(U + a31 * pseudo_dt * k1 + a32 * pseudo_dt * k2, p)
    k4 = RHS(U + a41 * pseudo_dt * k1 + a42 * pseudo_dt * k2 + a43 * pseudo_dt * k3, p)

    result = U + pseudo_dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4)
    if any(x -> x<0 , result[:,:,1]) || any(x -> x<0 , result[:,:,3])
        print("Negative value in space_time_RK4")
        return abs.(result)
    end

    return result
end


function space_time_solve(U0::Array{Float64, 3}, p::Any; pseudo_dt = 0.0001, num_of_pseudo_time_step = 100, num_of_variable = 3)

    num_of_cells = n_cells(p[1])
    num_of_nodes = Int(length(p[1].xyz_q[1]))
    # all_U = ones(Float64, (num_of_pseudo_time_step+1, num_of_cells, num_of_nodes, num_of_variable))
    # all_U[1, :, :, :] = pseudo_initial_condtion(p[1])
    all_U = Vector{Array{Float64, 3}}()
    push!(all_U, U0)
    diffs = zeros(num_of_pseudo_time_step)

    for pseudo_step in 1:num_of_pseudo_time_step
        curr_U = deepcopy(all_U[pseudo_step])
        next_U = space_time_RK4(curr_U, pseudo_dt, p)
        
        max_diff = maximum(abs.(next_U - curr_U))
        println("Pseudo time step: $pseudo_step, Maximum difference: ", max_diff, "\n")
        diffs[pseudo_step] = max_diff
        push!(all_U, next_U)
        push!(diffs, max_diff)

        if max_diff > 100 || isnan(max_diff)
            print("\n Unstable at pseudo time step: $pseudo_step, pseudo_dt = $pseudo_dt \n")
            return all_U, diffs
        elseif max_diff < 1e-13
            # u = next_u
            # break
            print("\n Congratulation! Solution converged, pseudo_dt = $pseudo_dt \n")
            return all_U, diffs
        end

    end
    print("\n Solution not converged, pseudo_dt = $pseudo_dt \n")
    return all_U, diffs
end

function observe_variable_iteration(grid, all_U, zlimit, jump; show_true_sol = true, variable = 1)
    """
    Draw a gif with x, y coordinates from grid, taking `all_U[i,:,:,1]` for plotting `ρ`.
    """
    num_of_cells = n_cells(grid)
    num_of_nodes = length(grid.xyz_q[1])
    num_of_pseudo_time_step = size(all_U)[1]
    # Pre-allocate x and y arrays
    x = Matrix{Float64}(undef, num_of_cells, num_of_nodes)
    y = Matrix{Float64}(undef, num_of_cells, num_of_nodes)
    all_vari = zeros(Float64,length(all_U[1:jump:end]), num_of_cells, num_of_nodes)
    if variable == 1
        println("Plotting ρ")
    elseif variable == 2
        println("Plotting u")
    elseif variable == 3
        println("Plotting p")
    end

    @inbounds for cell_id in 1:num_of_cells
        for node_id in 1:num_of_nodes
            x[cell_id, node_id] = grid.xyz_q[cell_id][node_id][1]
            y[cell_id, node_id] = grid.xyz_q[cell_id][node_id][2]
        end
    end

    @inbounds for i in 1:size(all_vari, 1)
        if variable == 1
            all_vari[i, :, :] = all_U[(i-1)*jump+1][:, :, variable]
        elseif variable == 2
            all_vari[i, :, :] = all_U[(i-1)*jump+1][:, :, variable] ./ all_U[(i-1)*jump+1][:, :, 1]
        else
            # p = (E - 0.5 * ρ * u^2) * (γ - 1)
            all_u = all_U[(i-1)*jump+1][:, :, 2] ./ all_U[(i-1)*jump+1][:, :, 1]
            all_vari[i, :, :] = (all_U[(i-1)*jump+1][:, :, 3] .- 0.5 * all_U[(i-1)*jump+1][:, :, 1] .* (all_u.^2)) * (γ - 1)
        end
    end

    @inbounds anim = @animate for i in ProgressBar(1:size(all_vari, 1))
        p = scatter3d()

        for cell_id in 1:num_of_cells
            # Plot numerical and analytical solutions for this cell
            z1 = all_vari[i, cell_id, :]
            scatter3d!(p, x[cell_id, :], y[cell_id, :], z1, markersize=1, label="", zlims=zlimit, color=:blue)
        end

        title!("Pseudo time step: $i")
    end

    return anim
end

newGrid = MyGrid(grid; num_of_variables = 3)
IC = set_IC(grid; WL = [3, 2, 10], WR = [3, -1, 5], γ = 1.4)
para = grid, newGrid, IC
U0 =  deepcopy(analytic_U)

all_U, diffs = space_time_solve(U0, para, pseudo_dt = 5, num_of_pseudo_time_step = 3000)
gif(observe_variable_iteration(grid, all_U[600:end], [-2.,15.], 1, variable = 1), joinpath(@__DIR__, "numerical_test_discts.gif"), fps=5)

plot_one_variable(grid, all_U[end-2]; xlimit = [-0.51, 0.51], ylimit = [-0.01, 0.151], size = (2400, 800), variable = 3)


rhs = RHS(analytic_U, para; WL = [3, 2, 10], WR = [3, -1, 5])

plot_one_variable(grid, rhs; xlimit = [-0.51, 0.51], ylimit = [-0.01, 0.151], size = (2400, 800), variable = 3)





newGrid = MyGrid(grid; num_of_variables = 3)


u1 = [x1[i]^2 + y1[i]^2 for i in 1:length(x1)]
grid.VOL[1][1] * u1 - [2 * x1[i] for i in 1:length(x1)]

W =  zeros(n_cells(grid), length(grid.xyz[1]), 3) # ρ = 3, u = 2, p = 10
ρ(x, y) = x + y + 10
u(x, y) = x
p(x, y) = 10

using Test
for cell_id in 1:n_cells(grid)
    xyz1 = grid.xyz[cell_id]
    x1 = [x[1] for x in xyz1]
    y1 = [x[2] for x in xyz1]
    W[cell_id, :, 1] .= ρ.(x1, y1)
    W[cell_id, :, 2] .= u.(x1, y1)
    W[cell_id, :, 3] .= p.(x1, y1)
end
U = zeros(size(W))
U[:, :, 1] = deepcopy(W[:, :, 1])
U[:, :, 2] = W[:, :, 2] .* W[:, :, 1]
U[:, :, 3] = W[:, :, 3] ./ 0.4 + 0.5 * W[:, :, 1] .* W[:, :, 2].^2

U[1, :, :]

W_test = evaluate_W(U)
maximum(abs.(W_test - W))

cell_id = 1
xyz1 = grid.xyz[cell_id]
x1 = [x[1] for x in xyz1]
y1 = [x[2] for x in xyz1]
test_U11 = two_point_state_flux_function(W[1,:,:], W[1,:,:])
dUdt = 2 * newGrid.D̄t[1,:,:] .* test_U11 * ones(30)
dUdt[1:3:end] .- [1 for i in 1:length(dUdt[1:3:end])]
dUdt[2:3:end] .- [x1[i] for i in 1:length(dUdt[1:3:end])]
dUdt[3:3:end] .- [0.5 * x1[i]^2 for i in 1:length(dUdt[1:3:end])]



for cell_id in 1:n_cells(grid)
    xyz1 = grid.xyz[cell_id]
    x1 = [x[1] for x in xyz1]
    y1 = [x[2] for x in xyz1]
    test_Ukk = two_point_state_flux_function(W[cell_id,:,:], W[cell_id,:,:])
    dUdt = 2 * newGrid.D̄t[cell_id,:,:] .* test_Ukk * ones(30)

    @test maximum(abs.(dUdt[1:3:end] .- [1 for i in 1:length(dUdt[1:3:end])])) <= 1e-4
    @test maximum(abs.(dUdt[2:3:end] .- [x1[i] for i in 1:length(dUdt[1:3:end])])) <= 1e-4
    @test maximum(abs.(dUdt[3:3:end] .- [0.5 * x1[i]^2 for i in 1:length(dUdt[1:3:end])])) <= 1e-4


end


W =  zeros(n_cells(grid), length(grid.xyz[1]), 3) # ρ = 3, u = 2, p = 10
ρ(x, t) = x^2
u(x, t) = t
p(x, y) = 10

for cell_id in 1:n_cells(grid)
    xyz1 = grid.xyz[cell_id]
    x1 = [x[1] for x in xyz1]
    y1 = [x[2] for x in xyz1]
    W[cell_id, :, 1] .= ρ.(x1, y1)
    W[cell_id, :, 2] .= u.(x1, y1)
    W[cell_id, :, 3] .= p.(x1, y1)
end

for cell_id in 1:n_cells(grid)
    xyzk = grid.xyz[cell_id]
    xk = [x[1] for x in xyzk]
    yk = [x[2] for x in xyzk]
    test_Fkk = two_point_flux_function(W[cell_id,:,:], W[cell_id,:,:])
    dFdx = 2 * newGrid.D̄x[cell_id,:,:] .* test_Fkk * ones(30)

    # print(maximum(abs.(diag(test_Fkk)[1:3:end] .- ρ.(xk, yk) .* u.(xk, yk))))
    # @test maximum(abs.(dFdx[1:3:end] .- [yk[i] for i in 1:length(yk)])) <= 1e-4
    # @test maximum(abs.(dFdx[2:3:end] .- [yk[i]^2 for i in 1:length(yk)])) <= 1e-4
    # @test maximum(abs.(dFdx[3:3:end] .- [0.5 * yk[i]^3 for i in 1:length(yk)])) <= 1e-4

    # print(maximum(abs.(dFdx[1:3:end] .- [2 * xk[i] * yk[i] for i in 1:length(yk)])))
    # print(maximum(abs.(dFdx[2:3:end] .- [2 * xk[i] * yk[i]^2 for i in 1:length(yk)])))
    print(maximum(abs.(dFdx[3:3:end] .- [xk[i] * yk[i]^3 for i in 1:length(yk)])))
    print("\n")


end


# polynomial exactness test



interface = interior_interfaces[1]

ck, lfγk = interface.face_1
cv, lfγv = interface.face_2
Pk, Pv = interface.P1, interface.P2
refk = grid.cells[ck].ref_data[]
refv = grid.cells[cv].ref_data[]
Rγk, Rγv = Matrix(refk.R[lfγk]), Matrix(refv.R[lfγv])
γk_id, γv_id = refk.R[lfγk].idxs, refv.R[lfγv].idxs

xyzk = grid.xyz[10]
xyzv = grid.xyz[17]
xk = [x[1] for x in xyz10]
yk = [x[2] for x in xyz10]
xv = [x[1] for x in xyz17]
yv = [x[2] for x in xyz17]

zk = [xk[i]^2 + yk[i]^2 for i in 1:length(xk)]
zv = [xv[i]^2 + yv[i]^2 for i in 1:length(xv)]

Rγk * zk
Rγv * zv

Rγkv = zeros(Float64, (length(refk.R[lfγk].idxs), num_of_nodes))
Rγkv[:,(refk.R[lfγv].idxs)[Pv]] = I(length(refk.R[lfγk].idxs))

Rγkv * zv

for interface in interior_interfaces
    ck, lfγk = interface.face_1
    cv, lfγv = interface.face_2
    Pk, Pv = interface.P1, interface.P2
    refk = grid.cells[ck].ref_data[]
    refv = grid.cells[cv].ref_data[]
    Rγk, Rγv = Matrix(refk.R[lfγk]), Matrix(refv.R[lfγv])
    γk_id, γv_id = refk.R[lfγk].idxs, refv.R[lfγv].idxs

    xyzk = grid.xyz[ck]
    xyzv = grid.xyz[cv]
    xk = [x[1] for x in xyzk]
    yk = [x[2] for x in xyzk]
    xv = [x[1] for x in xyzv]
    yv = [x[2] for x in xyzv]
    zk = [xk[i]^2 + yk[i]^2 for i in 1:length(xk)]
    zv = [xv[i]^2 + yv[i]^2 for i in 1:length(xv)]
    
    Rγkv = zeros(Float64, (length(γk_id), num_of_nodes))
    Rγkv[:,γv_id[Pv]] = I(length(γk_id))
    @test Rγk * zk ≈ Rγkv * zv

    Rγvk = zeros(Float64, (length(γv_id), num_of_nodes))
    Rγvk[:,γk_id[Pk]] = I(length(γv_id))
    @test Rγv * zv ≈ Rγvk * zk

end

interface = newGrid.interior_interfaces[1]
ck, lfγk = interface.face_1
cv, lfγv = interface.face_2
Wk = W[ck, :, :]
Wv = W[cv, :, :]
newGrid.Ē_for_interior_interfaces[interface].Exγk * vec(Wk') - newGrid.Ē_for_interior_interfaces[interface].Exkv * vec(Wv')



Chandrashekar1D_f1(3., 3., 2., 2.)
Chandrashekar1D_f2(3., 3., 2., 2., 3/20, 3/20.)
Chandrashekar1D_f3(3., 3., 2., 2., 3/20, 3/20.)

Chandrashekar1D_f1(3., 2., 2., -1.)
Chandrashekar1D_f1(2., 3., -1., 2.)
using Test
for i in 1:100
    ρi = rand()
    ρj = rand()
    ui = rand()
    uj = rand()
    @test Chandrashekar1D_f1(ρi, ρj, ui, uj) ≈ Chandrashekar1D_f1(ρj, ρi, uj, ui)
end

Chandrashekar1D_f2(3., 2., 2., -1., 3/20, 2/10.)
for i in 1:100
    ρi = rand()
    ρj = rand()
    ui = rand()
    uj = rand()
    βi = rand()
    βj = rand()
    @test Chandrashekar1D_f2(ρi, ρj, ui, uj, βi, βj) ≈ Chandrashekar1D_f2(ρj, ρi, uj, ui, βj, βi)
end

Chandrashekar1D_f3(3., 2., 2., -1., 3/20, 2/10.)

for i in 1:100
    ρi = rand()
    ρj = rand()
    ui = rand()
    uj = rand()
    βi = rand()
    βj = rand()
    @test Chandrashekar1D_f3(ρi, ρj, ui, uj, βi, βj) ≈ Chandrashekar1D_f3(ρj, ρi, uj, ui, βj, βi)
end

U = zeros((1,1,3)) # ρ = 3, u = 2, p = 10
U[:,:,1] .= 3.
U[:,:,2] .= 6.
U[:,:,3] .= 10/0.4 + 0.5 * 3 * 2^2

evaluate_W(U)
evaluate_Fx(U)

Wk = zeros((2,3)) # ρ = 3, u = 2, p = 10
Wk[:,1] .= 3.
Wk[:,2] .= 2.
Wk[:,3] .= 10.
Wv = zeros((2,3)) # ρ = 2, u = -1, p = 5
Wv[:,1] .= 2.
Wv[:,2] .= -1.
Wv[:,3] .= 5.
two_point_flux(Wk, Wv)

Wk[2,:,:] .= Wk[1,:,:] * 2
Wv[2,:,:] .= Wv[1,:,:] * 2

two_point_state_flux_function(Wk, Wv)
two_point_state_flux_function(Wk, Wk)
two_point_state_flux_function(Wv, Wv)

Uk = zeros((1,2,3)) # ρ = 3, u = 2, p = 10
Uk[1, :,1] .= 3.
Uk[1, :,2] .= 6.
Uk[1, :,3] .= 10/0.4 + 0.5 * 3 * 2^2
Uk[1,2,:] .= 6., 24., 20/0.4 + 0.5 * 6 * 4^2











plot_cell(grid, 1)
xyzk = grid.xyz[41]
xk = [x[1] for x in xyzk]
yk = [x[2] for x in xyzk]

all(yk .> S2 .* xk .- 1e-5) 
all(yk .< S3 .* xk .+ 1e-5)

all(yk .> S1 .* xk .- 1e-3)
all(yk .< S2 .* xk .+ 1e-3)


function max_eigenvalues_on_face(ρkγ, ukγ, pkγ, ρvγ, uvγ, pvγ)
    """
    inputs are variables on the face, not the element
    of size num_of_nodes * 1
    output the maximum of eigenvalues of the flux Jacobian matrix on the face
    """
    max_eigen = 0
    for i in eachindex(ρkγ)
        # a =  √γp/ρ
        aki = (γ * pkγ[i] / ρkγ[i])^(1/2)
        λ1ki = abs(ukγ[i] - aki)
        λ2ki = abs(ukγ[i])
        λ3ki = abs(ukγ[i] + aki)

        avi = (γ * pvγ[i] / ρvγ[i])^(1/2)
        λ1vi = abs(uvγ[i] - avi)
        λ2vi = abs(uvγ[i])
        λ3vi = abs(uvγ[i] + avi)

        max_eigen = maximum([max_eigen, λ1ki, λ2ki, λ3ki, λ1vi, λ2vi, λ3vi])
        # print("max_eigen: ", max_eigen, "\n")
    end
    return max_eigen

end



test_W
evaluate_entropy_variables(test_W)
test_W2 = zeros((3, 5, 3))
test_W2[:, :, 1] .= 3.
test_W2[:, :, 2] .= 2.
test_W2[:, :, 3] .= 10.
evaluate_entropy_variables(test_W2)


(1.4 - log(10 / 3^1.4)) / 0.4 - 3 * 4 /20

W = evaluate_W(U0)
for cell_id in 1:n_cells(grid)
    xy = grid.xyz[cell_id]
    x = [x[1] for x in xy]
    y = [x[2] for x in xy]
    W[cell_id, :, 1] .= x .* y .+ 1
    W[cell_id, :, 2] .= x.^2 .* y.^2 .+ 2
    W[cell_id, :, 3] .= sin.(x) .+ cos.(y) .+ 3
end

ck, _ = interface.face_1
cv, _ = interface.face_2
Pk, Pv = interface.P1, interface.P2

# Pk_W, Pv_W = permutation_for_W(Pk), permutation_for_W(Pv)

Rγk, Rγv = newGrid.Rγ[interface.face_1], newGrid.Rγ[interface.face_2]

# for cell k
Fxkk = two_point_flux_function(W[ck, :, :], W[ck, :, :])
Fxkv = two_point_flux_function(W[ck, :, :], W[cv, :, :])
Ukk = two_point_state_flux_function(W[ck, :, :], W[ck, :, :])
Ukv = two_point_state_flux_function(W[ck, :, :], W[cv, :, :])
uk = vec(U[ck,:,:]')
Fk = evaluate_Fx(W[ck, :, :])

ρkγ, ρvγ = Rγk * W[ck, :, 1], Rγv * W[cv, :, 1]
ukγ, uvγ = Rγk * W[ck, :, 2], Rγv * W[cv, :, 2]
pkγ, pvγ = Rγk * W[ck, :, 3], Rγv * W[cv, :, 3]

# for cell v

Fxvv = two_point_flux_function(W[cv, :, :], W[cv, :, :])
Fxvk = two_point_flux_function(W[cv, :, :], W[ck, :, :])
Uvv = two_point_state_flux_function(W[cv, :, :], W[cv, :, :])
Uvk = two_point_state_flux_function(W[cv, :, :], W[ck, :, :])

uv = vec(U[cv,:,:]')
Fv = evaluate_Fx(W[cv, :, :])

Ēxγk, Ētγk = newGrid.Ē_for_interior_interfaces[interface].Exγk, newGrid.Ē_for_interior_interfaces[interface].Etγk
Ēxkv, Ētkv = newGrid.Ē_for_interior_interfaces[interface].Exkv, newGrid.Ē_for_interior_interfaces[interface].Etkv

Ēxγv, Ētγv = newGrid.Ē_for_interior_interfaces[interface].Exγv, newGrid.Ē_for_interior_interfaces[interface].Etγv
Ēxvk, Ētvk = newGrid.Ē_for_interior_interfaces[interface].Exvk, newGrid.Ē_for_interior_interfaces[interface].Etvk

function dissipation_term(W, V, newGrid, interface; num_of_variables = 3)
    num_of_nodes = length(ρkγ)

    Rγk = newGrid.Rγ[interface.face_1]
    Rγv = newGrid.Rγ[interface.face_2]

    Pk, Pv = interface.P1, interface.P2

    ρk_face, ρk_adj = Rγk * W[ck, :, 1], (Rγvk * W[ck, :, 1])[Pv]
    uk_face, uk_adj = Rγk * W[ck, :, 2], (Rγvk * W[ck, :, 2])[Pv]
    pk_face, pk_adj = Rγk * W[ck, :, 3], (Rγvk * W[ck, :, 3])[Pv]

    ρv_face, ρv_adj = Rγv * W[cv, :, 1], (Rγkv * W[cv, :, 1])[Pk]
    uv_face, uv_adj = Rγv * W[cv, :, 2], (Rγkv * W[cv, :, 2])[Pk]
    pv_face, pv_adj = Rγv * W[cv, :, 3], (Rγkv * W[cv, :, 3])[Pk]



    α_max_k = zeros(Float64, num_of_nodes * num_of_variables)
    α_max_k = zeros(Float64, num_of_nodes * num_of_variables)

    # compute the max abs eigenvalues on each node on the face
    # then node by node will give me a long vector, something like α_max
    # then α_max * (Wkγ - Wvγ) is the dissipation term for element k, 
    # dont forget to multiply by transpose of the permutation matrix to extrapolate back
    # i.e. Rγk' * (α_max * (Wkγ - Wvγ))
    # of course we need more test

    for i in eachindex(ρkγ)
        # a =  √γp/ρ
        # for k
        ak_face_i = (γ * pk_face[i] / ρk_face[i])^(1/2)
        λ1k_face_i = abs(uk_face[i] - ak_face_i)
        λ2k_face_i = abs(uk_face[i])
        λ3k_face_i = abs(uk_face[i] + ak_face_i)
        ak_adj_i = (γ * pk_adj[i] / ρk_adj[i])^(1/2)
        λ1k_adj_i = abs(uk_adj[i] - ak_adj_i)
        λ2k_adj_i = abs(uk_adj[i])
        λ3k_adj_i = abs(uk_adj[i] + ak_adj_i)
        α_max_k[(i-1)*num_of_variables+1:i*num_of_variables] .= max(λ1k_face_i, λ2k_face_i, λ3k_face_i, λ1k_adj_i, λ2k_adj_i, λ3k_adj_i)

        # for v
        av_face_i = (γ * pv_face[i] / ρv_face[i])^(1/2)
        λ1v_face_i = abs(uv_face[i] - av_face_i)
        λ2v_face_i = abs(uv_face[i])
        λ3v_face_i = abs(uv_face[i] + av_face_i)
        av_adj_i = (γ * pv_adj[i] / ρv_adj[i])^(1/2)
        λ1v_adj_i = abs(uv_adj[i] - av_adj_i)
        λ2v_adj_i = abs(uv_adj[i])
        λ3v_adj_i = abs(uv_adj[i] + av_adj_i)
        α_max_v[(i-1)*num_of_variables+1:i*num_of_variables] .= max(λ1v_face_i, λ2v_face_i, λ3v_face_i, λ1v_adj_i, λ2v_adj_i, λ3v_adj_i)
    end

    duk_dissipation = (newGrid.Ē_for_interior_interfaces[interface].Exγk * V[ck,:] - newGrid.Ē_for_interior_interfaces[interface].Exkv * V[cv,:]) .* newGrid.R̄γ[interface.face_1]' * α_max_k
    duv_dissipation = (newGrid.Ē_for_interior_interfaces[interface].Exγv * V[cv,:] - newGrid.Ē_for_interior_interfaces[interface].Exvk * V[ck,:]) .* newGrid.R̄γ[interface.face_2]' * α_max_v

    return duk_dissipation, duv_dissipation

end

maximum(abs.(newGrid.Ē_for_interior_interfaces[interface].Exγk * V[ck,:] - newGrid.Ē_for_interior_interfaces[interface].Exkv * V[cv,:]))
newGrid.Ē_for_interior_interfaces[interface].Exγk * V[ck,:]

α_max = zeros(Float64, 3 * 3)
Rγk = newGrid.Rγ[interface.face_1]
Rγv = newGrid.Rγ[interface.face_2]
ρk_face, ρk_adj = Rγk * W[ck, :, 1], (Rγv * W[cv, :, 1])[Pv]
uk_face, uk_adj = Rγk * W[ck, :, 2], (Rγv * W[cv, :, 2])[Pv]
pk_face, pk_adj = Rγk * W[ck, :, 3], (Rγv * W[cv, :, 3])[Pv]

ρv_face, ρv_adj = Rγv * W[cv, :, 1], (Rγk * W[ck, :, 1])[Pk]
uv_face, uv_adj = Rγv * W[cv, :, 2], (Rγk * W[ck, :, 2])[Pk]
pv_face, pv_adj = Rγv * W[cv, :, 3], (Rγk * W[ck, :, 3])[Pk]

for i in eachindex(ρkγ)
    # a =  √γp/ρ
    aki = (γ * pkγ[i] / ρkγ[i])^(1/2)
    λ1ki = abs(ukγ[i] - aki)
    λ2ki = abs(ukγ[i])
    λ3ki = abs(ukγ[i] + aki)

    avi = (γ * pvγ[i] / ρvγ[i])^(1/2)
    λ1vi = abs(uvγ[i] - avi)
    λ2vi = abs(uvγ[i])
    λ3vi = abs(uvγ[i] + avi)

    α_max[(i-1)*3+1:i*3] .= max(λ1ki, λ2ki, λ3ki, λ1vi, λ2vi, λ3vi)
    
end
newGrid.Ē_for_interior_interfaces[interface].Exγk * V[ck,:]  .* newGrid.R̄γ[interface.face_1]' * α_max
newGrid.R̄γ[interface.face_1]' * α_max