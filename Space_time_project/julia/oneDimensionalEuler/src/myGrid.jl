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
    interfaces_aligning_contact_wave::Vector{FaceInterface} # the interfaces that are aligning with the contact wave
    interior_interfaces::Vector{FaceInterface} # the interior interfaces that are not aligning with the shock

    H::Array{Float64, 3}
    Hinv::Array{Float64, 3}
    Dx::Array{Float64, 3}
    Dt::Array{Float64, 3}
    Rγ::Dict{FaceIndex, Matrix{Float64}} # the R matrices for the face interfaces
    Hγ::Dict{FaceIndex, Matrix{Float64}} # the H matrices for the face interfaces
    Nxγ::Dict{FaceIndex, Matrix{Float64}} # the Nx matrices for the face interfaces
    Ntγ::Dict{FaceIndex, Matrix{Float64}} # the Nt matrices for the face interfaces
    E_for_interior_interfaces::Dict{FaceInterface, Ematrices} # the E matrices for the interior interfaces

    H̄::Array{Float64, 3}
    H̄inv::Array{Float64, 3}
    D̄x::Array{Float64, 3}
    D̄t::Array{Float64, 3}
    R̄γ::Dict{FaceIndex, Matrix{Float64}} # the R matrices for the face interfaces
    H̄γ::Dict{FaceIndex, Matrix{Float64}} # the H matrices for the face interfaces
    N̄xγ::Dict{FaceIndex, Matrix{Float64}} # the Nx matrices for the face interfaces
    N̄tγ::Dict{FaceIndex, Matrix{Float64}} # the Nt matrices for the face interfaces
    Ē_for_interior_interfaces::Dict{FaceInterface, Ematrices} # the E matrices for the interior interfaces

    # x_coord::Dict{Int, Array{Float64, 1}}()
    # y_coord::Dict{Int, Array{Float64, 1}}()
end

# function find_interfaces_align_shock(grid)
#     interfaces_set = Set{FaceInterface}()
#     for (cell_id, face_id) in union(get_face_set(grid, "ONE_WAVE"), get_face_set(grid, "THREE_WAVE"), get_face_set(grid, "CONTACT_WAVE"))
#         neighbor_id, neighbor_face_id = Tuple(findall(x -> x == FaceIndex((cell_id, face_id)), grid.topology.face_face_neighbours)[1])
#         interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((neighbor_id, neighbor_face_id)))
#         push!(interfaces_set, interface)
#     end
#     return interfaces_set
# end

function find_interfaces_align_shock(grid)
    interfaces_set = Set{FaceInterface}()
    for (cell_id, face_id) in union(get_face_set(grid, "ONE_WAVE"), get_face_set(grid, "THREE_WAVE"))
        neighbor_id, neighbor_face_id = Tuple(findall(x -> x == FaceIndex((cell_id, face_id)), grid.topology.face_face_neighbours)[1])
        interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((neighbor_id, neighbor_face_id)))
        push!(interfaces_set, interface)
    end
    return interfaces_set
end

function find_interfaces_align_contact_wave(grid)
    interfaces_set = Set{FaceInterface}()
    for (cell_id, face_id) in get_face_set(grid, "CONTACT_WAVE")
        neighbor_id, neighbor_face_id = Tuple(findall(x -> x == FaceIndex((cell_id, face_id)), grid.topology.face_face_neighbours)[1])
        interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((neighbor_id, neighbor_face_id)))
        push!(interfaces_set, interface)
    end
    return interfaces_set

end


function permutation_for_W(Pk::Vector{Int64}; num_of_variables = 3)

    n = length(Pk)
    result = zeros(Int, 3*n)
    
    for (i, val) in enumerate(Pk)
        base = (val-1) * 3
        result[(i-1)*3 + 1 : i*3] = [base + 1, base + 2, base + 3]
    end
    
    return result
end 

function MyGrid(grid::Grid; num_of_variables = 3)
    num_of_nodes = length(grid.xyz[1])
    num_of_cells = n_cells(grid)
    size_of_D = Int(num_of_nodes * num_of_variables)
    Im = I(num_of_variables)
    interfaces_aligning_shock = collect(find_interfaces_align_shock(grid))
    interfaces_aligning_contact_wave = collect(find_interfaces_align_contact_wave(grid))
    interior_interfaces = setdiff(grid.face_interfaces, interfaces_aligning_shock) # interfaces non-aligning with the shock

    H = zeros(num_of_cells, num_of_nodes, num_of_nodes)
    Hinv = zeros(num_of_cells, num_of_nodes, num_of_nodes)
    Dx = zeros(num_of_cells, num_of_nodes, num_of_nodes)
    Dt = zeros(num_of_cells, num_of_nodes, num_of_nodes)
    Rγ = Dict{FaceIndex, Matrix{Float64}}()
    Hγ = Dict{FaceIndex, Matrix{Float64}}()
    Nxγ = Dict{FaceIndex, Matrix{Float64}}()
    Ntγ = Dict{FaceIndex, Matrix{Float64}}()
    E_for_interior_interfaces = Dict{FaceInterface, Ematrices}()

    H̄ = zeros(num_of_cells, size_of_D, size_of_D)
    H̄inv = zeros(num_of_cells, size_of_D, size_of_D)
    D̂x = zeros(num_of_cells, size_of_D, size_of_D)
    D̂t = zeros(num_of_cells, size_of_D, size_of_D)
    R̄γ = Dict{FaceIndex, Matrix{Float64}}()
    H̄γ = Dict{FaceIndex, Matrix{Float64}}()
    N̄xγ = Dict{FaceIndex, Matrix{Float64}}()
    N̄tγ = Dict{FaceIndex, Matrix{Float64}}()
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

        Rγ[interface.face_1] = Rγk
        Rγ[interface.face_2] = Rγv
        R̄γ[interface.face_1] = kron(Rγk, I(3))
        R̄γ[interface.face_2] = kron(Rγv, I(3))

        Hγ[interface.face_1] = refk.H_face
        Hγ[interface.face_2] = refv.H_face
        H̄γ[interface.face_1] = kron(refk.H_face, Im)
        H̄γ[interface.face_2] = kron(refv.H_face, Im)

        maskγk, maskγv = @views refk.f_mask[lfγk], refv.f_mask[lfγv]
        normalγk, normalγv = @views grid.geometric_terms.N_f[ck][:, maskγk], grid.geometric_terms.N_f[cv][:, maskγv]
        Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
        Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]

        Nxγ[interface.face_1] = Diagonal(Nxγk)
        Nxγ[interface.face_2] = Diagonal(Nxγv)
        N̄xγ[interface.face_1] = kron(Diagonal(Nxγk), Im)
        N̄xγ[interface.face_2] = kron(Diagonal(Nxγv), Im)
        Ntγ[interface.face_1] = Diagonal(Ntγk)
        Ntγ[interface.face_2] = Diagonal(Ntγv)
        N̄tγ[interface.face_1] = kron(Diagonal(Ntγk), Im)
        N̄tγ[interface.face_2] = kron(Diagonal(Ntγv), Im)

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



    return myGrid(interfaces_aligning_shock, interfaces_aligning_contact_wave, interior_interfaces, 
            H, Hinv, Dx, Dt, Rγ, Hγ, Nxγ, Ntγ, E_for_interior_interfaces, 
            H̄, H̄inv, D̂x, D̂t, R̄γ, H̄γ, N̄xγ, N̄tγ, Ē_for_interior_interfaces)

end
