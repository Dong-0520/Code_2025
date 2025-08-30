struct Ematrices{T}

    # all E matrices needed for evaluating the interior interfaces contribution
    # Ex^γk, Ex^kv, Et^γk, Et^kv for ck contribution
    # Ex^γv, Ex^vk, Et^γv, Et^vk for cv contribution

    Exγk::Matrix{T}
    Exkv::Matrix{T}
    Etγk::Matrix{T}
    Etkv::Matrix{T}

    Exγv::Matrix{T}
    Exvk::Matrix{T}
    Etγv::Matrix{T}
    Etvk::Matrix{T}

end






# 这个struct 方便我们每次更新了 mesh坐标之后，能够更新 SBP operators
mutable struct SimpleGrid{dim, C <: AbstractCell, T <: Real}
    cells::Vector{C}
    ref::RefElemData
    xyz_gmsh::Vector{Coord{dim, T}}
    xyz_SBP::VectorOfArrays{Coord{dim, T}, 1}
    face_interfaces::Vector{FaceInterface} # 包含了所有的interior interfaces（ including shock)
    face_sets::Dict{String, Set{FaceIndex}}
    geometric_terms::GeometricTerms{T}
    # VOL::Vector{NTuple{dim, Matrix{T}}}

    xyz_gmsh_vec::Vector{T}              # 展平的坐标向量
    indices_moving_coords::Vector{Int64}       # 可动坐标的索引

    # interfaces_aligning_shock::Vector{FaceInterface} # the interfaces that are aligning with the shock
    # interfaces_aligning_contact_wave::Vector{FaceInterface} # the interfaces that are aligning with the contact wave
    # smooth_interior_interfaces::Vector{FaceInterface} # the interior interfaces that are not aligning with the shock and contact wave

    # H::Array{Float64, 3}
    # Hinv::Array{Float64, 3}
    # Dx::Array{Float64, 3}
    # Dt::Array{Float64, 3}
    # Rγ::Dict{FaceIndex, Matrix{Float64}} # the R matrices for the face interfaces
    # Hγ::Dict{FaceIndex, Matrix{Float64}} # the H matrices for the face interfaces
    # Nxγ::Dict{FaceIndex, Matrix{Float64}} # the Nx matrices for the face interfaces
    # Ntγ::Dict{FaceIndex, Matrix{Float64}} # the Nt matrices for the face interfaces
    # E_for_interior_interfaces::Dict{FaceInterface, Ematrices} # the E matrices for the interior interfaces
    # Eγ_for_bottom::Dict{FaceIndex, Matrix{Float64}} # the E matrices for the bottom faces
    # Eγ_for_boundary::Dict{FaceIndex, Matrix{Float64}} # the E matrices for the left and right boundary faces

    H̄::Array{T, 3}
    H̄inv::Array{T, 3}
    D̄x::Array{T, 3}
    D̄t::Array{T, 3}
    R̄γ::Dict{FaceIndex, Matrix{T}} # the R matrices for the face interfaces
    H̄γ::Dict{FaceIndex, Matrix{T}} # the H matrices for the face interfaces
    N̄xγ::Dict{FaceIndex, Matrix{T}} # the Nx matrices for the face interfaces
    N̄tγ::Dict{FaceIndex, Matrix{T}} # the Nt matrices for the face interfaces
    Ē_for_interior_interfaces::Dict{FaceInterface, Ematrices{T}} # the Ē matrices for the interior interfaces
    Ēγ_for_bottom::Dict{FaceIndex, Matrix{T}} # the Ē matrices for the bottom faces
    Ēγ_for_boundary::Dict{FaceIndex, Matrix{T}} # the Ē matrices for the left and right boundary faces

end

mutable struct SpaceTimeBuffer
    grid::Any                           # 网格
    S::Vector{Float64}                  # 参数 [S1_shock, S2_contact, S3_shock]
    initial_guess::Array{Float64, 1}    # 初始猜测, 包括所有的动的和不动的xyz_gms坐标，和 U0 作为最开始的解
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
    interfaces_aligning_shock::Vector{FaceInterface}
    interfaces_aligning_contact_wave::Vector{FaceInterface}
    smooth_interior_interfaces::Vector{FaceInterface}
    bottom_faceIndex::Vector{FaceIndex}
    boundary_faceIndex::Vector{FaceIndex}

    # 构造函数
    function SpaceTimeBuffer(grid::Grid, S::Vector{Float64}, oscillating_U::Array{Float64, 3})
        indices_moving_coords, coords_xyz_gmsh, t0, tF, xL, xR = get_meshinfo_for_buffer(grid)
        initial_guess = vcat(coords_xyz_gmsh, vec(oscillating_U))
        index_mesh = length(coords_xyz_gmsh)  # 网格坐标的索引
        ref = grid.cells[1].ref_data[]  # 获取参考元素数据
        interfaces_aligning_shock = collect(find_interfaces_align_shock(grid))
        interfaces_aligning_contact_wave = collect(find_interfaces_align_contact_wave(grid))
        smooth_interior_interfaces = collect(setdiff(Set(grid.face_interfaces), union(Set(interfaces_aligning_shock), Set(interfaces_aligning_contact_wave))))
        bottom_faceIndex = collect(union(grid.face_sets["BOTTOM_INFLOW_1"], grid.face_sets["BOTTOM_INFLOW_2"]))
        boundary_faceIndex = collect(union(grid.face_sets["LEFT_INFLOW"], grid.face_sets["RIGHT_INFLOW"]))
        new(grid, S,
        deepcopy(initial_guess),
        index_mesh,
        indices_moving_coords,  # 可移动坐标的索引
        xL, xR, t0, tF,
        zeros(Float64, length(initial_guess)),  # 最终解初始化为零
        Inf, Inf, Inf, false, Inf, ref,
        interfaces_aligning_shock, interfaces_aligning_contact_wave, smooth_interior_interfaces,
        bottom_faceIndex, boundary_faceIndex)  # 添加接口信息
    end

    function SpaceTimeBuffer(grid::SimpleGrid, S::Vector{Float64}, oscillating_U::Array{Float64, 3}, 
                            interfaces_aligning_shock::Vector{FaceInterface},
                            interfaces_aligning_contact_wave::Vector{FaceInterface},
                            smooth_interior_interfaces::Vector{FaceInterface})
        indices_moving_coords, coords_xyz_gmsh, t0, tF, xL, xR = get_meshinfo_for_buffer(grid)
        initial_guess = vcat(coords_xyz_gmsh, vec(oscillating_U))
        index_mesh = length(coords_xyz_gmsh)  # 网格坐标的索引
        ref = grid.cells[1].ref_data[]  # 获取参考元素数据
        bottom_faceIndex = collect(union(grid.face_sets["BOTTOM_INFLOW_1"], grid.face_sets["BOTTOM_INFLOW_2"]))
        boundary_faceIndex = collect(union(grid.face_sets["LEFT_INFLOW"], grid.face_sets["RIGHT_INFLOW"]))
        new(grid, S,
            deepcopy(initial_guess),
            index_mesh,
            indices_moving_coords,  # 可移动坐标的索引
            xL, xR, t0, tF,
            zeros(Float64, length(initial_guess)),  # 最终解初始化为零
            Inf, Inf, Inf, false, Inf, ref,
            interfaces_aligning_shock, interfaces_aligning_contact_wave, smooth_interior_interfaces,
            bottom_faceIndex, boundary_faceIndex)  # 添加接口信息
    end
end

import .SBPLite: get_face_set

function get_face_set(grid::SimpleGrid, face_type::String)
    face_set = Set{FaceIndex}()
    for (cell_id, face_id) in grid.face_sets[face_type]
        push!(face_set, FaceIndex((cell_id, face_id)))
    end
    return face_set
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

function find_interfaces_align_contact_wave(grid)
    interfaces_set = Set{FaceInterface}()
    for (cell_id, face_id) in get_face_set(grid, "CONTACT_WAVE")
        neighbor_id, neighbor_face_id = Tuple(findall(x -> x == FaceIndex((cell_id, face_id)), grid.topology.face_face_neighbours)[1])
        interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((neighbor_id, neighbor_face_id)))
        push!(interfaces_set, interface)
    end
    return interfaces_set

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

function find_boundaries(simple_grid::SimpleGrid)
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

function get_meshinfo_for_buffer(simple_grid::SimpleGrid; dim = 2, tol = 1e-8)
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
                            indices_moving_coords::Vector{Int64},
                            bottom_faceIndex::Vector{FaceIndex},
                            boundary_faceIndex::Vector{FaceIndex},
                            comp_coords::Vector{SVector{dim, T}} = 
                                        [SVector{dim, T}(-1.0, -1.0),
                                        SVector{dim, T}(1.0, -1.0),
                                        SVector{dim, T}(-1.0, 1.0)],
                            basis = MonomialType, 
                            gmsh_order = 1, num_of_variables = 3) where {dim, C, T<: Real} 

    # 重新计算映射
    n_cells = length(cells)
    num_of_nodes = size(ref.rst_q, 2)
    size_of_D = Int(num_of_nodes * num_of_variables)
    Im = I(num_of_variables)

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
    
    for i in 1:n_cells
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

    # H = zeros(n_cells, num_of_nodes, num_of_nodes)
    # Hinv = zeros(n_cells, num_of_nodes, num_of_nodes)
    # Dx = zeros(n_cells, num_of_nodes, num_of_nodes)
    # Dt = zeros(n_cells, num_of_nodes, num_of_nodes)
    # Rγ = Dict{FaceIndex, Matrix{Float64}}()
    # Hγ = Dict{FaceIndex, Matrix{Float64}}()
    # Nxγ = Dict{FaceIndex, Matrix{Float64}}()
    # Ntγ = Dict{FaceIndex, Matrix{Float64}}()
    # E_for_interior_interfaces = Dict{FaceInterface, Ematrices}()
    # Eγ_for_bottom = Dict{FaceIndex, Matrix{Float64}}()
    # Eγ_for_boundary = Dict{FaceIndex, Matrix{Float64}}()

    H̄ = zeros(T, n_cells, size_of_D, size_of_D)        # 改为 T
    H̄inv = zeros(T, n_cells, size_of_D, size_of_D)     # 改为 T  
    D̂x = zeros(T, n_cells, size_of_D, size_of_D)       # 改为 T
    D̂t = zeros(T, n_cells, size_of_D, size_of_D)       # 改为 T
    R̄γ = Dict{FaceIndex, Matrix{T}}()
    H̄γ = Dict{FaceIndex, Matrix{T}}()
    N̄xγ = Dict{FaceIndex, Matrix{T}}()
    N̄tγ = Dict{FaceIndex, Matrix{T}}()
    Ē_for_interior_interfaces = Dict{FaceInterface, Ematrices{T}}()
    Ēγ_for_bottom = Dict{FaceIndex, Matrix{T}}()
    Ēγ_for_boundary = Dict{FaceIndex, Matrix{T}}()

    for cell_id in 1:n_cells
        # Dx[cell_id, :, :] = VOL[cell_id][1]
        # Dt[cell_id, :, :] = VOL[cell_id][2]

        D̂x[cell_id, :, :] = kron(VOL[cell_id][1], Im)
        D̂t[cell_id, :, :] = kron(VOL[cell_id][2], Im)
        H_local = ref.H * Diagonal(geometric_terms.J_q[cell_id])
        # H[cell_id, :, :] = H_local
        H̄[cell_id, :, :] = kron(H_local, Im)
        # Hinv[cell_id, :, :] = inv(H_local)
        H̄inv[cell_id, :, :] = inv(H̄[cell_id, :, :])
    end

    for interface in face_interfaces
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        Rγk_matrix, Rγv_matrix = Matrix(ref.R[lfγk]), Matrix(ref.R[lfγv])
        γk_id, γv_id = ref.R[lfγk].idxs, ref.R[lfγv].idxs

        # Rγ[interface.face_1] = Rγk_matrix
        # Rγ[interface.face_2] = Rγv_matrix
        R̄γ[interface.face_1] = kron(Rγk_matrix, I(3))
        R̄γ[interface.face_2] = kron(Rγv_matrix, I(3))

        # Hγ[interface.face_1] = ref.H_face
        # Hγ[interface.face_2] = ref.H_face
        H̄γ[interface.face_1] = kron(ref.H_face, Im)
        H̄γ[interface.face_2] = kron(ref.H_face, Im)

        maskγk, maskγv = @views ref.f_mask[lfγk], @views ref.f_mask[lfγv]
        normalγk, normalγv = @views geometric_terms.N_f[ck][:, maskγk], @views geometric_terms.N_f[cv][:, maskγv]
        Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
        Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]

        # Nxγ[interface.face_1] = Diagonal(Nxγk)
        # Nxγ[interface.face_2] = Diagonal(Nxγv)
        N̄xγ[interface.face_1] = kron(Diagonal(Nxγk), Im)
        N̄xγ[interface.face_2] = kron(Diagonal(Nxγv), Im)
        # Ntγ[interface.face_1] = Diagonal(Ntγk)
        # Ntγ[interface.face_2] = Diagonal(Ntγv)
        N̄tγ[interface.face_1] = kron(Diagonal(Ntγk), Im)
        N̄tγ[interface.face_2] = kron(Diagonal(Ntγv), Im)

        Exγk = Rγk_matrix' * ref.H_face * Diagonal(Nxγk) * Rγk_matrix
        Etγk = Rγk_matrix' * ref.H_face * Diagonal(Ntγk) * Rγk_matrix
        Exγv = Rγv_matrix' * ref.H_face * Diagonal(Nxγv) * Rγv_matrix
        Etγv = Rγv_matrix' * ref.H_face * Diagonal(Ntγv) * Rγv_matrix

        Rγkv = zeros(T, (length(γk_id), num_of_nodes))
        Rγkv[:, γv_id[Pv]] = I(length(γk_id))

        Rγvk = zeros(T, (length(γv_id), num_of_nodes))
        Rγvk[:, γk_id[Pk]] = I(length(γv_id))

        Exkv = Rγk_matrix' * ref.H_face * Diagonal(Nxγk) * Rγkv
        Etkv = Rγk_matrix' * ref.H_face * Diagonal(Ntγk) * Rγkv
        Exvk = Rγv_matrix' * ref.H_face * Diagonal(Nxγv) * Rγvk
        Etvk = Rγv_matrix' * ref.H_face * Diagonal(Ntγv) * Rγvk

        # curr_Ematrics = Ematrices(Exγk, Exkv, Etγk, Etkv, Exγv, Exvk, Etγv, Etvk)
        curr_Ēmatrices = Ematrices{T}(kron(Exγk, Im), kron(Exkv, Im), kron(Etγk, Im), kron(Etkv, Im), kron(Exγv, Im), kron(Exvk, Im), kron(Etγv, Im), kron(Etvk, Im))
        # E_for_interior_interfaces[interface] = curr_Ematrics
        Ē_for_interior_interfaces[interface] = curr_Ēmatrices
    end

    for faceindex in bottom_faceIndex
        cell_id, face_id = faceindex
        Rγ_local = Matrix(ref.R[face_id])
        normal = geometric_terms.N_f[cell_id][:, ref.f_mask[face_id]]
        Ntγ_local = normal[2, :]

        Etγ_local = Rγ_local' * ref.H_face * Diagonal(Ntγ_local) * Rγ_local
        Ētγ_local = kron(Etγ_local, I(num_of_variables))

        # Eγ_for_bottom[faceindex] = Etγ_local
        Ēγ_for_bottom[faceindex] = Ētγ_local
    end

    for faceindex in boundary_faceIndex
        cell_id, face_id = faceindex
        Rγ_local = Matrix(ref.R[face_id])
        normal = geometric_terms.N_f[cell_id][:, ref.f_mask[face_id]]
        Nxγ_local = normal[1, :]

        Exγ_local = Rγ_local' * ref.H_face * Diagonal(Nxγ_local) * Rγ_local
        Ēxγ_local = kron(Exγ_local, I(num_of_variables))

        # Eγ_for_boundary[faceindex] = Exγ_local
        Ēγ_for_boundary[faceindex] = Ēxγ_local
    end

    # 构造并返回新的 SimpleGrid
    return SimpleGrid{dim, C, T}(
        cells,
        ref,
        new_xyz_gmsh,
        VectorOfArrays{Coord{dim, T}, 1}(xyz_SBP),
        face_interfaces,
        face_sets,
        geometric_terms,
        # VOL,
        xyz_gmsh_vec,
        indices_moving_coords,
        # interfaces_aligning_shock,
        # interfaces_aligning_contact_wave,
        # smooth_interior_interfaces,
        # H, Hinv, Dx, Dt, Rγ, Hγ, Nxγ, Ntγ, E_for_interior_interfaces, Eγ_for_bottom, Eγ_for_boundary,
        H̄, H̄inv, D̂x, D̂t, R̄γ, H̄γ, N̄xγ, N̄tγ, Ē_for_interior_interfaces, Ēγ_for_bottom, Ēγ_for_boundary
    )
    
end