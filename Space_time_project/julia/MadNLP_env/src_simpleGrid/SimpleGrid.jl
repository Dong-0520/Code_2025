mutable struct SpaceTimeBuffer
    grid::Any                           # 网格
    a::Vector{Float64}                  # 参数 [ax, at]
    initial_guess::Array{Float64, 1}    # 初始猜测, 包括所有的动的和不动的xyz_gms坐标，和 oscillating_u 作为最开始的解
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
    
    # 构造函数
    function SpaceTimeBuffer(grid, a::Vector{Float64}, oscillating_u::Array{Float64, 2})
        indices_moving_coords, coords_xyz_gmsh, t0, tF, xL, xR = get_meshinfo_for_buffer(grid)
        initial_guess = vcat(coords_xyz_gmsh, vec(oscillating_u))
        index_mesh = length(coords_xyz_gmsh)  # 网格坐标的索引
        ref = grid.cells[1].ref_data[]  # 获取参考元素数据
        new(grid, a, 
        deepcopy(initial_guess), 
        index_mesh, 
        indices_moving_coords,  # 可移动坐标的索引
        xL, xR, t0, tF,
        zeros(Float64, length(initial_guess)),  # 最终解初始化为零
        Inf, Inf, Inf, false, Inf, ref)  # 初始目标函数值、最终目标函数值、最终残差、是否收敛、求解时间
    end
end


# 这个struct 方便我们每次更新了 mesh坐标之后，能够更新 SBP operators
mutable struct SimpleGrid{dim, C <: AbstractCell, T <: Real}
    cells::Vector{C}
    ref::RefElemData
    xyz_gmsh::Vector{Coord{dim, T}}
    xyz_SBP::VectorOfArrays{Coord{dim, T}, 1}
    face_interfaces::Vector{FaceInterface}
    face_sets::Dict{String, Set{FaceIndex}}
    geometric_terms::GeometricTerms{T}
    VOL::Vector{NTuple{dim, Matrix{T}}}

    xyz_gmsh_vec::Vector{T}              # 展平的坐标向量
    indices_moving_coords::Vector{Int64}       # 可动坐标的索引
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



function construct_simpleGrid(cells::Vector{C},
                            ref::RefElemData, 
                            new_xyz_gmsh::Vector{Coord{dim, T}}, 
                            face_interfaces::Vector{FaceInterface},
                            face_sets::Dict{String, Set{FaceIndex}}, 
                            indices_moving_coords::Vector{Int64},; 
                            comp_coords::Vector{SVector{dim, T}} = 
                                        [SVector{dim, T}(-1.0, -1.0),
                                        SVector{dim, T}(1.0, -1.0),
                                        SVector{dim, T}(-1.0, 1.0)],
                            basis = MonomialType, 
                            gmsh_order = 1) where {dim, C, T<: Real} 

    # 重新计算映射
    n_cells = length(cells)
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
    
    @inbounds for i in 1:n_cells
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
            # 修复：确保 E 是正确的类型，并使用正确的函数签名
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

    # 构造并返回新的 SimpleGrid
    return SimpleGrid{dim, C, T}(
        cells,
        ref,
        new_xyz_gmsh,
        VectorOfArrays{Coord{dim, T}, 1}(xyz_SBP),
        face_interfaces,
        face_sets,
        geometric_terms,
        VOL,
        xyz_gmsh_vec,
        indices_moving_coords
    )
    
end