using Pkg
using JLD2

using LinearAlgebra, SparseArrays, Trixi, BlockArrays, ArraysOfArrays
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics        

include("../../src/SBPLite.jl")
using .SBPLite


@load joinpath(@__DIR__, "../lin_adv/lin_adv_grid_nonalign_50.jld2") grid

order = 2
ref = TriangleDiagELG(order, 2 * order)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)

include(joinpath(@__DIR__, "../plot_helper.jl"))



mutable struct SimpleGrid{dim, C <: AbstractCell, T <: Real}
    cells::Vector{C}
    ref::RefElemData
    xyz_gmsh::Vector{Coord{dim, T}}
    xyz_SBP::VectorOfArrays{Coord{dim, T}, 1}
    # mapping::Vector{PolynomialCurvilinearMapping}
    face_interfaces::Vector{FaceInterface}
    face_sets::Dict{String, Set{FaceIndex}}
    # topology::Topology
    geometric_terms::GeometricTerms{T}
    VOL::Vector{NTuple{dim, Matrix{T}}}
end




function construct_simpleGrid(cells::Vector{C},
                            ref::RefElemData, 
                            new_xyz_gmsh::Vector{Coord{dim, T}}, 
                            face_interfaces::Vector{FaceInterface},
                            face_sets::Dict{String, Set{FaceIndex}}, ; 
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
    FAC = Vector(undef, n_cells)
    
    @inbounds for i in 1:n_cells
        # 物理坐标
        xyz_SBP[i] = comp_to_phys(mapping_ref_phys[i], ref.rst_q |> Coords)
        
        # 面几何项
        Λ_f[i], J_f[i] = metric_terms_exact(mapping_ref_phys[i], ref.rst_f |> Coords)
        J_f[i] = abs.(J_f[i]) # make sure the Jacobian is positive
        N_f[i] = zeros(T, dim, size(ref.rst_f, 2))
        for m in 1:dim
            N_f[i][m, :] .= sum([Λ_f[i][:, n, m] .* ref.n_rst[n, :] for n in 1:dim])
        end
        
        E = compute_E_phys(ref, N_f[i])
        # Λ_q[i], J_q[i] = metric_terms_optimised(mapping[i], ref_elem.rst_q |> Coords, E, ref_elem)
        Λ_q[i], J_q[i] = metric_terms_optimised(mapping_ref_phys[i], ref.rst_q |> Coords, E, ref.Qt, ref.Qt_inv)
        J_q[i] = abs.(J_q[i]) # make sure the Jacobian is positive
        # VOL[i] = compute_VOL_phys(ref_elem, Λ_q[i], J_q[i], E)
        VOL[i] = compute_VOL_phys(ref, Λ_q[i], J_q[i], E)
        FAC[i] = compute_FAC_phys(ref, J_q[i])
    end
    
    # 创建几何项结构
     geometric_terms = GeometricTerms(VectorOfArrays.((J_q, Λ_q, J_f, Λ_f, N_f))...)
    
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
    )
    
end

simple_grid = construct_simpleGrid(
    deepcopy(grid.cells),
    ref,
    deepcopy(grid.xyz_gmsh),
    deepcopy(grid.face_interfaces),
    deepcopy(grid.face_sets)
)


"""
Designed for one-dimensional space-time slab,
return t0, tF, xL, xR
"""
function find_boundaries(simple_grid::SimpleGrid)
    t0 = minimum([coord[2] for coord in simple_grid.xyz_gmsh])
    tF = maximum([coord[2] for coord in simple_grid.xyz_gmsh])
    xL = minimum([coord[1] for coord in simple_grid.xyz_gmsh])
    xR = maximum([coord[1] for coord in simple_grid.xyz_gmsh])
    return t0, tF, xL, xR
end

t0, tF, xL, xR = find_boundaries(simple_grid)
xyz_gmsh = simple_grid.xyz_gmsh


"""
This function returns the total number of coordinates in the xyz_gmsh
    so the first 
"""
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

indices_for_moving, coords, t0, tF, xL, xR = get_meshinfo_for_buffer(simple_grid)


new_xyz_gmsh = map(coord -> coord.*1.2 , grid.xyz_gmsh)


simple_grid = construct_simpleGrid(
    deepcopy(grid.cells),
    ref,
    new_xyz_gmsh,
    deepcopy(grid.face_interfaces),
    deepcopy(grid.face_sets)
)


plot_SBP_mesh(grid)
plot_SBP_mesh_simpleGrid(test_simple_grid)


#------------ Polynomial exactness test --------------
# Test the polynomial exactness for the derivative operators
# 根据order，针对二维，建立一个从 exponent = 0 到 order 的多项式
# 比如 order = 3 时，我要建立 r1, r2 两个随机数字
# 再建立所有可能的 多项式的exponent，保证总exponent不超过 order
# e1 e2
# 0 0
# 0 1
# 0 2
# 0 3
# 1 0
# 1 1
# 1 2
# 2 1
# 那么多项式为 r1 * x^e1 + r2 * y^e2
# 其中 e1, e2 是 exponent 的一个组合
# 分别测试Dx Dy是否能得到正确的derivative
using Random
function polynomial_exactness(grid::SimpleGrid; tol = 1e-8)

    success_count = 0
    total_tests = 0
    total_exponent = grid.ref.degree

    # 生成所有可能的 exponent 组合
    exponents = []
    for total_exp in 0:total_exponent
        for e1 in 0:total_exp
            e2 = total_exp - e1
            push!(exponents, (e1, e2))  # 直接添加，不需要判断 e2 >= 0
        end
    end

    if any(sum(exponent) > total_exponent for exponent in exponents)
        error("Some exponents exceed the total degree.")
    end

    r1, r2, r = rand(3) .* 2 .- 1  # 三个随机系数

    @inbounds for cell_id in 1:length(grid.cells)

        x = [coord[1] for coord in grid.xyz_SBP[cell_id]]
        y = [coord[2] for coord in grid.xyz_SBP[cell_id]]

        Dx = grid.VOL[cell_id][1]
        Dy = grid.VOL[cell_id][2]

        for (e1, e2) in exponents
            total_tests += 2
            # 计算多项式值
            poly_add = r1 * x .^ e1 + r2 * y .^ e2
            
            # 计算Dx和Dy
            Dx_poly_add = if e1 > 0
                r1 * e1 * x .^(e1 - 1)
            else
                zeros(length(x))
            end

            Dy_poly_add = if e2 > 0
                r2 * e2 * y .^(e2 - 1)
            else
                zeros(length(y))
            end

            # SBP 算子结果
            Dx_SBP_add = Dx * poly_add
            Dy_SBP_add = Dy * poly_add

            # 检查加法形式
            if isapprox(Dx_SBP_add, Dx_poly_add, atol=tol) && isapprox(Dy_SBP_add, Dy_poly_add, atol=tol)
                success_count += 1
                println("    ✓ Additive form: r1*x^$e1 + r2*y^$e2 PASSED")
            else
                println("    ✗ Additive form: r1*x^$e1 + r2*y^$e2 FAILED")
                println("      Dx error: $(maximum(abs.(Dx_SBP_add - Dx_poly_add)))")
                println("      Dy error: $(maximum(abs.(Dy_SBP_add - Dy_poly_add)))")
            end

            # ===== 测试乘法形式：r * x^e1 * y^e2 =====
            poly_mult = (r * x .^ e1) .* (y .^ e2)

            # 解析导数：
            # ∂/∂x (r * x^e1 * y^e2) = r * e1 * x^(e1-1) * y^e2
            Dx_poly_mult = if e1 > 0
                (r * e1 * x .^(e1 - 1)) .* (y .^ e2)
            else
                zeros(length(x))
            end
            
            # ∂/∂y (r * x^e1 * y^e2) = r * x^e1 * e2 * y^(e2-1)
            Dy_poly_mult = if e2 > 0
                (r * x .^ e1) .* (e2 * y .^(e2 - 1))
            else
                zeros(length(y))
            end

            # SBP 算子结果
            Dx_SBP_mult = Dx * poly_mult
            Dy_SBP_mult = Dy * poly_mult

            # 检查乘法形式
            if isapprox(Dx_SBP_mult, Dx_poly_mult, atol=tol) && isapprox(Dy_SBP_mult, Dy_poly_mult, atol=tol)
                success_count += 1
                println("    ✓ Multiplicative form: r*x^$e1*y^$e2 PASSED")
            else
                println("    ✗ Multiplicative form: r*x^$e1*y^$e2 FAILED")
                println("      Dx error: $(maximum(abs.(Dx_SBP_mult - Dx_poly_mult)))")
                println("      Dy error: $(maximum(abs.(Dy_SBP_mult - Dy_poly_mult)))")
            end
        end

    end
     # 总结
    println("\n" * "="^60)
    println("POLYNOMIAL EXACTNESS TEST SUMMARY")
    println("="^60)
    println("Total tests: $total_tests")
    println("Passed: $success_count")
    println("Failed: $(total_tests - success_count)")
    println("Success rate: $(round(100*success_count/total_tests, digits=2))%")
    
    if success_count == total_tests
        println("🎉 ALL TESTS PASSED!")
    else
        println("⚠️  Some tests failed.")
    end
    
    return success_count == total_tests
    
end


function polynomial_exactness(grid::Grid; tol = 1e-8)

    success_count = 0
    total_tests = 0
    total_exponent = grid.cells[1].ref_data[].degree

    # 生成所有可能的 exponent 组合
    exponents = []
    for total_exp in 0:total_exponent
        for e1 in 0:total_exp
            e2 = total_exp - e1
            push!(exponents, (e1, e2))  # 直接添加，不需要判断 e2 >= 0
        end
    end

    if any(sum(exponent) > total_exponent for exponent in exponents)
        error("Some exponents exceed the total degree.")
    end

    r1, r2, r = rand(3) .* 2 .- 1  # 三个随机系数

    @inbounds for cell_id in 1:length(grid.cells)

        x = [coord[1] for coord in grid.xyz_q[cell_id]]
        y = [coord[2] for coord in grid.xyz_q[cell_id]]

        Dx = grid.VOL[cell_id][1]
        Dy = grid.VOL[cell_id][2]

        for (e1, e2) in exponents
            total_tests += 2
            # 计算多项式值
            poly_add = r1 * x .^ e1 + r2 * y .^ e2
            
            # 计算Dx和Dy
            Dx_poly_add = if e1 > 0
                r1 * e1 * x .^(e1 - 1)
            else
                zeros(length(x))
            end

            Dy_poly_add = if e2 > 0
                r2 * e2 * y .^(e2 - 1)
            else
                zeros(length(y))
            end

            # SBP 算子结果
            Dx_SBP_add = Dx * poly_add
            Dy_SBP_add = Dy * poly_add

            # 检查加法形式
            if isapprox(Dx_SBP_add, Dx_poly_add, atol=tol) && isapprox(Dy_SBP_add, Dy_poly_add, atol=tol)
                success_count += 1
                println("    ✓ Additive form: r1*x^$e1 + r2*y^$e2 PASSED")
            else
                println("    ✗ Additive form: r1*x^$e1 + r2*y^$e2 FAILED")
                println("      Dx error: $(maximum(abs.(Dx_SBP_add - Dx_poly_add)))")
                println("      Dy error: $(maximum(abs.(Dy_SBP_add - Dy_poly_add)))")
            end

            # ===== 测试乘法形式：r * x^e1 * y^e2 =====
            poly_mult = (r * x .^ e1) .* (y .^ e2)

            # 解析导数：
            # ∂/∂x (r * x^e1 * y^e2) = r * e1 * x^(e1-1) * y^e2
            Dx_poly_mult = if e1 > 0
                (r * e1 * x .^(e1 - 1)) .* (y .^ e2)
            else
                zeros(length(x))
            end
            
            # ∂/∂y (r * x^e1 * y^e2) = r * x^e1 * e2 * y^(e2-1)
            Dy_poly_mult = if e2 > 0
                (r * x .^ e1) .* (e2 * y .^(e2 - 1))
            else
                zeros(length(y))
            end

            # SBP 算子结果
            Dx_SBP_mult = Dx * poly_mult
            Dy_SBP_mult = Dy * poly_mult

            # 检查乘法形式
            if isapprox(Dx_SBP_mult, Dx_poly_mult, atol=tol) && isapprox(Dy_SBP_mult, Dy_poly_mult, atol=tol)
                success_count += 1
                println("    ✓ Multiplicative form: r*x^$e1*y^$e2 PASSED")
            else
                println("    ✗ Multiplicative form: r*x^$e1*y^$e2 FAILED")
                println("      Dx error: $(maximum(abs.(Dx_SBP_mult - Dx_poly_mult)))")
                println("      Dy error: $(maximum(abs.(Dy_SBP_mult - Dy_poly_mult)))")
            end
        end

    end
     # 总结
    println("\n" * "="^60)
    println("POLYNOMIAL EXACTNESS TEST SUMMARY")
    println("="^60)
    println("Total tests: $total_tests")
    println("Passed: $success_count")
    println("Failed: $(total_tests - success_count)")
    println("Success rate: $(round(100*success_count/total_tests, digits=2))%")
    
    if success_count == total_tests
        println("🎉 ALL TESTS PASSED!")
    else
        println("⚠️  Some tests failed.")
    end
    
    return success_count == total_tests
    
end
polynomial_exactness(test_simple_grid)
polynomial_exactness(grid)


xyz_gmsh = grid.xyz_gmsh


function extract_info_for_xyzgmsh(grid::SimpleGrid, 
                                tF::Float64,
                                xL::Float64,
                                xR ::Float64;
                                t0 = 0.0,)

    xyz_gmsh = grid.xyz_gmsh
    x_gmsh = [coord[1] for coord in xyz_gmsh]
    y_gmsh = [coord[2] for coord in xyz_gmsh]
    num_of_nodes = length(x_gmsh)

    index_for_nodes_on_left_boundary = findall(x -> x ≈ xL, x_gmsh)
    index_for_nodes_on_right_boundary = findall(x -> x ≈ xR, x_gmsh)
    index_for_nodes_on_initial_time = findall(y -> y ≈ t0, y_gmsh)
    index_for_nodes_on_final_time = findall(y -> y ≈ tF, y_gmsh)

    index_for_four_corners = Vector{Int}()
    corners = [
        (xL, t0), (xR, t0), 
        (xL, tF), (xR, tF)
    ]
    for (i, coord) in enumerate(xyz_gmsh)
        x, y = coord[1], coord[2]
        if any(corner -> (x ≈ corner[1]) && (y ≈ corner[2]), corners)
            push!(index_for_four_corners, i)
        end
    end
    


end


print("end of file ")
#----------------------------------------------------
# codes from SBPLite


# cells::Vector{C},
#                             ref::RefElemData, 
#                             new_xyz_gmsh::Vector{Coord{dim, T}}, 
#                             face_interfaces::Vector{FaceInterface},
#                             face_sets::Dict{String, Set{FaceIndex}}, 
#                             topology::Topology
# xyz = Vector{Vector{Coord{dim, T}}}(undef, length(cells))
# xyz_q = Vector{Vector{Coord{dim, T}}}(undef, length(cells))
# xyz_f = Vector{Vector{Coord{dim, T}}}(undef, length(cells))
# Λ_q = Vector{Array{T, 3}}(undef, length(cells))
# Λ_f = Vector{Array{T, 3}}(undef, length(cells))
# J_f = Vector{Vector{T}}(undef, length(cells))
# J_q = Vector{Vector{T}}(undef, length(cells))
# N_f = Vector{Array{T, 2}}(undef, length(cells))
# VOL = Vector{NTuple{dim, Matrix{T}}}(undef, length(cells))
# FAC = Vector(undef, length(cells))


# @inbounds Threads.@threads for i in ProgressBar(1:length(cells))
#     cell = cells[i]
#     ref_elem = cell.ref_data[]
#     xyz[i] = comp_to_phys(mapping[i], ref_elem.rst |> Coords)
#     xyz_q[i] = comp_to_phys(mapping[i], ref_elem.rst_q |> Coords)
#     xyz_f[i] = comp_to_phys(mapping[i], ref_elem.rst_f |> Coords)
#     Λ_f[i], J_f[i] = metric_terms_exact(mapping[i], ref_elem.rst_f |> Coords)
#     J_f[i] = abs.(J_f[i]) # make sure the Jacobian is positive
#     N_f[i] = zeros(T, dim, size(ref_elem.rst_f, 2))
#     for m in 1:dim
#         N_f[i][m, :] .= sum([Λ_f[i][:, n, m] .* ref_elem.n_rst[n, :] for n in 1:dim])
#     end
#     E = compute_E_phys(ref_elem, N_f[i])
#     # Λ_q[i], J_q[i] = metric_terms_optimised(mapping[i], ref_elem.rst_q |> Coords, E, ref_elem)
#     Λ_q[i], J_q[i] = metric_terms_optimised(mapping[i], ref_elem.rst_q |> Coords, E, ref_elem.Qt, ref_elem.Qt_inv)
#     J_q[i] = abs.(J_q[i]) # make sure the Jacobian is positive
#     # VOL[i] = compute_VOL_phys(ref_elem, Λ_q[i], J_q[i], E)
#     VOL[i] = compute_VOL_phys(ref_elem, Λ_q[i], J_q[i], E)
#     FAC[i] = compute_FAC_phys(ref_elem, J_q[i])
# end