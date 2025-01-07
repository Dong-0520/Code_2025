
include("tensor_product_sbp.jl")
include("cg_triangle.jl")
# Normal vectors for faces
"""
Return matrix of normal vector at each face quadrature point for a given reference element.
"""
function normals_face_quad(::Type{R}, f_mask::Vector{Vector{T}}) where {R <: AbstractRefShape, T}
    normals = [repeat(normal(R, f), length(mask))' for (f, mask) in enumerate(f_mask)]
    return hcat(normals...)
end

struct RefElemData{T <: Number, N, M, R_type}
    degree::Integer
    H::AbstractMatrix{T}
    H_inv::AbstractMatrix{T}
    H_face::AbstractMatrix{T}
    D::NTuple{N, Matrix{T}}
    Q::NTuple{N, Matrix{T}}
    E::NTuple{N, Matrix{T}}
    Qt::Matrix{T}
    Qt_inv::Matrix{T}
    R::NTuple{M, R_type}
    f_mask::Vector{Vector{Int}}
    n_rst::Matrix{T}
    rst::Matrix{T}
    rst_q::Matrix{T}
    rst_f::Matrix{T}
end
@inline face_coords(elem::RefElemData, f::Int) = elem.rst_f[:, elem.f_mask[f]]

function TriangleDiagE(degree::Integer, quad_degree::Integer, vertices::Bool)
    @assert 1 <= degree <= 10
    @assert quad_degree == 2 * degree || quad_degree == 2 * degree - 1
    w, w_face, Qr, Qs, R, rst, rst_q, rst_f, f_mask = getSbpOperatorsTriDiagE(degree, quad_degree;
                                                                              vertices = vertices)
    n_rst = normals_face_quad(RefTriangle, f_mask)
    H, H_inv, H_face = Diagonal(w), Diagonal(1 ./ w), Diagonal(w_face)
    D = H_inv * Qr, H_inv * Qs
    E = Qr + Qr', Qs + Qs'
    Qt, Qt_inv = [Qr' Qs'], pinv([Qr' Qs'])
    return RefElemData(degree, H, H_inv, H_face, D, (Qr, Qs), E, Qt, Qt_inv, R, f_mask, n_rst, rst,
                       rst_q, rst_f)
end

TriangleDiagELGL(degree::Integer, quad_degree::Integer) = TriangleDiagE(degree, quad_degree, true)
TriangleDiagELG(degree::Integer, quad_degree::Integer) = TriangleDiagE(degree, quad_degree, false)

function TriRefElemOmega(degree::Integer)
    @assert 1 <= degree <= 4
    w, w_face, Qr, Qs, R, rst, rst_q, rst_f, f_mask = getSbpOperatorsTriOmega(degree)
    n_rst = normals_face_quad(RefTriangle, f_mask)
    H, H_inv, H_face = Diagonal(w), Diagonal(1 ./ w), Diagonal(w_face)
    D = H_inv * Qr, H_inv * Qs
    E = Qr + Qr', Qs + Qs'
    Qt, Qt_inv = [Qr' Qs'], pinv([Qr' Qs'])
    return RefElemData(degree, H, H_inv, H_face, (Qr, Qs), D, E, Qt, Qt_inv, R, f_mask, n_rst, rst,
                       rst_q, rst_f)
end

function TetrahedronDiagE(degree::Integer, quad_degree::Int)
    @assert 1 <= degree <= 5
    @assert quad_degree == 2 * degree || quad_degree == 2 * degree - 1
    w, w_face, Qr, Qs, Qt, R, rst, rst_quad, rst_face, f_mask = getSBPOperatorsTetDiagE(degree,
                                                                                        quad_degree)
    n_rst = normals_face_quad(RefTetrahedron, f_mask)
    H, H_inv, H_face = Diagonal(w), Diagonal(1 ./ w), Diagonal(w_face)
    D = H_inv * Qr, H_inv * Qs, H_inv * Qt
    E = Qr + Qr', Qs + Qs', Qt + Qt'
    Qt, Qt_inv = [Qr' Qs' Qt'], pinv([Qr' Qs' Qt'])
    return RefElemData(degree, H, H_inv, H_face, D, (Qr, Qs, Qt), E, Qt, Qt_inv, R, f_mask, n_rst,
                       rst, rst_quad, rst_face)
end



function QuadRefElemGL_David(degree::Integer)
    @assert 1 <= degree <= 14

    ξ, H_face, Qξ1d, Dξ1d, Mξ1d, tLξ1d, tRξ1d = tensor_product_sbp.d1_gl_sbp(degree, false)

    n_face_nodes = length(ξ)
    x = repeat(ξ, n_face_nodes)
    y = repeat(ξ, inner = n_face_nodes)
    rst = Matrix{Float64}(hcat(x, y)')
    rst_q = rst

    I_n = I(n_face_nodes)

    # 2D derivative operator on the reference element
    Dr = my_dropzeros(tensor_product_sbp.tensor_product_AB(I_n, Dξ1d))
    Ds = my_dropzeros(tensor_product_sbp.tensor_product_AB(Dξ1d, I_n))
    D = Dr, Ds

    H = my_dropzeros(tensor_product_sbp.tensor_product_AB(H_face, H_face))
    H_inv = my_dropzeros(inv(H))

    Qr = my_dropzeros(H * Dr)
    Qs = my_dropzeros(H * Ds)

    E = my_dropzeros(Qr + Qr'), my_dropzeros(Qs + Qs')

    Qt = [Qr' Qs']
    
    Qt_inv = pinv([Qr' Qs'])

    # 2D derivative operator on the reference element

    total_num_nodes = Int(n_face_nodes^2)

    R1 = tensor_product_sbp.tensor_product_AB(tLξ1d', I_n) # bottom
    R2 = tensor_product_sbp.tensor_product_AB(I_n, tRξ1d') # right
    R3 = tensor_product_sbp.tensor_product_AB(tRξ1d', I_n) # top
    R4 = tensor_product_sbp.tensor_product_AB(I_n, tLξ1d') # left

    # the following code is for the selection map version of R operator
    
    # positions1 = findall(x -> x == 1.0, R1)
    # face1_node_index = unique([pos[2] for pos in positions1])
    # R1_selectionMap = SBPLite.SelectionMap{Float64}(face1_node_index, total_num_nodes)
    
    # positions2 = findall(x -> x == 1.0, R2)
    # face2_node_index = unique([pos[2] for pos in positions2])
    # R2_selectionMap = SBPLite.SelectionMap{Float64}(face2_node_index, total_num_nodes)
    
    # positions3 = findall(x -> x == 1.0, R3)
    # face3_node_index = unique([pos[2] for pos in positions3])
    # R3_selectionMap = SBPLite.SelectionMap{Float64}(face3_node_index, total_num_nodes)
    
    # positions4 = findall(x -> x == 1.0, R4)
    # face4_node_index = unique([pos[2] for pos in positions4])
    # R4_selectionMap = SBPLite.SelectionMap{Float64}(face4_node_index, total_num_nodes)

    # R = Tuple(R1_selectionMap, R2_selectionMap, R3_selectionMap, R4_selectionMap)
    R = (R1, R2, R3, R4)

    rst_f1 = rst[:, 1:n_face_nodes]
    rst_f2 = rst[:, n_face_nodes:n_face_nodes:total_num_nodes]
    rst_f3 = rst[:,total_num_nodes: -1: total_num_nodes - n_face_nodes + 1]
    rst_f4 = rst[:, total_num_nodes - n_face_nodes + 1: -n_face_nodes:1]
    rst_f = hcat(rst_f1, rst_f2, rst_f3, rst_f4)

    f_mask = [collect(((i-1) * n_face_nodes + 1):(i * n_face_nodes)) for i in 1:4]

    n_rst = normals_face_quad(RefQuadrilateral, f_mask)

    return RefElemData(degree, H, H_inv, H_face, D, (Qr, Qs), E, Qt, Qt_inv, R, f_mask, n_rst, rst, rst_q, rst_f)
end

function QuadRefElemGL_Hicken(degree::Integer; boundary_nodes = true)
    @assert 1 <= degree <= 4
    w, Q, R, rs, rs_quad, rs_face = getSbpOperatorsQuad2d(degree, boundary_nodes = boundary_nodes)


    n_face_nodes = length(w)
    I_n = I(n_face_nodes)

    H_face = Diagonal(w)
    D1d = inv(H_face) * Q
    H = kron(H_face, H_face)
    H_inv = inv(H)

    Dr = my_dropzeros(kron(I_n, D1d))
    Ds = my_dropzeros(kron(D1d, I_n))
    D = Dr, Ds

    # D = H_inv * Qr, H_inv * Qs, H_inv * Qt
    Qr = my_dropzeros(H * Dr)
    Qs = my_dropzeros(H * Ds)
    Qt = [Qr' Qs']
    Qt_inv = pinv([Qr' Qs'])


    E = my_dropzeros(Qr + Qr'), my_dropzeros(Qs + Qs')

    rst_f = hcat(rs_face[1], rs_face[2], rs_face[3], rs_face[4])

    f_mask = [collect(((i-1) * n_face_nodes + 1):(i * n_face_nodes)) for i in 1:4]

    n_rst = normals_face_quad(RefQuadrilateral, f_mask)

    if boundary_nodes
        tLξ1d = zeros(n_face_nodes)
        tLξ1d[1] = 1
        tRξ1d = zeros(n_face_nodes)
        tRξ1d[end] = 1
        R1 = kron(tLξ1d', I_n) # bottom
        R2 = kron(I_n, tRξ1d') # right
        R3 = kron(tRξ1d', I_n) # top
        R4 = kron(I_n, tLξ1d') # left
        R = (R1, R2, R3, R4)
    else
        error("Not implemented yet")
    end


    return RefElemData(degree, H, H_inv, H_face, D, (Qr, Qs), E, Qt, Qt_inv, R, f_mask, n_rst, rs, rs_quad, rst_f)

end



function QuadRefElemFromDiagETriangle(degree::Integer, quad_degree::Integer, vertices::Bool)
    """
    Create a reference operators on Quad based on the operators on Triangle
    Graphically, the Quad is divided into two triangles, and the operators on the Quad are created by CG approach
    """
    xt, yt, Pt, Qxt, Qyt, Ext, Eyt = cg_triangle.triangle_from_Jesse(degree)
    quadξ, quadη, quadH, quadQξ, quadQη, quadEξ, quadEη = cg_triangle.quad_from_triangle_SBP(size(xt)[1], xt, yt, Pt, Qxt, Qyt, Ext, Eyt)
    num_of_nodes = size(quadξ)[1]
    
    rs = Matrix{Float64}(hcat(quadξ, quadη)')
    rs_quad = deepcopy(rs)


    Dr = inv(quadH)*quadQξ
    Ds = inv(quadH)*quadQη
    Dr = my_dropzeros(Dr)
    Ds = my_dropzeros(Ds)
    D = Dr, Ds

    quadH = my_dropzeros(quadH)
    quadHinv = inv(quadH)
    quadHinv = my_dropzeros(quadHinv)


    if degree == 1
        node_id_on_south = [1,4,2]
        node_id_on_east = [2,10,8]
        node_id_on_north = [8,9,3]
        node_id_on_west = [3,6,1]
    elseif degree == 2
        node_id_on_south = [1, 7, 8, 2]
        node_id_on_east = [2, 19, 20, 13]
        node_id_on_north = [13, 17, 18, 3]
        node_id_on_west = [3, 11, 12, 1]
    elseif degree == 3
        node_id_on_south = [1, 13, 4, 14, 2]
        node_id_on_east = [2, 30, 21, 31, 19]
        node_id_on_north = [19, 28, 20, 29, 3]
        node_id_on_west = [3, 17, 6, 18, 1]
    elseif degree == 4
        node_id_on_south = [1, 16, 10, 11, 17, 2]
        node_id_on_east = [2, 41, 37, 38, 42, 28]
        node_id_on_north = [28, 39, 35, 36, 40, 3]
        node_id_on_west = [3, 20, 14, 15, 21, 1]
    else
        error("Not implemented yet")
    end
    H_face_west = Diagonal([abs(quadEξ[i, i]) for i in node_id_on_west])
    H_face_east = Diagonal([abs(quadEξ[i, i]) for i in node_id_on_east])
    H_face_south = Diagonal([abs(quadEη[i, i]) for i in node_id_on_south])
    H_face_north = Diagonal([abs(quadEη[i, i]) for i in node_id_on_north])
    @assert H_face_west ≈ H_face_east ≈ H_face_south ≈ H_face_north  "H_faces should be the same, check your ref elem data"
    H_face = H_face_west

    Qr, Qs = quadQξ, quadQη
    Qr = my_dropzeros(Qr)
    Qs = my_dropzeros(Qs)

    E = my_dropzeros(Qr + Qr'), my_dropzeros(Qs + Qs')

    Qt = [Qr' Qs']
    Qt = my_dropzeros(Qt)
    Qt_inv = pinv([Qr' Qs'])
    Qt_inv = my_dropzeros(Qt_inv)

    num_of_nodes_on_face = length(node_id_on_south)
    R1 = zeros(num_of_nodes_on_face, num_of_nodes)
    R2 = zeros(num_of_nodes_on_face, num_of_nodes)
    R3 = zeros(num_of_nodes_on_face, num_of_nodes)
    R4 = zeros(num_of_nodes_on_face, num_of_nodes)
    for i in eachindex(node_id_on_east)
        R1[i, node_id_on_south[i]] = 1
        R2[i, node_id_on_east[i]] = 1
        R3[i, node_id_on_north[i]] = 1
        R4[i, node_id_on_west[i]] = 1
    end
    R = (R1, R2, R3, R4)

    rst_f1 = rs[:, node_id_on_south]
    rst_f2 = rs[:, node_id_on_east]
    rst_f3 = rs[:, node_id_on_north]
    rst_f4 = rs[:, node_id_on_west]
    rst_f = hcat(rst_f1, rst_f2, rst_f3, rst_f4)

    f_mask = [collect(((i-1) * num_of_nodes_on_face + 1):(i * num_of_nodes_on_face)) for i in 1:4]
    n_rst = normals_face_quad(RefQuadrilateral, f_mask)

    return RefElemData(degree, quadH, quadHinv, H_face, D, (Qr, Qs), E, Qt, Qt_inv, R, f_mask, n_rst, rs, rs_quad, rst_f)

end


function my_dropzeros(A::Matrix{T}) where {T <: Number}
    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            if abs(A[i, j]) < 1e-14
                A[i, j] = 0.0
            end
        end
    end
    return A
end