@inline Coord(x::Vector{T}) where {T <: Real} = Coord(SVector{length(x), T}(x))
@inline Coord(x::NTuple{N, T}) where {N, T <: Real} = Coord(SVector{N, T}(x))

"""
Matrix -> Vector{Coord}
Vector{Vector} -> Vector{Coord}
Vector{NTuple} -> Vector{Coord}
"""
@inline Coords(x::Union{Vector{Vector{T}}, Vector{NTuple{N, T}}}) where {N, T <: Real} = Coord.(x)
@inline Coords(x::AbstractMatrix{T}) where {T <: Real} = copy(reinterpret(SVector{size(x, 1), T},
                                                                          vec(x)))
"""
Get i'th coordinate of each point in `coords`
"""
@inline get_i_coordinates(coords::Vector{Coord{N, T}}, i::Int) where {N, T} = [c[i] for c in coords]
@inline get_i_coordinates(coords::AbstractVector{Coord{N, T}}, i::Int) where {N, T} = [c[i] for c in coords]
"""
Index for a cell in the global mesh
"""
struct CellIndex
    idx::Int
end

"""
Represent a unique face in the global mesh: (global_cell_idx, local_face_idx)
"""
struct FaceIndex
    idx::Tuple{Int, Int}
end

"""
Represent a unique edge in the global mesh: (global_cell_idx, local_edge_idx)
"""
struct EdgeIndex
    idx::Tuple{Int, Int}
end

"""
Represent a unique vertex in the global mesh: (global_cell_idx, local_vertex_idx)
"""
struct VertexIndex
    idx::Tuple{Int, Int}
end

for INDEX in (:VertexIndex, :EdgeIndex, :FaceIndex)
    @eval begin
        # Constructor
        ($INDEX)(a::Int, b::Int) = ($INDEX)((a, b))
        # Allow a, b = FaceIndex
        Base.getindex(I::($INDEX), i::Int) = I.idx[i]
        Base.iterate(I::($INDEX), state::Int = 1) = (state == 3) ? nothing : (I[state], state + 1)
        # Check for equality
        Base.isequal(x::$INDEX, y::$INDEX) = x.idx == y.idx
        Base.isequal(x::Tuple{Int, Int}, y::$INDEX) = x[1] == y.idx[1] && x[2] == y.idx[2]
        Base.isequal(y::$INDEX, x::Tuple{Int, Int}) = x[1] == y.idx[1] && x[2] == y.idx[2]
    end
end

struct FaceInterface
    face_1::FaceIndex
    face_2::FaceIndex
    P1::Vector{Int}
    P2::Vector{Int}
end

struct Topology
    cell_to_vertex::Vector{NTuple}
    vertex_to_cell::Vector{Set{Int}}
    cell_neighbours::Vector{Vector{CellIndex}}
    face_face_neighbours::Matrix{FaceIndex}
end

struct GeometricTerms{T <: Real}
    J_q::VectorOfArrays{T, 1}       # elem, quad
    Λ_q::VectorOfArrays{T, 3}       # elem, quad, d, d
    J_f::VectorOfArrays{T, 1}       # elem, face_q
    Λ_f::VectorOfArrays{T, 3}       # elem, face_q, d, d
    N_f::VectorOfArrays{T, 2}       # elem, d, face_q
end

struct Grid{dim, C <: AbstractCell, T <: Real}
    cells::Vector{C}
    xyz_gmsh::Vector{Coord{dim, T}}
    xyz::VectorOfArrays{Coord{dim, T}, 1}
    xyz_q::VectorOfArrays{Coord{dim, T}, 1}
    xyz_f::VectorOfArrays{Coord{dim, T}, 1}
    mapping::Vector{PolynomialCurvilinearMapping}
    ref_elems_data::Dict{String, RefElemData}
    face_interfaces::Vector{FaceInterface}
    cell_sets::Dict{String, Set{CellIndex}}
    face_sets::Dict{String, Set{FaceIndex}}
    topology::Topology
    geometric_terms::GeometricTerms{T}
    VOL::Vector{NTuple{dim, Matrix{T}}}
    FAC::Vector{Any}
end

# TOPOLOGY
function compute_topology(cells::Vector{C}) where {C <: AbstractCell}
    @info "Computing grid topology..."
    cell_to_vertex = vertices.(cells)
    vertex_to_cell = create_vertex_to_cell_table(cell_to_vertex)
    cell_to_cell, face_to_face = compute_face_neighbours(cells, vertex_to_cell)
    return Topology(cell_to_vertex, vertex_to_cell, cell_to_cell, face_to_face)
end

function create_vertex_to_cell_table(cell_to_vertex::Vector{NTuple{N, Int}}) where {N}
    vertex_to_cell = Set{Int}[Set{Int}() for _ in 1:maximum(maximum.(cell_to_vertex))]
    for (cell_id, cell_vertices) in enumerate(cell_to_vertex)
        for vertex in cell_vertices
            push!(vertex_to_cell[vertex], cell_id)
        end
    end
    return vertex_to_cell
end

function compute_max_sizes(cells::Vector{C}) where {C <: AbstractCell}
    cell_type = eltype(cells)
    max_vertices, max_faces, max_edges = 0, 0, 0

    if isconcretetype(cell_type)
        max_vertices = n_vertices(cells[1])
        max_faces = n_faces(cells[1])
        max_edges = n_edges(cells[1])
    else
        cell_types = Set(typeof.(cells))
        for cell_type in cell_types
            cell_type_idx = findfirst(x -> typeof(x) == cell_type, cells)
            max_vertices = max(max_vertices, n_vertices(cells[cell_type_idx]))
            max_faces = max(max_faces, n_faces(cells[cell_type_idx]))
            max_edges = max(max_edges, n_edges(cells[cell_type_idx]))
        end
    end
    return (max_vertices, max_faces, max_edges)
end

function add_face_neighbour!(face_to_face::Matrix{FaceIndex}, cell::C1, cell_id::Int,
                             cell_neighbour::C2,
                             cell_neighbour_id::Int) where {C1 <: AbstractCell,
                                                            C2 <: AbstractCell}
    for (local_face, face) in enumerate(faces(cell))
        unique_face = sortface(face)
        for (local_face_2, face_neighbour) in enumerate(faces(cell_neighbour))
            unique_face_2 = sortface(face_neighbour)
            if unique_face == unique_face_2
                face_to_face[cell_id, local_face] = FaceIndex(cell_neighbour_id, local_face_2)
                return
            end
        end
    end
end

function compute_face_neighbours(cells::Vector{C},
                                 vertex_to_cell::Vector{Set{Int}}) where {C <: AbstractCell}
    _, max_faces, _ = compute_max_sizes(cells)
    face_to_face = Matrix{FaceIndex}(undef, length(cells), max_faces)
    fill!(face_to_face, FaceIndex(-1, -1))                            #TODO: Improve this
    cell_to_cell = Vector{Vector{CellIndex}}(undef, length(cells))
    for (cell_id, cell) in ProgressBar(enumerate(cells))
        cell_neighbour_ids = Set{Int}()
        for vertex in vertices(cell)
            for vertex_cell_id in vertex_to_cell[vertex]
                if vertex_cell_id != cell_id
                    push!(cell_neighbour_ids, vertex_cell_id)
                end
            end
        end
        cell_to_cell[cell_id] = CellIndex.(collect(cell_neighbour_ids))

        for cell_neighbour_id in cell_neighbour_ids
            cell_neighbour = cells[cell_neighbour_id]
            add_face_neighbour!(face_to_face, cell, cell_id, cell_neighbour, cell_neighbour_id)
        end
    end
    return cell_to_cell, face_to_face
end

function compute_face_interfaces(cells, xyz_f::Vector{Vector{Coord{dim, T}}},
                                 face_to_face::Matrix{FaceIndex}) where {dim, T}
    #TODO: Implement Set equality: isequal and hash.
    face_interfaces = Set{Set{FaceIndex}}()
    for i in axes(face_to_face)[1]
        for j in axes(face_to_face)[2]
            if face_to_face[i, j] != FaceIndex(-1, -1)
                push!(face_interfaces, Set([FaceIndex(i, j), face_to_face[i, j]]))
            end
        end
    end

    face_interfaces = collect(face_interfaces)
    interfaces = FaceInterface[]
    for face_interface in face_interfaces
        face_interface = collect(face_interface)
        c1, face_1 = face_interface[1]
        c2, face_2 = face_interface[2]
        ref_elem1, ref_elem2 = cells[c1].ref_data[], cells[c2].ref_data[]
        p1, p2 = match_coords(xyz_f[c1][ref_elem1.f_mask[face_1]],
                              xyz_f[c2][ref_elem2.f_mask[face_2]])
        push!(interfaces, FaceInterface(face_interface[1], face_interface[2], p1, p2))
    end
    return interfaces
end

function match_coords(u::Vector{Coord{N, T}}, v::Vector{Coord{N, T}}, tol = 1e-13) where {N, T}
    p1 = zeros(Int, length(u))
    p2 = zeros(Int, length(v))
    scaled_tol = tol * length(u)
    @inbounds for (i, coord1) in enumerate(u)
        coord1_norm = norm(coord1)
        for (j, coord2) in enumerate(v)
            max_u_v = max(coord1_norm, norm(coord2))
            if norm(coord1 - coord2) < scaled_tol * max(one(T), max_u_v)
                p1[i] = j
                p2[j] = i
            end
        end
    end
    @assert all(p1 .!= 0)&&all(p2 .!= 0) "Could not match all coordinates"
    return p1, p2
end


# TOPOLOGY for periodic BCs
#-------------------------#
# functions for periodic BC
function compute_topology_periodic(cells::Vector{C}; coords::Vector{Coord{dim, T}} = Vector{Coord{dim, T}}()
                                                    , vector_of_nodeTags::Vector{Vector{Int}} = Vector{Vector{Int}}()
                                                    , vector_of_nodeTagsMaster::Vector{Vector{Int}} = Vector{Vector{Int}}()) where {dim, C <: AbstractCell, T}

    @info "Computing periodic grid topology..."
    cell_to_vertex = vertices.(cells)
    vertex_to_cell = create_vertex_to_cell_table_periodic(cell_to_vertex, vector_of_nodeTags, vector_of_nodeTagsMaster)
    cell_to_cell, face_to_face = compute_face_neighbours_periodic(cells, coords, vertex_to_cell, vector_of_nodeTags, vector_of_nodeTagsMaster)
    return Topology(cell_to_vertex, vertex_to_cell, cell_to_cell, face_to_face)
end

# 先假定有两对周期边界吧。后续改3D再说
# looks working well, need tests latter
function create_vertex_to_cell_table_periodic(cell_to_vertex::Vector{NTuple{N, Int}}, 
                                                vector_of_nodeTags::Vector{Vector{C}}, vector_of_nodeTagsMaster::Vector{Vector{C}}) where {N, C}
    vertex_to_cell = Set{Int}[Set{Int}() for _ in 1:maximum(maximum.(cell_to_vertex))]
    for (cell_id, cell_vertices) in enumerate(cell_to_vertex)
        for vertex in cell_vertices
            push!(vertex_to_cell[vertex], cell_id)
        end
    end

    # 这里我的想法是用V的代码加完之后，在对那些边界上的顶点进行处理，要注意只拿边界而不是所有的边界上的点
    # 那么原来的vertex_to_cell，比如点(2, 0), 只有两个cells接触，现在要加上周期边界的cell，所以x方向加上两个，y方向加上两个
    # 比如 非cornor的顶点，就要加上对应x/y 方向的 cell
    for (nodeTags, masterNodeTags) in zip(vector_of_nodeTags, vector_of_nodeTagsMaster)
        err_msg = "The length of nodeTags and masterNodeTags is not equal, which means the number of nodes on one of the periodic boundary is not equal to the other. Check the gmsh file."
        @assert length(nodeTags) == length(masterNodeTags) err_msg
        
        for i in eachindex(nodeTags)
            curr_cells = Set{Int}()
            
            # 合并当前节点和其对应周期边界节点的 cells
            curr_cells = union(curr_cells, vertex_to_cell[nodeTags[i]], vertex_to_cell[masterNodeTags[i]])
            
            # 更新两个节点的 vertex_to_cell
            vertex_to_cell[nodeTags[i]] = deepcopy(curr_cells)
            vertex_to_cell[masterNodeTags[i]] = deepcopy(curr_cells)
        end
    end

    return vertex_to_cell
end


function is_two_faces_periodic(face1::NTuple{N, Int}, 
                               face2::NTuple{N, Int}, 
                               coords::Vector{Coord{dim, T}},
                               vector_of_nodeTags::Vector{Vector{C}}, 
                               vector_of_nodeTagsMaster::Vector{Vector{C}}) where {dim, N, C, T}

    @views nodeTags1 = vector_of_nodeTags[1]
    @views nodeTags2 = vector_of_nodeTags[2]
    @views masterNodeTags1 = vector_of_nodeTagsMaster[1]
    @views masterNodeTags2 = vector_of_nodeTagsMaster[2]

    # 一个face两个点定义一边，所以一共有四个点定义两个边
    # face 1 的
    # coord1_x = get_coordinate(coords[face1[1]])[1]
    # coord1_y = get_coordinate(coords[face1[1]])[2]
    # coord2_x = get_coordinate(coords[face1[2]])[1]
    # coord2_y = get_coordinate(coords[face1[2]])[2]
    coord1_x = get_i_coordinates(coords, 1)[face1[1]]
    coord1_y = get_i_coordinates(coords, 2)[face1[1]]
    coord2_x = get_i_coordinates(coords, 1)[face1[2]]
    coord2_y = get_i_coordinates(coords, 2)[face1[2]]
    # face 2 的
    # coord3_x = get_coordinate(coords[face2[1]])[1]
    # coord3_y = get_coordinate(coords[face2[1]])[2]
    # coord4_x = get_coordinate(coords[face2[2]])[1]
    # coord4_y = get_coordinate(coords[face2[2]])[2]
    coord3_x = get_i_coordinates(coords, 1)[face2[1]]
    coord3_y = get_i_coordinates(coords, 2)[face2[1]]
    coord4_x = get_i_coordinates(coords, 1)[face2[2]]
    coord4_y = get_i_coordinates(coords, 2)[face2[2]]

    if coord1_y == coord2_y && coord3_y == coord4_y && coord1_y !== coord3_y # y 方向上下周期边界
        if coord1_x == coord3_x && coord2_x == coord4_x
            return true
        elseif coord1_x == coord4_x && coord2_x == coord3_x
            return true
        else
            return false
        end
    elseif coord1_x == coord2_x && coord3_x == coord4_x && coord1_x !== coord3_x # x 方向左右周期边界
        if coord1_y == coord3_y && coord2_y == coord4_y
            return true
        elseif coord1_y == coord4_y && coord2_y == coord3_y
            return true
        else
            return false
        end
    else
        return false
    end
end


function add_face_neighbour_periodic!(face_face_neighbours::Matrix{FaceIndex}, cell::C1, cell_id::Int,
                              cell_neighbour::C2,
                              cell_neighbour_id::Int, 
                              coords::Vector{Coord{dim, T}},
                              vector_of_nodeTags::Vector{Vector{C}}, 
                              vector_of_nodeTagsMaster::Vector{Vector{C}}) where {C1 <: AbstractCell,
                                                             C2 <: AbstractCell, C, dim, T}
    for (local_face, face) in enumerate(faces(cell))
        unique_face = sortface(face) 
        for (local_face_2, face_neighbour) in enumerate(faces(cell_neighbour))
            unique_face_2 = sortface(face_neighbour)
            if unique_face == unique_face_2
                # 相邻的两个element的共边可能是（2，6）和（6，2），所以用sort来给排序一下。
                # 这种方法只能用真的相邻边， 周期性边界边就不行了，所以要单独处理
                face_face_neighbours[cell_id, local_face] = FaceIndex(cell_neighbour_id, local_face_2)
            elseif is_two_faces_periodic(face, face_neighbour, coords, vector_of_nodeTags, vector_of_nodeTagsMaster)
                face_face_neighbours[cell_id, local_face] = FaceIndex(cell_neighbour_id, local_face_2)
            end
        end
    end
end

function compute_face_neighbours_periodic(cells::Vector{C},
                                        coords::Vector{Coord{dim, T}},
                                        vertex_to_cell::Vector{Set{Int}}, 
                                        vector_of_nodeTags::Vector{Vector{Int}}, 
                                        vector_of_nodeTagsMaster::Vector{Vector{Int}}) where {C <: AbstractCell, T, dim}

    
    _, max_faces, _ = compute_max_sizes(cells)
    face_to_face = Matrix{FaceIndex}(undef, length(cells), max_faces)
    fill!(face_to_face, FaceIndex(-1, -1))                            #TODO: Improve this
    cell_to_cell = Vector{Vector{CellIndex}}(undef, length(cells))
    # cell_to_cell 是指当前cell的所有的neighbour，包括顶点的neighbor
    # face_to_face 是指当前cell的所有的face的neighbor

    for (cell_id, cell) in ProgressBar(enumerate(cells))
        cell_neighbour_ids = Set{Int}()
        for vertex in vertices(cell)
            for vertex_cell_id in vertex_to_cell[vertex]
                if vertex_cell_id != cell_id
                    push!(cell_neighbour_ids, vertex_cell_id)
                end
            end
        end


        cell_to_cell[cell_id] = SBPLite.CellIndex.(collect(cell_neighbour_ids))

        # 重点要改的是这个for loop, 注意到v的代码把边界element的边界face的neighbor记成了FaceIndex((-1, -1))
        # 但我们要把这个改成对应边界边的FaceIndex((global_cell_idx, local_face_idx))
        # struct FaceIndex
        #     idx::Tuple{Int, Int}
        # end
        # 想法是 分成两种情况，一种是非边界element，一种是边界element，非边界的直接走V的代码就行， 边界的要走我的_periodic!的代码
        cell_nodes = Set(get_nodes(cell))
        # 如果当前cell的node_id, 和四个边界上的node_id 都没有交集，那么就是非边界element
        # if isempty(intersect(cell_nodes, Set(vcat(vector_of_nodeTags...)))) && isempty(intersect(cell_nodes, Set(vcat(vector_of_nodeTagsMaster...))))
        if ((length(intersect(cell_nodes, Set(vcat(vector_of_nodeTags...)))) in (0, 1)) && (length(intersect(cell_nodes, Set(vcat(vector_of_nodeTagsMaster...)))) in (0, 1)))
            # 非边界element直接走V的代码
            for cell_neighbour_id in cell_neighbour_ids
                cell_neighbour = cells[cell_neighbour_id]
                add_face_neighbour!(face_to_face, cell, cell_id, cell_neighbour, cell_neighbour_id)
            end
        else
            # 边界element 走我的代码
            for cell_neighbour_id in cell_neighbour_ids
                cell_neighbour = cells[cell_neighbour_id]
                add_face_neighbour_periodic!(face_to_face, cell, cell_id, cell_neighbour, 
                                                cell_neighbour_id, coords, vector_of_nodeTags, vector_of_nodeTagsMaster)
            end
        end
    end
    
    return cell_to_cell, face_to_face
end

function compute_face_interfaces_periodic(cells, xyz_f::Vector{Vector{Coord{dim, T}}},
                                 face_to_face::Matrix{FaceIndex},
                                 xL::R, xR::R, yB::R, yT::R; tol = 1e-6) where {dim, T, R <: Real}
    #TODO: Implement Set equality: isequal and hash.
    # for periodicity, this for loop is fine
    face_interfaces = Set{Set{FaceIndex}}()
    for i in axes(face_to_face)[1]
        for j in axes(face_to_face)[2]
            if face_to_face[i, j] != FaceIndex(-1, -1)
                push!(face_interfaces, Set([FaceIndex(i, j), face_to_face[i, j]]))
            end
        end
    end

    face_interfaces = collect(face_interfaces)
    interfaces = FaceInterface[]
    for face_interface in face_interfaces
        face_interface = collect(face_interface)
        c1, face_1 = face_interface[1]
        c2, face_2 = face_interface[2]
        cell1 = cells[c1]
        cell2 = cells[c2]
        # ref_elem1, ref_elem2 = ref_elems_data[typeof(cell1)], ref_elems_data[typeof(cell2)]
        ref_elem1 = cell1.ref_data[]
        ref_elem2 = cell2.ref_data[]
        cell1_nodes = Set(get_nodes(cell1))
        cell2_nodes = Set(get_nodes(cell2))
        u = xyz_f[c1][ref_elem1.f_mask[face_1]]
        v = xyz_f[c2][ref_elem2.f_mask[face_2]]
        ux = get_i_coordinates(u, 1)
        uy = get_i_coordinates(u, 2)
        vx = get_i_coordinates(v, 1)
        vy = get_i_coordinates(v, 2)
        # for now this if statement works only for 2D
        # 边界element 走我的代码
    
        if (all(x -> isapprox(x, xL, atol = tol), ux) && all(x -> isapprox(x, xR, atol = tol), vx) || 
            all(x -> isapprox(x, xR, atol = tol), ux) && all(x -> isapprox(x, xL, atol = tol), vx) ||
            all(x -> isapprox(x, yB, atol = tol), uy) && all(x -> isapprox(x, yT, atol = tol), vy) ||
            all(x -> isapprox(x, yT, atol = tol), uy) && all(x -> isapprox(x, yB, atol = tol), vy))
            p1, p2 = match_coords_periodic(u, v, c1 = c1, c2 = c2)
        else 
            p1, p2 = match_coords(u, v)
        end
        push!(interfaces, FaceInterface(face_interface[1], face_interface[2], p1, p2))
    end

    return interfaces
end

function match_coords_periodic(u::Vector{Coord{N, T}}, v::Vector{Coord{N, T}}; tol = 1e-10, c1 = -1, c2 = -1) where {N, T}
    p1 = zeros(Int, length(u))
    p2 = zeros(Int, length(v))
    scaled_tol = tol * length(u)
    # 先判断u v是左右边界还是上下边界
    # 如果是左右边界，那么u 或者 v的 所有坐标的x坐标都是一样的，y坐标不一样
    # 如果是上下边界，那么u 或者 v的 所有坐标的y坐标都是一样的，x坐标不一样
    ux = get_i_coordinates(u, 1)
    uy = get_i_coordinates(u, 2)
    vx = get_i_coordinates(v, 1)
    vy = get_i_coordinates(v, 2)
    if all(x -> isapprox(x, ux[1], atol=tol), ux) && all(x -> isapprox(x, vx[1], atol=tol), vx) && abs(ux[1] - vx[1]) > 1e-3 
        # true if all x coordinates of u / v are the same, u and v are on left/right boundary
        @inbounds for (i, coord1) in enumerate(u)
            coord1_norm = coord1[2] # 与非边界element不同，这时我们只看y坐标
            for (j, coord2) in enumerate(v)
                max_u_v = max(coord1_norm, coord2[2])
                if abs(coord1[2] - coord2[2]) < scaled_tol * max(one(T), max_u_v)
                    p1[i] = j
                    p2[j] = i
                end
            end
        end
    elseif all(x -> isapprox(x, uy[1], atol=tol), uy) && all(x -> isapprox(x, vy[1], atol=tol), vy) && abs(uy[1] - vy[1]) > 1e-3
        # true if all y coordinates of u / v are the same, u and v are on top/bottom boundary
        @inbounds for (i, coord1) in enumerate(u)
            coord1_norm = coord1[1] # 与非边界element不同，这时我们只看x坐标
            for (j, coord2) in enumerate(v)
                max_u_v = max(coord1_norm, coord2[1])
                if abs(coord1[1] - coord2[1]) < scaled_tol * max(one(T), max_u_v)
                    p1[i] = j
                    p2[j] = i
                end
            end
        end
    else
        error("the face coupling $c1 and $c2 is not correctly dealt with, check match_coords_periodic and compute_face_interfaces_periodic functions")
    end
    @assert all(p1 .!= 0)&&all(p2 .!= 0) "Could not match all coordinates"
    return p1, p2
end
#---------------------------------#

function get_perm_matrix(seq1::NTuple{N, Integer}, seq2::NTuple{N, Integer}) where {N}
    perm = zeros(Int, N, N)
    @inbounds for j in 1:N
        for i in 1:N
            perm[j, i] = seq1[i] == seq2[j] ? 1 : 0
        end
    end
    return perm
end

function sortface(face::Tuple{Int, Int, Int, Int})
    a, b, c, d = face
    c, d = minmax(c, d)
    b, d = minmax(b, d)
    a, d = minmax(a, d)
    b, c = minmax(b, c)
    a, c = minmax(a, c)
    a, b = minmax(a, b)
    return (a, b, c) # If 3 out of 4 are equal, the fourth one is also equal
end

function sortface(face::Tuple{Int, Int, Int})
    a, b, c = face
    b, c = minmax(b, c)
    a, c = minmax(a, c)
    a, b = minmax(a, b)
    return (a, b, c)
end

function sortface(face::Tuple{Int, Int})
    a, b = face
    a < b ? (return face) : (return (b, a))
end

"""
Compute E matrix for given ref_elem and scaled face normals. The scaled face normals are along a
particular dimension (x, y, or z).
"""
#Vyom
# function compute_E_phys(ref_elem::RefElemData, N_f::AbstractMatrix{T}) where {T}
#     dim = size(N_f, 1)
#     E = Vector{Matrix{T}}(undef, dim)
#     for m in 1:dim
#         E[m] = zeros(T, size(first(ref_elem.Q))...)
#         for (f, f_mask) in enumerate(ref_elem.f_mask)
#             E[m] .+= Matrix(ref_elem.R[f]' * ref_elem.H_face * Diagonal(N_f[m, :][f_mask]) *
#                             ref_elem.R[f])
#         end
#         @assert isapprox(sum(E[m]), 0.0, atol = 1e-12)              # Check 1^T E 1 = 0
#     end
#     return Tuple(E)
# end

function compute_E_phys(ref_elem_data::RefElemData, N_f::Matrix{T}) where {T}
    dim = size(N_f, 1)
    E = zeros(T, size(first(ref_elem_data.Q))..., dim)
    for m in 1:dim
        for (f, f_mask) in enumerate(ref_elem_data.f_mask)
            E[:, :, m] .+= Matrix(ref_elem_data.R[f]' * ref_elem_data.H_face *
                                  Diagonal(N_f[m, :][f_mask]) * ref_elem_data.R[f])
        end
    end
    return E
end

# Vyom
# function compute_VOL_phys(ref_elem::RefElemData, Λ_q::Array{T, 3}, J_q::AbstractArray{T},
#                           E::NTuple{N, Matrix{T}}) where {T, N}
#     H, Q = ref_elem.H, ref_elem.Q
#     VOL = Vector{Matrix{T}}(undef, N)
#     for i in 1:N
#         VOL[i] = 0.5 * inv(H * Diagonal(J_q)) *
#                  (sum(Q[j]' * Diagonal(Λ_q[:, j, i]) - Diagonal(Λ_q[:, j, i]) * Q[j]
#                       for j in 1:N) + E[i])
#     end
#     return Tuple(VOL)
# end

function compute_FAC_phys(ref_elem::RefElemData, J_q::AbstractArray{T}) where {T}
    n_faces = length(ref_elem.f_mask)
    FAC = Vector{Matrix{T}}(undef, n_faces)
    for f in 1:n_faces
        FAC[f] = inv(ref_elem.H * Diagonal(J_q)) * ref_elem.R[f]' * ref_elem.H_face
    end
    return Tuple(FAC)
end


function compute_VOL_phys(ref_elem::RefElemData, Λ_q::Array{T, 3}, J_q, E::Array{T, 3}) where {T}
    H, Q = ref_elem.H, ref_elem.Q
    dim = length(Q)
    D_phy = Vector{Matrix{T}}(undef, dim)
    for i in 1:dim
        D_phy[i] = inv( Diagonal(J_q) * H) * (0.5 * (sum(Diagonal(Λ_q[:, j, i]) * Q[j] - Q[j]' * Diagonal(Λ_q[:, j, i]) for j in 1:dim) + E[:, :, i]))         
    end
    return Tuple(D_phy)
end

# GRID
function Grid(cells::Vector{C},
              xyz_gmsh::Vector{Coord{dim, T}},
              mapping::Vector{PolynomialCurvilinearMapping},
              ref_elems_data::Dict{String, RefElemData};
              cell_sets::Dict{String, Set{CellIndex}} = Dict{String, Set{CellIndex}}(),
              face_sets::Dict{String, Set{FaceIndex}} = Dict{String, Set{FaceIndex}}(), 
              vector_of_nodeTags::Vector{Vector{Int64}} =  Vector{Vector{Int64}}(),
              vector_of_nodeTagsMaster::Vector{Vector{Int64}} = Vector{Vector{Int64}}(),
              is_periodic::Bool = false) where {dim, C, T}
    @info "Setting up grid..."
    xyz = Vector{Vector{Coord{dim, T}}}(undef, length(cells))
    xyz_q = Vector{Vector{Coord{dim, T}}}(undef, length(cells))
    xyz_f = Vector{Vector{Coord{dim, T}}}(undef, length(cells))
    Λ_q = Vector{Array{T, 3}}(undef, length(cells))
    Λ_f = Vector{Array{T, 3}}(undef, length(cells))
    J_f = Vector{Vector{T}}(undef, length(cells))
    J_q = Vector{Vector{T}}(undef, length(cells))
    N_f = Vector{Array{T, 2}}(undef, length(cells))
    VOL = Vector{NTuple{dim, Matrix{T}}}(undef, length(cells))
    FAC = Vector(undef, length(cells))

    @inbounds Threads.@threads for i in ProgressBar(1:length(cells))
        cell = cells[i]
        ref_elem = cell.ref_data[]
        xyz[i] = comp_to_phys(mapping[i], ref_elem.rst |> Coords)
        xyz_q[i] = comp_to_phys(mapping[i], ref_elem.rst_q |> Coords)
        xyz_f[i] = comp_to_phys(mapping[i], ref_elem.rst_f |> Coords)
        Λ_f[i], J_f[i] = metric_terms_exact(mapping[i], ref_elem.rst_f |> Coords)
        J_f[i] = abs.(J_f[i]) # make sure the Jacobian is positive
        N_f[i] = zeros(T, dim, size(ref_elem.rst_f, 2))
        for m in 1:dim
            N_f[i][m, :] .= sum([Λ_f[i][:, n, m] .* ref_elem.n_rst[n, :] for n in 1:dim])
        end
        E = compute_E_phys(ref_elem, N_f[i])
        # Λ_q[i], J_q[i] = metric_terms_optimised(mapping[i], ref_elem.rst_q |> Coords, E, ref_elem)
        Λ_q[i], J_q[i] = metric_terms_optimised(mapping[i], ref_elem.rst_q |> Coords, E, ref_elem.Qt, ref_elem.Qt_inv)
        J_q[i] = abs.(J_q[i]) # make sure the Jacobian is positive
        # VOL[i] = compute_VOL_phys(ref_elem, Λ_q[i], J_q[i], E)
        VOL[i] = compute_VOL_phys(ref_elem, Λ_q[i], J_q[i], E)
        FAC[i] = compute_FAC_phys(ref_elem, J_q[i])
    end

    if is_periodic
        all_x = []
        all_y = []
        for i in 1:length(cells)
            x = get_i_coordinates(xyz[i], 1)
            y = get_i_coordinates(xyz[i], 2)
            append!(all_x, x)
            append!(all_y, y)
        end
        xL, xR = round(minimum(all_x), digits=5), round(maximum(all_x), digits=12)
        yB, yT = round(minimum(all_y), digits=5), round(maximum(all_y), digits=12)
        topology = compute_topology_periodic(cells, coords = xyz_gmsh, vector_of_nodeTags = vector_of_nodeTags, vector_of_nodeTagsMaster = vector_of_nodeTagsMaster)
        face_interfaces = compute_face_interfaces_periodic(cells, xyz_f, topology.face_face_neighbours, xL, xR, yB, yT, tol=1e-5)
    else
        topology = compute_topology(cells)
        face_interfaces = compute_face_interfaces(cells, xyz_f, topology.face_face_neighbours)
    end

    geometric_terms = GeometricTerms(VectorOfArrays.((J_q, Λ_q, J_f, Λ_f, N_f))...)
    # geometric_terms = GeometricTerms(J_q, Λ_q, J_f, Λ_f, N_f)

    
    return Grid(cells, xyz_gmsh, VectorOfArrays(xyz), VectorOfArrays(xyz_q), VectorOfArrays(xyz_f),
                mapping, ref_elems_data, face_interfaces, cell_sets, face_sets, topology,
                geometric_terms, VOL, FAC)
end




@inline n_cells(grid::Grid) = length(grid.cells)
@inline get_cell_sets(grid::Grid) = grid.cell_sets
@inline get_cell_set(grid::Grid, set_name::String) = grid.cell_sets[set_name]
@inline get_cells(grid::Grid) = grid.cells
@inline get_cells(grid::Grid, idxs::Union{Int, Vector{Int}}) = grid.cells[idxs]
@inline get_face_sets(grid::Grid) = grid.face_sets
@inline get_face_set(grid::Grid, set_name::String) = grid.face_sets[set_name]
@inline get_cell_type(grid::Grid) = eltype(grid.cells)
@inline get_cell_types(grid::Grid) = Set(typeof.(grid.cells))
@inline get_cell_type(grid::Grid, idx::Int) = eltype(grid.cells[idx])
@inline get_cell_coords(grid::Grid, idx::Int) = grid.xyz_gmsh[collect(get_nodes(grid.cells[idx]))]
@inline get_cell_coords(grid::Grid, cell::AbstractCell) = grid.xyz_gmsh[collect(get_nodes(cell))]
@inline get_ref_elem_data(grid::Grid, cell::AbstractCell) = grid.ref_elems_data[typeof(cell)]

@inline function find_FaceInterface(grid::Grid, faceIndex1::FaceIndex, faceIndex2::FaceIndex)
    for interface in grid.face_interfaces
         if ((faceIndex1 == interface.face_1 && faceIndex2 == interface.face_2) ||
                (faceIndex1 == interface.face_2 && faceIndex2 == interface.face_1))
              return interface
         end
    end
    error("No such interface found \n")
end


function get_nodes_per_cell(grid::Grid)
    if !isconcretetype(get_cell_type(grid))
        error("There are diff. types of cell in the grid. Cannot determine nodes per cell.")
    end
    return n_nodes(first(grid.cells))
end
@inline get_nodes_per_cell(grid::Grid, idx::Int) = n_nodes(grid.cells[idx])

function get_faces_per_cell(grid::Grid)
    if !isconcretetype(get_cell_type(grid))
        error("There are diff. types of cell in the grid. Cannot determine faces per cell.")
    end
    return n_faces(first(grid.cells))
end
@inline get_faces_per_cell(grid::Grid, idx::Int) = n_faces(grid.cells[idx])

