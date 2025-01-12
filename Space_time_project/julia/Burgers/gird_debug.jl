using LinearAlgebra, BlockArrays, SparseArrays
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics                                                                                                                                                                                                                                                                          

include("../src/SBPLite.jl")
using .SBPLite
include("plotting_helper.jl")

order = 2
ref = TriangleDiagELGL(order, 2 * order)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)


mesh_file = "C:\\MyGraduateResearch\\Code_2025\\Space_time_project\\julia\\Burgers\\mesh\\BurgersMeshSlab_005_SimpleTriangle.msh"
using gmsh_jll
include(gmsh_jll.gmsh_api)
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)               # Show output in terminal
gmsh.open(mesh_file)
gmsh.model.mesh.renumber_nodes()                            # Consecutive node and elem numbering
gmsh.model.mesh.renumber_elements()


dim = Int64(gmsh.model.get_dimension())
face_dim = dim - 1

xyz_gmsh = SBPLite.extract_nodes()                   #TODO: Think of long term soln.
cells, elem_tags, mapping = SBPLite.extract_elems(dim, xyz_gmsh, ref_elems_data)
cell_sets = SBPLite.extract_cell_sets(dim, elem_tags)

boundary_dict = SBPLite.extract_boundary_face_nodes(face_dim)
face_sets = SBPLite.extract_boundary_faces(boundary_dict, cells)

is_periodic, line_tags = SBPLite.is_mesh_periodic(face_dim)
vector_of_nodeTags, vector_of_nodeTagsMaster = SBPLite._extract_periodicNodes(face_dim, line_tags)

Grid(cells, xyz_gmsh, mapping, ref_elems_data, 
                    cell_sets = cell_sets, 
                        face_sets = face_sets, 
                            vector_of_nodeTags = vector_of_nodeTags,    
                                vector_of_nodeTagsMaster = vector_of_nodeTagsMaster, 
                                    is_periodic = is_periodic)


xyz = Vector{Vector{Coord{dim, Float64}}}(undef, length(cells))
xyz_q = Vector{Vector{Coord{dim, Float64}}}(undef, length(cells))
xyz_f = Vector{Vector{Coord{dim, Float64}}}(undef, length(cells))
Λ_q = Vector{Array{Float64, 3}}(undef, length(cells))
Λ_f = Vector{Array{Float64, 3}}(undef, length(cells))
J_f = Vector{Vector{Float64}}(undef, length(cells))
J_q = Vector{Vector{Float64}}(undef, length(cells))
N_f = Vector{Array{Float64, 2}}(undef, length(cells))
VOL = Vector{NTuple{dim, Matrix{Float64}}}(undef, length(cells))
FAC = Vector(undef, length(cells))

for i in 1:length(cells)
    cell = cells[i]
    ref_elem = cell.ref_data[]
    xyz[i] = SBPLite.comp_to_phys(mapping[i], ref_elem.rst |> Coords)
    xyz_q[i] = SBPLite.comp_to_phys(mapping[i], ref_elem.rst_q |> Coords)
    xyz_f[i] = SBPLite.comp_to_phys(mapping[i], ref_elem.rst_f |> Coords)
    Λ_f[i], J_f[i] = SBPLite.metric_terms_exact(mapping[i], ref_elem.rst_f |> Coords)
    J_f[i] = abs.(J_f[i]) # make sure the Jacobian is positive
    N_f[i] = zeros(Float64, dim, size(ref_elem.rst_f, 2))
    for m in 1:dim
        N_f[i][m, :] .= sum([Λ_f[i][:, n, m] .* ref_elem.n_rst[n, :] for n in 1:dim])
    end
    E = SBPLite.compute_E_phys(ref_elem, N_f[i])
    # Λ_q[i], J_q[i] = metric_terms_optimised(mapping[i], ref_elem.rst_q |> Coords, E, ref_elem)
    Λ_q[i], J_q[i] = SBPLite.metric_terms_optimised(mapping[i], ref_elem.rst_q |> Coords, E, ref_elem.Qt, ref_elem.Qt_inv)
    J_q[i] = abs.(J_q[i]) # make sure the Jacobian is positive
    # VOL[i] = compute_VOL_phys(ref_elem, Λ_q[i], J_q[i], E)
    VOL[i] = SBPLite.compute_VOL_phys(ref_elem, Λ_q[i], J_q[i], E)
    FAC[i] = SBPLite.compute_FAC_phys(ref_elem, J_q[i])
end
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
topology = compute_topology_periodic_test(cells, coords = xyz_gmsh, vector_of_nodeTags = vector_of_nodeTags, vector_of_nodeTagsMaster = vector_of_nodeTagsMaster, vertices_of_domain = [xL, xR, yB, yT])



vertex_to_cell = SBPLite.create_vertex_to_cell_table_periodic(vertices.(cells), vector_of_nodeTags, vector_of_nodeTagsMaster)
_, max_faces, _ = SBPLite.compute_max_sizes(cells)
face_to_face = Matrix{FaceIndex}(undef, length(cells), max_faces)
fill!(face_to_face, FaceIndex(-1, -1))                            #TODO: Improve this
cell_to_cell = Vector{Vector{SBPLite.CellIndex}}(undef, length(cells))
# cell_to_cell 是指当前cell的所有的neighbour，包括顶点的neighbor
# face_to_face 是指当前cell的所有的face的neighbor
cell_id = 12
cell = cells[cell_id]

cell_neighbour_ids = Set{Int}()
for vertex in vertices(cell)
    for vertex_cell_id in vertex_to_cell[vertex]
        if vertex_cell_id != cell_id
            push!(cell_neighbour_ids, vertex_cell_id)
        end
    end
end


cell_to_cell[cell_id] = CellIndex.(collect(cell_neighbour_ids))

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

cell_neighbour_id = 14
cell_neighbour = cells[cell_neighbour_id]
add_face_neighbour_periodic!(face_to_face, cell, cell_id, cell_neighbour, 
                                    cell_neighbour_id, coords, vector_of_nodeTags, vector_of_nodeTagsMaster)

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

local_face = 1
face = faces(cell)[local_face]
unique_face = SBPLite.sortface(face) 
for (local_face_2, face_neighbour) in enumerate(faces(cell_neighbour))
    unique_face_2 = SBPLite.sortface(face_neighbour)
    if unique_face == unique_face_2
        # 相邻的两个element的共边可能是（2，6）和（6，2），所以用sort来给排序一下。
        # 这种方法只能用真的相邻边， 周期性边界边就不行了，所以要单独处理
        # face_face_neighbours[cell_id, local_face] = FaceIndex(cell_neighbour_id, local_face_2)
        println("local_face2 = ", local_face_2, "face_neighbour = ", face_neighbour, "nonPeriodic")
    elseif SBPLite.is_two_faces_periodic(face, face_neighbour, xyz_gmsh, vector_of_nodeTags, vector_of_nodeTagsMaster)
        # face_face_neighbours[cell_id, local_face] = FaceIndex(cell_neighbour_id, local_face_2)
        println("local_face2 = ", local_face_2, "face_neighbour = ", face_neighbour, "Periodic")
    end
end








# face_interfaces = SBPLite.compute_face_interfaces_periodic(cells, xyz_f, topology.face_face_neighbours, xL, xR, yB, yT, tol=1e-5)
face_interfaces = compute_face_interfaces_periodic_test(cells, xyz_f, topology.face_face_neighbours, xL, xR, yB, yT, tol=1e-5)




u = StaticArraysCore.SVector{2, Float64}[[0.06249999999985273, 3.469446951953614e-18], [0.06249999999985273, 0.013819660112501058], [0.06249999999985273, 0.036180339887498955], [0.06249999999985273, 0.05]]
v = StaticArraysCore.SVector{2, Float64}[[0.0, 0.05], [0.0, 0.036180339887498955], [0.0, 0.013819660112501058], [0.0, 3.469446951953614e-18]]

scatter([0.06249999999985273, 0.06249999999985273, 0.06249999999985273, 0.06249999999985273], [3.469446951953614e-18, 0.013819660112501058, 0.036180339887498955, 0.05], label="u")
scatter!([0, 0, 0, 0], [0.05, 0.036180339887498955, 0.013819660112501058, 3.469446951953614e-18], label="v")
face_interfaces = Set{Set{FaceIndex}}()
for i in axes(topology.face_face_neighbours)[1]
    for j in axes(topology.face_face_neighbours)[2]
        if topology.face_face_neighbours[i, j] != FaceIndex(-1, -1)
            if topology.face_face_neighbours[i, j] == FaceIndex((14,1)) || topology.face_face_neighbours[i, j] == FaceIndex((12,1))
                println("i = ", i, "j = ", j)
            end
            push!(face_interfaces, Set([FaceIndex(i, j), topology.face_face_neighbours[i, j]]))
        end
    end
end
face_interfaces = collect(face_interfaces)
interfaces = FaceInterface[]
i = 36
face_interface = collect(face_interfaces[i])
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
    p1, p2 = SBPLite.match_coords_periodic(u, v, c1 = c1, c2 = c2)
else 
    print("u = ", u, "\n", "v = ", v, "\n", "i = ", i, "\n")
    p1, p2 = match_coords_test(u, v)
end
push!(interfaces, FaceInterface(face_interface[1], face_interface[2], p1, p2))

function compute_topology_periodic_test(cells::Vector{C}; coords::Vector{Coord{dim, T}} = Vector{Coord{dim, T}}()
                                                    , vector_of_nodeTags::Vector{Vector{Int}} = Vector{Vector{Int}}()
                                                    , vector_of_nodeTagsMaster::Vector{Vector{Int}} = Vector{Vector{Int}}()
                                                    , vertices_of_domain = Vector{Float64}()) where {dim, C <: AbstractCell, T}

    @info "Computing periodic grid topology..."
    cell_to_vertex = vertices.(cells)
    vertex_to_cell = SBPLite.create_vertex_to_cell_table_periodic(cell_to_vertex, vector_of_nodeTags, vector_of_nodeTagsMaster)
    cell_to_cell, face_to_face = compute_face_neighbours_periodic(cells, coords, vertex_to_cell, vector_of_nodeTags, vector_of_nodeTagsMaster, vertices_of_domain = vertices_of_domain)
    return Topology(cell_to_vertex, vertex_to_cell, cell_to_cell, face_to_face)
end


function match_coords_test(u::Vector{Coord{N, T}}, v::Vector{Coord{N, T}}, tol = 1e-10) where {N, T}
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

function compute_face_interfaces_periodic_test(cells, xyz_f::Vector{Vector{Coord{dim, T}}},
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
    for i in eachindex(face_interfaces)
        face_interface = collect(face_interfaces[i])
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
            p1, p2 = SBPLite.match_coords_periodic(u, v, c1 = c1, c2 = c2)
        else 
            print("u = ", u, "\n", "v = ", v, "\n", "i = ", i, "\n")
            p1, p2 = match_coords_test(u, v)
        end
        push!(interfaces, FaceInterface(face_interface[1], face_interface[2], p1, p2))
    end

    return interfaces
end

function match_coords_test(u::Vector{Coord{N, T}}, v::Vector{Coord{N, T}}, tol = 1e-10) where {N, T}
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


uk, uv = u[ck, :], u[cv, :]
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]


interface = grid.face_interfaces[1]
c1, lf1 = interface.face_1
c2, lf2 = interface.face_2
P1, P2 = interface.P1, interface.P2
elem1, elem2 = grid.cells[c1].ref_data[], grid.cells[c2].ref_data[]
R_1, R_2 = elem1.R[lf1], elem2.R[lf2]
# cell1_face_x = R_1 * SBPLite.get_i_coordinates(grid.xyz[c1], 1)
# cell1_face_y = R_1 * SBPLite.get_i_coordinates(grid.xyz[c1], 2)
# cell2_face_x = R_2 * SBPLite.get_i_coordinates(grid.xyz[c2], 1)
# cell2_face_y = R_2 * SBPLite.get_i_coordinates(grid.xyz[c2], 2)
uk = sinpi
dx = abs(cell1_face_x[end] - cell1_face_x[1])
dy = abs(cell1_face_y[end] - cell1_face_y[1])
if dx > 1e-3 && dy > 1e-3
    @test cell1_face_x ≈ cell2_face_x[P2] atol=1e-3
    @test cell1_face_y ≈ cell2_face_y[P2] atol=1e-3
    @test cell1_face_x[P1] ≈ cell2_face_x atol=1e-3
    @test cell1_face_y[P1] ≈ cell2_face_y atol=1e-3
elseif isapprox(dx, 0, atol=1e-10)
    @test cell1_face_y ≈ cell2_face_y[P2] atol=1e-3
    @test cell1_face_y[P1] ≈ cell2_face_y atol=1e-3
elseif isapprox(dy, 0, atol=1e-10)
    @test cell1_face_x ≈ cell2_face_x[P2] atol=1e-3
    @test cell1_face_x[P1] ≈ cell2_face_x atol=1e-3
end

if dx > 1e-3 && dy > 1e-3
    @test cell1_face_x ≈ cell2_face_x atol=1e-3
    @test cell1_face_y ≈ cell2_face_y atol=1e-3
    @test cell1_face_x ≈ cell2_face_x atol=1e-3
    @test cell1_face_y ≈ cell2_face_y atol=1e-3
elseif isapprox(dx, 0, atol=1e-10)
    @test cell1_face_y ≈ cell2_face_y atol=1e-3
    @test cell1_face_y ≈ cell2_face_y atol=1e-3
elseif isapprox(dy, 0, atol=1e-10)
    @test cell1_face_x ≈ cell2_face_x atol=1e-3
    @test cell1_face_x ≈ cell2_face_x atol=1e-3
end

uk, uv = u[ck, :], u[cv, :]
uk_face, uv_face = Rγk * uk, Rγv * uv
uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

face_to_face = topology.face_face_neighbours
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
face_interface = face_interfaces[1]
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


function match_coords_test(u::Vector{Coord{N, T}}, v::Vector{Coord{N, T}}, tol = 1e-13) where {N, T}
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


function is_two_faces_periodic_test(face1::NTuple{N, Int}, 
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



face_interfaces = compute_face_interfaces_periodic(cells, xyz_f, topology.face_face_neighbours, xL, xR, yB, yT, tol=1e-5)


function compute_topology_periodic_test(cells::Vector{C}; coords::Vector{Coord{dim, T}}, vector_of_nodeTags::Vector{Vector{Int}}, vector_of_nodeTagsMaster::Vector{Vector{Int}}) where {dim, C <: AbstractCell, T}

    @info "Computing periodic grid topology..."
    cell_to_vertex = vertices.(cells)
    vertex_to_cell = create_vertex_to_cell_table_periodic(cell_to_vertex, vector_of_nodeTags, vector_of_nodeTagsMaster)
    cell_to_cell, face_to_face = compute_face_neighbours_periodic(cells, coords, vertex_to_cell, vector_of_nodeTags, vector_of_nodeTagsMaster)
    return Topology(cell_to_vertex, vertex_to_cell, cell_to_cell, face_to_face)
end

function create_vertex_to_cell_table_periodic_test(cell_to_vertex::Vector{NTuple{N, Int}}, 
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

function compute_face_neighbours_periodic_test(cells::Vector{C},
                                        coords::Vector{Coord{dim, T}},
                                        vertex_to_cell::Vector{Set{Int}}, 
                                        vector_of_nodeTags::Vector{Vector{Int}}, 
                                        vector_of_nodeTagsMaster::Vector{Vector{Int}}) where {C <: AbstractCell, T, dim}

    
    _, max_faces, _ = compute_max_sizes(cells)
    face_to_face = Matrix{FaceIndex}(undef, length(cells), max_faces)
    fill!(face_to_face, FaceIndex(-1, -1))                            #TODO: Improve this
    cell_to_cell = Vector{Vector{SBPLite.CellIndex}}(undef, length(cells))
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
                SBPLite.add_face_neighbour_periodic!(face_to_face, cell, cell_id, cell_neighbour, 
                                                cell_neighbour_id, coords, vector_of_nodeTags, vector_of_nodeTagsMaster)
            end
        end
    end
    
    return cell_to_cell, face_to_face
end

elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.get_elements(dim)
if isempty(elem_types)
    error("No elements found in dimension $dim")
end
elem_node_tags = convert(Vector{Vector{Int64}}, elem_node_tags)

elem_name, _, _, _, _, _ = gmsh.model.mesh.get_element_properties(elem_types[1])
elements = SBPLite.get_cell_type(elem_name)[]

curvilinear_mapping = Vector{PolynomialCurvilinearMapping}()

i = 1
elem_type = elem_types[i]
node_tags = elem_node_tags[i]
elem_name, dim, order, n_nodes, comp_coords, _ = gmsh.model.mesh.get_element_properties(elem_type)
comp_coords, rescale, basis = SBPLite.rescale_coords(comp_coords, elem_name)
comp_coords = reshape(comp_coords, (dim, n_nodes)) |> Coords
cell = SBPLite.get_cell_type(elem_name)
cells = [cell(Tuple(node_tags[j:(j + (n_nodes - 1))]), ref_elems_data[elem_name])
            for j in 1:n_nodes:length(node_tags)]
for c in ProgressBar(cells)
    phys_coords = xyz_gmsh[collect(c.nodes)]
    mapping = get_polynomial_curvilinear_mapping(basis, comp_coords, phys_coords,
                                                    Int(order), rescale=rescale)
    push!(curvilinear_mapping, mapping)
end
append!(elements, cells)


grid = read_mesh(mesh_file, ref_elems_data, identity)


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
compute_max_sizes(cells)