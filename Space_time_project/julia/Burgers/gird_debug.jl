using LinearAlgebra, SparseArrays
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics                                                                                                                                                                                                                                                                          

include("../src/SBPLite.jl")
using .SBPLite
include("plotting_helper.jl")

order = 4
ref = TriangleDiagELGL(order, 2 * order)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)

using gmsh_jll
mesh_file = joinpath(@__DIR__, "2025_SIAM_pre", "convergence_study_sin", "Burgers_sin_12.msh")
@assert isfile(mesh_file)
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
topology = SBPLite.compute_topology_periodic(cells, coords = xyz_gmsh, vector_of_nodeTags = vector_of_nodeTags, vector_of_nodeTagsMaster = vector_of_nodeTagsMaster, vertices_of_domain = [xL, xR, yB, yT])

face_interfaces = compute_face_interfaces_periodic_test(cells, xyz_f, topology.face_face_neighbours, xL, xR, yB, yT, tol=1e-5)

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
tol = 1e-6
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
        p1, p2 = SBPLite.match_coords_periodic(u, v, c1 = c1, c2 = c2)
    else 
        p1, p2 = SBPLite.match_coords(u, v)
    end
    push!(interfaces, FaceInterface(face_interface[1], face_interface[2], p1, p2))
end

geometric_terms = SBPLite.GeometricTerms(VectorOfArrays.((J_q, Λ_q, J_f, Λ_f, N_f))...)


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