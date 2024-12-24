const abstract_cells = Dict("Line 2" => Line,
                            "Triangle 3" => Triangle,
                            "Triangle 6" => Triangle2,
                            "Triangle 10" => Triangle3,
                            "Quadrilateral 4" => Quadrilateral,
                            "Quadrilateral 9" => Quadrilateral2,
                            "Quadrilateral 16" => Quadrilateral3,
                            "Tetrahedron 4" => Tetrahedron,
                            "Tetrahedron 10" => Tetrahedron2,
                            "Tetrahedron 20" => Tetrahedron3,
                            "Hexahedron 8" => Cube,
                            "Hexahedron 27" => Cube2,
                            "Hexahedron 64" => Cube3)

"""
Load cell type from dict if available, otherwise fallback to CellN type.
"""
function get_cell_type(key::String)
    get(abstract_cells, key) do
        base_name = split(key)[1]
        fallback_name = Symbol(base_name * "N")
        return eval(fallback_name)
    end
end

"""
Rescale coordinates to [-1, 1] for tensor product elements and [0, 1] for simplex elements.
Additionally, return a boolean indicating if polynomial_mapping needs to rescale the coordinates
and the type of polynomial basis to use.
"""
function rescale_coords(coords::AbstractArray{T, N}, elem_name::String) where {T, N}
    if any(occursin.(["Triangle", "Tetrahedron"], (elem_name,)))
        return 2 * coords .- 1.0, false, MonomialType
    elseif any(occursin.(["Quadrilateral", "Hexahedron"], (elem_name,)))
        return coords, true, BernsteinType
    end
end

"""
Extract node coordinates from the mesh file.

Returns:
    nodes: Vector{Coord}: Vector of `Coord` objects representing the nodes in the mesh.
"""
function extract_nodes()
    node_id, node_coords, _ = gmsh.model.mesh.get_nodes()
    dim = Int64(gmsh.model.get_dimension())
    return [Coord(node_coords[i:(i + (dim - 1))]) for i in 1:3:length(node_coords)]
end

"""
Extract elements from the mesh file for given dimension.

Returns:
    elements: Vector{AbstractCell}: Vector of AbstractCell objects representing the elements
              in the mesh.
    elem_tags: Vector{Int}: Vector of element tags.
"""
function extract_elems(dim::Int, coords::Vector{Coord{N, T}},
                       ref_elems_data::Dict{String, RefElemData}) where {N, T}
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.get_elements(dim)
    if isempty(elem_types)
        error("No elements found in dimension $dim")
    end
    elem_node_tags = convert(Vector{Vector{Int64}}, elem_node_tags)
    if length(elem_types) == 1
        elem_name, _, _, _, _, _ = gmsh.model.mesh.get_element_properties(elem_types[1])
        elements = get_cell_type(elem_name)[]
    else
        elements = AbstractCell[]
    end
    curvilinear_mapping = Vector{PolynomialCurvilinearMapping}()
    @info "Extracting elems and computing mappings..."
    for (i, elem_type) in enumerate(elem_types)
        node_tags = elem_node_tags[i]
        elem_name, dim, order, n_nodes, comp_coords, _ = gmsh.model.mesh.get_element_properties(elem_type)
        comp_coords, rescale, basis = rescale_coords(comp_coords, elem_name)
        comp_coords = reshape(comp_coords, (dim, n_nodes)) |> Coords
        cell = get_cell_type(elem_name)
        cells = [cell(Tuple(node_tags[j:(j + (n_nodes - 1))]), ref_elems_data[elem_name])
                 for j in 1:n_nodes:length(node_tags)]
        for c in ProgressBar(cells)
            phys_coords = coords[collect(c.nodes)]
            mapping = get_polynomial_curvilinear_mapping(basis, comp_coords, phys_coords,
                                                         Int(order), rescale=rescale)
            push!(curvilinear_mapping, mapping)
        end
        append!(elements, cells)
    end
    return elements, reduce(vcat, convert(Vector{Vector{Int64}}, elem_tags)), curvilinear_mapping
end

"""
Extract cell sets defined during mesh generation.

Returns:
    cell_sets: Dict{String, Set{Int}}: Dictionary of cell set names and their corresponding
               element indices.
"""
function extract_cell_sets(dim::Int, elem_tags::Vector{Int})
    cell_sets = Dict{String, Set{CellIndex}}()
    physical_groups = gmsh.model.get_physical_groups(dim)
    for (_, tag) in physical_groups
        name = gmsh.model.get_physical_name(dim, tag)
        isempty(name) ? (name = "$tag") : (name = name)
        entities = gmsh.model.get_entities_for_physical_group(dim, tag)
        cell_set_elems = Set{CellIndex}()
        for entity in entities
            _, group_elem_tags, _ = gmsh.model.mesh.get_elements(dim, entity)
            group_elem_tags = reduce(vcat, group_elem_tags) |> x -> convert(Vector{Int}, x)
            for elem in group_elem_tags
                push!(cell_set_elems, CellIndex(findfirst(isequal(elem), elem_tags)))
            end
            cell_sets[name] = cell_set_elems
        end
    end
    return cell_sets
end

"""
Extract nodes of boundary faces from the mesh file for given dimension.

Returns:
    boundary_nodes: Dict{String, Vector}: Dictionary of boundary names and their
                    corresponding face nodes.
"""
function extract_boundary_face_nodes(dim::Int)
    boundary_nodes = Dict{String, Vector}()
    boundaries = gmsh.model.get_physical_groups(dim)
    for (dim, tag) in boundaries
        name = gmsh.model.get_physical_name(dim, tag)
        boundary_entities = gmsh.model.get_entities_for_physical_group(dim, tag)
        boundary_connectivity = Tuple[]
        for entity in boundary_entities
            boundary_types, boundary_tags, boundary_node_tags = gmsh.model.mesh.get_elements(dim,
                                                                                             entity)
            _, _, _, n_nodes, _, _ = gmsh.model.mesh.get_element_properties(boundary_types[1])
            boundary_node_tags = convert(Vector{Vector{Int64}}, boundary_node_tags)[1]
            append!(boundary_connectivity,
                    [Tuple(boundary_node_tags[i:(i + (n_nodes - 1))])
                     for i in 1:n_nodes:length(boundary_node_tags)])
        end
        boundary_nodes[name] = boundary_connectivity
    end
    return boundary_nodes
end

"""
Helper function to add boundary faces to the face set tuple as FaceIndex objects.
"""
function add_to_face_set_tuple!(face_set_tuple::Set{FaceIndex}, boundary_face::Tuple,
                                element_faces)
    for (elem_idx, elem_face) in enumerate(element_faces)
        if any(issubset.(elem_face, (boundary_face,)))
            local_face = findfirst(x -> issubset(x, boundary_face), elem_face)
            push!(face_set_tuple, FaceIndex(elem_idx, local_face))
        end
    end
    return face_set_tuple
end

"""
Extract boundary faces from the mesh file for given dimension. The faces are represented
as FaceIndex objects.

Args:
    boundary_dict: Dict{String, Vector}: Dictionary of boundary names and their corresponding
                   face nodes.
    elements: Vector{AbstractCell}: Vector of cells.

Returns:
    boundary_faces: Dict{String, Set{FaceIndex}}: Dictionary of boundary names and their
                    faces along the corresponding boundary.
"""
function extract_boundary_faces(boundary_dict::Dict{String, Vector},
                                elements::Vector{<:AbstractCell})
    elem_faces = faces.(elements)
    face_sets = Dict{String, Set{FaceIndex}}()
    for (boundary_name, boundary_faces) in boundary_dict
        face_set_tuple = Set{FaceIndex}()
        for boundary_face in boundary_faces
            add_to_face_set_tuple!(face_set_tuple, boundary_face, elem_faces)
        end
        face_sets[boundary_name] = face_set_tuple
    end
    return face_sets
end

#-------------------------------------------------#
# try to implement periodic boundary properties
# Dong
function _extract_periodicNodes(face_dim, tags::Set{Int})

    vector_of_tags = collect(tags)
    vector_of_nodeTags = Vector{Vector{Int}}()
    vector_of_nodeTagsMaster = Vector{Vector{Int}}()

    for i in vector_of_tags
        _, nodeTags, nodeTagsMaster, _ = gmsh.model.mesh.getPeriodicNodes(face_dim, i, true)
        if !isempty(nodeTags)
            push!(vector_of_nodeTags, nodeTags)
            push!(vector_of_nodeTagsMaster, nodeTagsMaster)
        end
    end
    return vector_of_nodeTags, vector_of_nodeTagsMaster

end

# Dong
"""
如果periodic, lines的tags和 对应的master tags不一样。
如果一样, 则不满足周期性边界条件,返回false
"""
function is_mesh_periodic(face_dim::Int)
    boundaries = gmsh.model.get_physical_groups(face_dim)
    masterTags = Set{Int}()
    curve_tags = Set{Int}()
    for (_, physical_tag) in boundaries
        # name = gmsh.model.get_physical_name(face_dim, boundaries[i][2])
        curve_tag = gmsh.model.get_entities_for_physical_group(face_dim, physical_tag)[1]
        push!(curve_tags, curve_tag)
        push!(masterTags, gmsh.model.mesh.getPeriodic(face_dim, [curve_tag])[1])
    end
    return length(curve_tags) != length(masterTags), curve_tags
end
#-------------------------------------------------#


"""
Read gmsh mesh file and return Grid object.

Args:
    mesh_file: String: Path to the mesh file.

Returns:
    grid: Grid: Grid object representing the mesh.
"""
function read_mesh(mesh_file, ref_elems_data::Dict{String, RefElemData},
                   transform::Function = identity)
    if !isfile(mesh_file)
        error("Msh file not found: $mesh_file")
    end

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)                # Show output in terminal
    gmsh.open(mesh_file)
    gmsh.model.mesh.renumber_nodes()                            # Consecutive node and elem numbering
    gmsh.model.mesh.renumber_elements()
    dim = Int64(gmsh.model.get_dimension())
    face_dim = dim - 1

    xyz_gmsh = transform.(extract_nodes())                         #TODO: Think of long term soln.
    cells, elem_tags, mapping = extract_elems(dim, xyz_gmsh, ref_elems_data)
    cell_sets = extract_cell_sets(dim, elem_tags)

    boundary_dict = extract_boundary_face_nodes(face_dim)
    face_sets = extract_boundary_faces(boundary_dict, cells)

    #-------------------------------------------------#
    is_periodic, line_tags = is_mesh_periodic(face_dim)
    vector_of_nodeTags, vector_of_nodeTagsMaster = _extract_periodicNodes(face_dim, line_tags)
    #-------------------------------------------------#

    gmsh.clear()
    # gmsh.finalize()
    return Grid(cells, xyz_gmsh, mapping, ref_elems_data, 
                    cell_sets = cell_sets, 
                        face_sets = face_sets, 
                            vector_of_nodeTags = vector_of_nodeTags,    
                                vector_of_nodeTagsMaster = vector_of_nodeTagsMaster, 
                                    is_periodic = is_periodic)
end
