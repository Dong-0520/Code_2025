function _generate_2d_nodes!(coords, nx, ny, LL, LR, UR, UL)
    for i in 0:(ny - 1)
        ratio_bounds = i / (ny - 1)

        x0 = LL[1] * (1 - ratio_bounds) + ratio_bounds * UL[1]
        x1 = LR[1] * (1 - ratio_bounds) + ratio_bounds * UR[1]

        y0 = LL[2] * (1 - ratio_bounds) + ratio_bounds * UL[2]
        y1 = LR[2] * (1 - ratio_bounds) + ratio_bounds * UR[2]

        for j in 0:(nx - 1)
            ratio = j / (nx - 1)
            x = x0 * (1 - ratio) + ratio * x1
            y = y0 * (1 - ratio) + ratio * y1
            push!(coords, Coord((x, y)))
        end
    end
end

function generate_grid(::Type{Quadrilateral2},
                       ref_elems_data::Dict{DataType, RefElemData},
                       n_elem::NTuple{2, Int}, LL::NTuple{2, T}, LR::NTuple{2, T}, UR::NTuple{2, T},
                       UL::NTuple{2, T}) where {T}
    n_elem_x = n_elem[1]
    n_elem_y = n_elem[2]
    nel_tot = n_elem_x * n_elem_y
    n_nodes_x = 2 * n_elem_x + 1
    n_nodes_y = 2 * n_elem_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate coords
    coords = Coord{2, T}[]
    _generate_2d_nodes!(coords, n_nodes_x, n_nodes_y, LL, LR, UR, UL)

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    cells = QuadraticQuadrilateral[]
    for j in 1:n_elem_y
        for i in 1:n_elem_x
            push!(cells,
                  QuadraticQuadrilateral((node_array[2 * i - 1, 2 * j - 1],
                                          node_array[2 * i, 2 * j - 1],
                                          node_array[2 * i + 1, 2 * j - 1],
                                          node_array[2 * i - 1, 2 * j],
                                          node_array[2 * i, 2 * j],
                                          node_array[2 * i + 1, 2 * j],
                                          node_array[2 * i - 1, 2 * j + 1],
                                          node_array[2 * i, 2 * j + 1],
                                          node_array[2 * i + 1, 2 * j + 1])))
        end
    end

    # Cell face sets
    cell_array = reshape(collect(1:nel_tot), (n_elem_x, n_elem_y))
    boundary = FaceIndex[[FaceIndex(cl, 1) for cl in cell_array[:, 1]];
                         [FaceIndex(cl, 2) for cl in cell_array[end, :]];
                         [FaceIndex(cl, 3) for cl in cell_array[:, end]];
                         [FaceIndex(cl, 4) for cl in cell_array[1, :]]]
    offset = 0
    face_sets = Dict{String, Set{FaceIndex}}()
    face_sets["INFLOW"] = Set{FaceIndex}(boundary[(1:length(cell_array[:, 1])) .+ offset])
    offset += length(cell_array[:, 1])
    face_sets["right"] = Set{FaceIndex}(boundary[(1:length(cell_array[end, :])) .+ offset])
    offset += length(cell_array[end, :])
    face_sets["top"] = Set{FaceIndex}(boundary[(1:length(cell_array[:, end])) .+ offset])
    offset += length(cell_array[:, end])
    union!(face_sets["INFLOW"], Set{FaceIndex}(boundary[(1:length(cell_array[1, :])) .+ offset]))
    offset += length(cell_array[1, :])

    curvilinear_mapping = Vector{PolynomialCurvilinearMapping}()
    comp_coords = ref_coords(cells[1]) |> Coords
    for cell in cells
        phys_coords = coords[collect(get_nodes(cell))]
        mapping = get_polynomial_curvilinear_mapping(comp_coords, phys_coords)
        push!(curvilinear_mapping, mapping)
    end
    return Grid(cells, coords, curvilinear_mapping, ref_elems_data, face_sets = face_sets)
end
