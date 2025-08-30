# gmsh 中的 三角形的ref elem的坐标是 (0,0) (1, 0), (0, 1)
# 所以在SBPLite中的 rescale_coords 中，先把这个 给map到我们SBP 的ref elem上了
# elem_type 是interger，从gmsh传过来的数字，在gmsh中， 2 就代表 Triangle 3
# 用 gmsh.model.mesh.get_element_properties(elem_type) 可以得到
# elem_name, dim, order, _, comp_coords, _ 
# elem_name就是 "Triangle 3" 这样的字符串， dim = 2, order = 1, n_nodes 是 在gmsh中三角形的点数量
# 此时的comp coords 是 gmsh的ref elem的coords
# 如果是 三角形或者Tetrahedron， 需要rescale 到我们SBP的ref elem上 rescale_coords 方程在 load_mesh.jl 中
# 通常我们用三角形，所以 rescale_coords 会得到如下结果：
# ([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0], false, MonomialType)


using Logging
using ProgressBars
using Combinatorics: multiexponents
using Transducers: Map
using StaticArrays
using ArraysOfArrays
using LinearMaps
using MuladdMacro
using ForwardDiff: jacobian!
using LinearAlgebra:
                     LinearAlgebra, lu, diagm, I, inv, norm, Diagonal, det, pinv
using SummationByParts:
                        SummationByParts, Cubature, getLineSegSBPLegendre, getLineSegSBPLobbato,
                        SymCubatures, getTriSBPDiagE, getLineSegFace, TriFace, getTriSBPOmega,
                        getTetSBPDiagE, TetFace, getTriCubatureForTetFaceDiagE
using gmsh_jll
include(gmsh_jll.gmsh_api)

SBP_order = 2
ref = TriangleDiagELG(SBP_order, 2 * SBP_order)
ref_elems_data = Dict{String, RefElemData}("Triangle 3" => ref)

include("../src/exports.jl")

"""
`Coord` is a point in space.
"""
Coord{N, T} = SVector{N, T}

include("../src/sbp.jl")
include("../src/linear_maps.jl")
include("../src/ref_shapes.jl")
include("../src/ref_elem_data.jl")
include("../src/ref_cells.jl")
include("../src/curvilinear.jl")
include("../src/geometry.jl")
include("../src/load_mesh.jl")
include("../src/grid_generators.jl")

mesh_file = joinpath(@__DIR__, "lin_adv_grid__50.msh")

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

xyz_gmsh = extract_nodes()
# cells, elem_tags, mapping = extract_elems(dim, xyz_gmsh, ref_elems_data)

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

i, elem_type = 1, 2
node_tags = elem_node_tags[i]
elem_name, dim, order, _, comp_coords, _ = gmsh.model.mesh.get_element_properties(elem_type)
comp_coords, rescale, basis = rescale_coords(comp_coords, elem_name)
comp_coords = reshape(comp_coords, (dim, 3)) |> Coords
cell = get_cell_type(elem_name)
cells = [cell(Tuple(node_tags[j:(j + (3 - 1))]), ref_elems_data[elem_name])
            for j in 1:3:length(node_tags)]
c1 = cells[1]
phys_coords = xyz_gmsh[collect(c1.nodes)]
mapping = (basis, comp_coords, phys_coords, Int(order), rescale=rescale)

comp_to_phys(mapping, ref.rst |> Coords)


xyz[i] = comp_to_phys(mapping[i], ref_elem.rsget_polynomial_curvilinear_mappingt |> Coords)
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

for c in ProgressBar(cells)
    phys_coords = coords[collect(c.nodes)]
    mapping = get_polynomial_curvilinear_mapping(basis, comp_coords, phys_coords,
                                                    Int(order), rescale=rescale)
    push!(curvilinear_mapping, mapping)
end
append!(elements, cells)

return elements, reduce(vcat, convert(Vector{Vector{Int64}}, elem_tags)), curvilinear_mapping









cell_sets = extract_cell_sets(dim, elem_tags)


