# 在这个文档中我们思考如何把已经建立的grid，假如我更改了坐标，得到新的grid
# 如果点在边界上，或许我们可以把其中一个坐标作为变量输入进去，比如说如果在 t = tF 上
# 那就只改变 x 坐标，t 坐标不变

using Pkg

using LinearAlgebra, SparseArrays, Trixi, BlockArrays
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics        
using ArraysOfArrays 

include("../src/SBPLite.jl")
using .SBPLite
@load joinpath(@__DIR__, "lin_adv_grid__50.jld2") grid

cell_id = 1
cell1 = grid.cells[cell_id]

plot_SBP_elements(grid, cell_id)

vertices_IDS = vertices(grid.cells[cell_id]) |> collect
vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

# phys_coords = grid.xyz_gmsh[vertices_IDS] |> Coords
phys_coords = [SVector(vertice1[1], vertice1[2]),
               SVector(vertice2[1], vertice2[2]),
               SVector(vertice3[1], vertice3[2])]



function get_polynomial_curvilinear_mapping(::Type{P},
                                            comp_coords::Vector{Coord{N, T}},
                                            phys_coords::Vector{Coord{N, T}},
                                            degree::Int = 1;
                                            rescale::Bool = false) where {N, T,
                                                                          P <:
                                                                          AbstractPolynomialType}
    @assert length(comp_coords) == length(phys_coords)
    if (rescale)                        # Rescale from [-1, 1] to [0, 1]
        comp_coords = [(coord .+ 1) / 2 for coord in comp_coords]
    end
    feats = reduce(hcat, basis_functions(P, comp_coords, degree))' |> Matrix
    coeffs = Tuple(feats \ get_i_coordinates(phys_coords, i) for i in 1:N)
    return PolynomialCurvilinearMapping{N, T, P}(coeffs)
end
# comp_coords = grid.xyz_gmsh[vertices(grid.cells[cell_id]) |> collect]
comp_coords = [SVector(-1.0, -1.0),
               SVector(1.0, -1.0),
               SVector(-1.0, 1.0)]
SBPLite.basis_functions(MonomialType, comp_coords, 1)

feats = reduce(hcat, SBPLite.basis_functions(MonomialType, comp_coords, 1))' |> Matrix
feats \ get_i_coordinates(phys_coords, 1)
coeffs = Tuple(feats \ get_i_coordinates(phys_coords, i) for i in 1:2)

mutable struct newGrid{dim, C <: AbstractCell, T <: Real}
    cells::Vector{C}
    xyz_gmsh::Vector{Coord{dim, T}}
    xyz::VectorOfArrays{Coord{dim, T}, 1}
    xyz_q::VectorOfArrays{Coord{dim, T}, 1}
    xyz_f::VectorOfArrays{Coord{dim, T}, 1}
    mapping::Vector{PolynomialCurvilinearMapping}
    ref_elems_data::Dict{String, RefElemData}
    face_interfaces::Vector{FaceInterface}
    cell_sets::Dict{String, Set{SBPLite.CellIndex}}
    face_sets::Dict{String, Set{SBPLite.FaceIndex}}
    topology::SBPLite.Topology
    geometric_terms::SBPLite.GeometricTerms{T}
    VOL::Vector{NTuple{dim, Matrix{T}}}
    FAC::Vector{Any}
end

newgrid = newGrid(deepcopy(grid.cells),deepcopy(grid.xyz_gmsh),
                  deepcopy(grid.xyz), deepcopy(grid.xyz_q),
                  deepcopy(grid.xyz_f), deepcopy(grid.mapping),
                  deepcopy(grid.ref_elems_data), deepcopy(grid.face_interfaces),
                  deepcopy(grid.cell_sets), deepcopy(grid.face_sets),
                  grid.topology, grid.geometric_terms,
                  deepcopy(grid.VOL), deepcopy(grid.FAC))

newgrid.xyz_gmsh = map(coord -> coord.*1.2 , grid.xyz_gmsh)

Grid(cells, xyz_gmsh, mapping, ref_elems_data, 
        cell_sets = cell_sets, 
            face_sets = face_sets, 
                vector_of_nodeTags = vector_of_nodeTags,    
                    vector_of_nodeTagsMaster = vector_of_nodeTagsMaster, 
                        is_periodic = is_periodic)


"""
This is a test function to create a new grid based on the existing one.
    we will modify the xyz_gmsh coordinates by adding a small value to the x or y coordinate.
    to create a new grid, and see if the operators are still valid.
This function is not complete, we just consider the case of triangle for now
"""
function create_new_grid(grid::Grid{dim, C, T}; 
                        comp_coords::Vector{SVector{dim, T}} = 
                            [SVector{dim, T}(-1.0, -1.0),
                             SVector{dim, T}(1.0, -1.0),
                             SVector{dim, T}(-1.0, 1.0)],
                        basis = MonomialType, 
                        gmsh_order = 1) where {dim, C, T}

    new_xyz_gmsh = map(coord -> SVector(coord[1] * 1.1, coord[2] * 1.2), grid.xyz_gmsh)

    # 更新mapping
    curvilinear_mapping = Vector{PolynomialCurvilinearMapping}()

    for c in grid.cells
        phys_coords = new_xyz_gmsh[collect(c.nodes)]
        feats = reduce(hcat, SBPLite.basis_functions(basis, comp_coords, gmsh_order))' |> Matrix
        coeffs = Tuple(feats \ get_i_coordinates(phys_coords, i) for i in 1:dim)
        push!(curvilinear_mapping, PolynomialCurvilinearMapping{N, T, typeof(basis)}(coeffs))

    end

    return Grid(deepcopy(grid.cells), new_xyz_gmsh, curvilinear_mapping, deepcopy(grid.ref_elems_data),
                cell_sets = deepcopy(grid.cell_sets),
                face_sets = deepcopy(grid.face_sets),
                vector_of_nodeTags = [],
                vector_of_nodeTagsMaster = [],
                is_periodic = false)
end

comp_coords = reshape([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0], (2, 3)) |> Coords
new_xyz_gmsh = map(coord -> SVector(coord[1] * 1.1, coord[2] * 1.2), grid.xyz_gmsh)
curvilinear_mapping = Vector{PolynomialCurvilinearMapping}()
c1 = grid.cells[1]
phys_coords = new_xyz_gmsh[collect(c1.nodes)]
# mapping = get_polynomial_curvilinear_mapping(basis, comp_coords, phys_coords,
feats = reduce(hcat, SBPLite.basis_functions(MonomialType, comp_coords, 1))' |> Matrix
coeffs = Tuple(feats \ get_i_coordinates(phys_coords, i) for i in 1:2)

newgrid = create_new_grid(grid)

r1 = SVector(-1, -1)
r2 = r1 .+ 0.1
