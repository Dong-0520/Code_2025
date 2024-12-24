module SBPLite

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

include("exports.jl")

"""
`Coord` is a point in space.
"""
Coord{N, T} = SVector{N, T}

include("sbp.jl")
include("linear_maps.jl")
include("ref_shapes.jl")
include("ref_elem_data.jl")
include("ref_cells.jl")
include("curvilinear.jl")
include("geometry.jl")
include("load_mesh.jl")
include("grid_generators.jl")

end
