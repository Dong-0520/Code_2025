"""
Entity representing each physical unit in a grid.
"""
abstract type AbstractCell{refshape <: AbstractRefShape} end

struct Line <: AbstractCell{RefLine}
    nodes::NTuple{2, Int}
    ref_data::Ref{RefElemData}
end
struct Triangle <: AbstractCell{RefTriangle}
    nodes::NTuple{3, Int}
    ref_data::Ref{RefElemData}
end
struct Triangle2 <: AbstractCell{RefTriangle}
    nodes::NTuple{6, Int}
    ref_data::Ref{RefElemData}
end
struct Triangle3 <: AbstractCell{RefTriangle}
    nodes::NTuple{10, Int}
    ref_data::Ref{RefElemData}
end
struct TriangleN <: AbstractCell{RefTriangle}
    nodes::Tuple{Vararg{Int}}
    ref_data::Ref{RefElemData}
end
struct Quadrilateral <: AbstractCell{RefQuadrilateral}
    nodes::NTuple{4, Int}
    ref_data::Ref{RefElemData}
end
struct Quadrilateral2 <: AbstractCell{RefQuadrilateral}
    nodes::NTuple{9, Int}
    ref_data::Ref{RefElemData}
end
struct Quadrilateral3 <: AbstractCell{RefQuadrilateral}
    nodes::NTuple{16, Int}
    ref_data::Ref{RefElemData}
end
struct QuadrilateralN <: AbstractCell{RefQuadrilateral}
    nodes::Tuple{Vararg{Int}}
    ref_data::Ref{RefElemData}
end
struct Tetrahedron <: AbstractCell{RefTetrahedron}
    nodes::NTuple{4, Int}
    ref_data::Ref{RefElemData}
end
struct Tetrahedron2 <: AbstractCell{RefTetrahedron}
    nodes::NTuple{10, Int}
    ref_data::Ref{RefElemData}
end
struct Tetrahedron3 <: AbstractCell{RefTetrahedron}
    nodes::NTuple{20, Int}
    ref_data::Ref{RefElemData}
end
struct TetrahedronN <: AbstractCell{RefTetrahedron}
    nodes::Tuple{Vararg{Int}}
    ref_data::Ref{RefElemData}
end
struct Cube <: AbstractCell{RefCube}
    nodes::NTuple{8, Int}
    ref_data::Ref{RefElemData}
end
struct Cube2 <: AbstractCell{RefCube}
    nodes::NTuple{27, Int}
    ref_data::Ref{RefElemData}
end
struct Cube3 <: AbstractCell{RefCube}
    nodes::NTuple{64, Int}
    ref_data::Ref{RefElemData}
end
struct CubeN <: AbstractCell{RefCube}
    nodes::Tuple{Vararg{Int}}
    ref_data::Ref{RefElemData}
end

"""
Return tuple with node indices (w.r.t global grid) for each vertex in a cell.
"""
vertices(::AbstractCell)

"""
Return tuple of 2-tuples where each tuple contains ordered node indices (w.r.t. global grid) that
represents an edge. The order of node indices induces an edge orientation.
"""
edges(::AbstractCell)

"""
Return tuple of n-tuples containing locally ordered node indices corresponding to n-vertices that
define a shape.
"""
reference_faces(::AbstractRefShape)

"""
Return tuple of n-tuples containing ordered node indices (w.r.t global grid) corresponding to
n-vertices that define a face. The vertex ordering induces face orientation.
"""
faces(::AbstractCell)

function vertices(cell::AbstractCell{RefShape}) where {RefShape}
    nodes = get_nodes(cell)
    return map(i -> nodes[i], reference_vertices(RefShape))
end

function edges(cell::AbstractCell{RefShape}) where {RefShape}
    nodes = get_nodes(cell)
    return map(reference_edges(RefShape)) do re
        map(i -> nodes[i], re)
    end
end

function faces(cell::AbstractCell{RefShape}) where {RefShape}
    nodes = get_nodes(cell)
    return map(reference_faces(RefShape)) do rf
        map(i -> nodes[i], rf)
    end
end

@inline get_nodes(cell::AbstractCell) = cell.nodes
@inline n_vertices(cell::AbstractCell) = length(vertices(cell))
@inline n_edges(cell::AbstractCell) = length(edges(cell))
@inline n_faces(cell::AbstractCell) = length(faces(cell))
@inline n_nodes(cell::AbstractCell) = length(get_nodes(cell))
@inline ref_face_normal(cell::AbstractCell{RefShape}, face::Int) where {RefShape} = reference_normals(RefShape)[face]
