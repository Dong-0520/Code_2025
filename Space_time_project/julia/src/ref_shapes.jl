"""
Supertype for all reference shapes.
"""
abstract type AbstractRefShape{ref_dim} end

# TODO: Add detailed documentation on these ref_types
struct RefLine <: AbstractRefShape{1} end
struct RefTriangle <: AbstractRefShape{2} end
struct RefQuadrilateral <: AbstractRefShape{2} end
struct RefTetrahedron <: AbstractRefShape{3} end
struct RefCube <: AbstractRefShape{3} end

# TODO: Add diagrams for each of these shapes
# RefLine (ref_dim = 1)
reference_vertices(::Type{RefLine}) = (1, 2)
reference_edges(::Type{RefLine}) = ((1, 2),)
reference_faces(::Type{RefLine}) = ((1,), (2,))

# RefTriangle (ref_dim = 2)
reference_vertices(::Type{RefTriangle}) = (1, 2, 3)
reference_edges(::Type{RefTriangle}) = ((1, 2), (2, 3), (3, 1))
reference_faces(::Type{RefTriangle}) = ((1, 2), (2, 3), (3, 1))
reference_normals(::Type{RefTriangle}) = ([0.0 -1.0], [1.0 1.0], [-1.0 0.0])

# RefQuadrilateral (ref_dim = 2)
reference_vertices(::Type{RefQuadrilateral}) = (1, 2, 3, 4)
reference_edges(::Type{RefQuadrilateral}) = ((1, 2), (2, 3), (3, 4), (4, 1))
reference_faces(::Type{RefQuadrilateral}) = ((1, 2), (2, 3), (3, 4), (4, 1))
reference_normals(::Type{RefQuadrilateral}) = ([0.0 -1.0], [1.0 0.0], [0.0 1.0], [-1.0 0.0])

# RefTetrahedron (ref_dim = 3)
reference_vertices(::Type{RefTetrahedron}) = (1, 2, 3, 4)
reference_edges(::Type{RefTetrahedron}) = ((1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4))
reference_faces(::Type{RefTetrahedron}) = ((1, 3, 2), (1, 2, 4), (2, 3, 4), (1, 4, 3))
function reference_normals(::Type{RefTetrahedron})
    ([0.0 0.0 -1.0], [0.0 -1.0 0.0], [1.0 1.0 1.0], [-1.0 0.0 0.0])
end

# RefCube (ref_dim = 3)
reference_vertices(::Type{RefCube}) = (1, 2, 3, 4, 5, 6, 7, 8)
function reference_edges(::Type{RefCube})
    ((1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (8, 5),
     (1, 5), (2, 6), (3, 7), (4, 8))
end
function reference_faces(::Type{RefCube})
    ((1, 4, 3, 2), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (1, 5, 8, 4),
     (5, 6, 7, 8))
end
function reference_normals(::Type{RefCube})
    ([0.0 0.0 -1.0], [0.0 -1.0 0.0], [1.0 0.0 0.0], [0.0 1.0 0.0], [-1.0 0.0 0.0], [0.0 0.0 1.0])
end

@inline n_vertices(::Type{T}) where {T <: AbstractRefShape} = length(reference_vertices(T))
@inline n_edges(::Type{T}) where {T <: AbstractRefShape} = length(reference_edges(T))
@inline n_faces(::Type{T}) where {T <: AbstractRefShape} = length(reference_faces(T))
@inline normal(::Type{T}, face::Int) where {T <: AbstractRefShape} = reference_normals(T)[face]
