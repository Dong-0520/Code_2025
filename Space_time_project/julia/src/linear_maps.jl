"""
    Simple linear operator to select/permute elements of a vector.
"""
struct SelectionMap{T <: Number} <: LinearMaps.LinearMap{T}
    idxs::Vector{Int}                    # selection indices
    inv_idxs::Tuple{Vararg{Vector{Int}}} # inverse selection indices (for transpose)

    function SelectionMap{T}(idxs::Vector{Int}, n_total::Int) where {T}
        return new{T}(idxs, Tuple(findall(j -> j == i, idxs) for i in 1:n_total))
    end
end

Base.size(R::SelectionMap) = (length(R.idxs), length(R.inv_idxs))

@inline function LinearAlgebra.mul!(y::AbstractVector, R::SelectionMap, x::AbstractVector)
    LinearMaps.check_dim_mul(y, R, x)
    y[:] = x[R.idxs]
    return y
end

@inline function LinearMaps._unsafe_mul!(y::AbstractVector,
                                         Rt::LinearMaps.TransposeMap{T, <:SelectionMap},
                                         x::AbstractVector) where {T}
    LinearMaps.check_dim_mul(y, Rt, x)
    (; inv_idxs) = Rt.lmap
    for i in eachindex(inv_idxs)
        y[i] = zero(T)
        for j in inv_idxs[i]
            @muladd y[i] = y[i] + x[j]
        end
    end
    return y
end
