struct PolynomialCurvilinearMapping{dim, T}
    params::NTuple{dim, Vector{T}}
    params_deriv::NTuple{dim, NTuple{dim, Vector{T}}}
    function PolynomialCurvilinearMapping(coeffs::Matrix{T}) where {T <: Real}
        dim = size(coeffs, 2)
        params = tuple([collect(col) for col in eachcol(coeffs)]...)
        params_deriv = []
        for (i, param) in enumerate(params)
            push!(params_deriv, tuple([polyder(param, j, dim) for j in 1:dim]...))
        end
        return new{dim, T}(params, (params_deriv...,))
    end
end

function comp_to_phys_deriv(mapping::PolynomialCurvilinearMapping{2, T},
                            coords::Vector{Coord{2, T}}) where {T}
    dx_drs = polyval_monomial.(coords, (mapping.params_deriv[1],))
    dy_drs = polyval_monomial.(coords, (mapping.params_deriv[2],))
    jacobian_matrix = Array{Float64}(undef, length(coords), 2, 2)
    for i in 1:length(coords)
        jacobian_matrix[i, 1, 1] = dx_drs[i][1]
        jacobian_matrix[i, 1, 2] = dx_drs[i][2]
        jacobian_matrix[i, 2, 1] = dy_drs[i][1]
        jacobian_matrix[i, 2, 2] = dy_drs[i][2]
    end
    return jacobian_matrix
end

function comp_to_phys_deriv(mapping::PolynomialCurvilinearMapping{3, T},
                            coords::Vector{Coord{3, T}}) where {T}
    dx_dr = polyval_monomial(coords, mapping.params_deriv[1][1])
    dx_ds = polyval_monomial(coords, mapping.params_deriv[1][2])
    dx_dt = polyval_monomial(coords, mapping.params_deriv[1][3])
    dy_dr = polyval_monomial(coords, mapping.params_deriv[2][1])
    dy_ds = polyval_monomial(coords, mapping.params_deriv[2][2])
    dy_dt = polyval_monomial(coords, mapping.params_deriv[2][3])
    dz_dr = polyval_monomial(coords, mapping.params_deriv[3][1])
    dz_ds = polyval_monomial(coords, mapping.params_deriv[3][2])
    dz_dt = polyval_monomial(coords, mapping.params_deriv[3][3])
    jacobian_matrix = reshape(hcat(dx_dr, dy_dr, dz_dr, dx_ds, dy_ds, dz_ds, dx_dt, dy_dt, dz_dt),
                              (:, 3, 3))
    return jacobian_matrix
end

function polyder(coeffs::Vector{T}, axis::Int, dim::Int) where {T}
    if dim == 2
        return polyder_2D(coeffs, axis)
    elseif dim == 3
        return polyder_3D(coeffs, axis)
    else
        error("Invalid dimension. Only dims 2 and 3 supported.")
    end
end

function polyder_2D(coeffs::Vector{T}, axis::Int) where {T}
    degree = _get_degree(length(coeffs), 2)
    coeff_matrix = _lexicographic_coeff_vector_to_coeff_mat(coeffs, 2)
    coeff_matrix_deriv = zeros(T, degree + 1, degree + 1)
    if axis == 1
        coeff_matrix_deriv[1:(end - 1), :] .= coeff_matrix[2:end, :]
        for i in 1:degree
            coeff_matrix_deriv[i, :] *= i
        end
        coeff_matrix_deriv[end, :] .= 0.0
    elseif axis == 2
        coeff_matrix_deriv[:, 1:(end - 1)] .= coeff_matrix[:, 2:end]
        for i in 1:degree
            coeff_matrix_deriv[:, i] *= i
        end
        coeff_matrix_deriv[:, end] .= 0.0
    else
        error("Invalid axis")
    end
    return _lexicographic_coeff_mat_to_coeff_vector(coeff_matrix_deriv)
end

function polyder_3D(coeffs::Vector{T}, axis::Int) where {T}
    degree = _get_degree(length(coeffs), 3)
    coeff_matrix = _lexicographic_coeff_vector_to_coeff_mat(coeffs, 3)
    coeff_matrix_deriv = zeros(T, degree + 1, degree + 1, degree + 1)
    if axis == 1
        coeff_matrix_deriv[1:(end - 1), :, :] .= coeff_matrix[2:end, :, :]
        for i in 1:degree
            coeff_matrix_deriv[i, :, :] *= i
        end
        coeff_matrix_deriv[end, :, :] .= 0.0
    elseif axis == 2
        coeff_matrix_deriv[:, 1:(end - 1), :] .= coeff_matrix[:, 2:end, :]
        for i in 1:degree
            coeff_matrix_deriv[:, i, :] *= i
        end
        coeff_matrix_deriv[:, end, :] .= 0.0
    elseif axis == 3
        coeff_matrix_deriv[:, :, 1:(end - 1)] .= coeff_matrix[:, :, 2:end]
        for i in 1:degree
            coeff_matrix_deriv[:, :, i] *= i
        end
        coeff_matrix_deriv[:, :, end] .= 0.0
    else
        error("Invalid axis")
    end
    return _lexicographic_coeff_mat_to_coeff_vector(coeff_matrix_deriv)
end

"""
[i+1,j+1,k+1]'th elemment of the matrix is coefficient of x^i * y^j * z^k in the monomial.
"""
function _lexicographic_coeff_vector_to_coeff_mat(coeffs::Vector{T}, dim::Int) where {T}
    degree = _get_degree(length(coeffs), dim)
    out = zeros(T, (degree + 1) * ones(Int, dim)...)
    idxs = multiexponents(dim + 1, degree) |> collect
    for i in 1:length(idxs)
        idx = idxs[i][2:end] .+ 1
        out[idx...] = coeffs[i]
    end
    return out
end

function _lexicographic_coeff_mat_to_coeff_vector(coeffs::Array{T, dim}) where {T, dim}
    degree = size(coeffs, 1) - 1
    n_terms = _n_terms(dim, degree)
    out = zeros(T, n_terms)
    idxs = multiexponents(dim + 1, degree) |> collect
    for i in 1:length(idxs)
        idx = idxs[i][2:end] .+ 1
        out[i] = coeffs[idx...]
    end
    return out
end
