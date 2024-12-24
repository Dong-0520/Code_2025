abstract type AbstractPolynomialType end
struct MonomialType <: AbstractPolynomialType end
struct BernsteinType <: AbstractPolynomialType end

struct PolynomialCurvilinearMapping{dim, T, P <: AbstractPolynomialType}
    params::NTuple{dim, Vector{T}}
end

function comp_to_phys(mapping::PolynomialCurvilinearMapping{dim, T, P},
                      coords::Vector{Coord{dim, T}}) where {dim, T, P}
    return Coord.(polyval.(P, coords, Ref(mapping.params)))
end

function comp_to_phys_deriv(mapping::PolynomialCurvilinearMapping{dim, T, P},
                            coords::Vector{Coord{dim, T}}) where {dim, T, P}
    J = zeros(T, length(coords), dim, dim)
    for i in 1:length(coords)
        jacobian!(@view(J[i, :, :]), c -> Coord(polyval(P, c, mapping.params)), coords[i])
    end
    return J
end

# function metric_terms_exact(mapping::PolynomialCurvilinearMapping{2, T, P},
#                             coords::Vector{Coord{2, T}}) where {T, P}
#     J = comp_to_phys_deriv(mapping, coords)
#     detJ = (J[:, 1, 1] .* J[:, 2, 2]) .- (J[:, 1, 2] .* J[:, 2, 1])
#     metric_terms = zeros(T, length(coords), 2, 2)
#     metric_terms[:, 1, 1] = J[:, 2, 2]
#     metric_terms[:, 2, 2] = J[:, 1, 1]
#     metric_terms[:, 1, 2] = -J[:, 1, 2]
#     metric_terms[:, 2, 1] = -J[:, 2, 1]
#     # metric_terms .*= sign.(detJ)                            #TODO: Check this
#     return metric_terms, detJ
# end

# function metric_terms_exact(mapping::PolynomialCurvilinearMapping{3, T, P},
#                             coords::Vector{Coord{3, T}}) where {T, P}
#     J = comp_to_phys_deriv(mapping, coords)
#     dr_dx = zeros(T, length(coords), 3, 3)
#     @. dr_dx[:, 1, 1] = J[:, 2, 2] * J[:, 3, 3] - J[:, 2, 3] * J[:, 3, 2]
#     @. dr_dx[:, 1, 2] = J[:, 1, 3] * J[:, 3, 2] - J[:, 1, 2] * J[:, 3, 3]
#     @. dr_dx[:, 1, 3] = J[:, 1, 2] * J[:, 2, 3] - J[:, 1, 3] * J[:, 2, 2]
#     @. dr_dx[:, 2, 1] = J[:, 2, 3] * J[:, 3, 1] - J[:, 2, 1] * J[:, 3, 3]
#     @. dr_dx[:, 2, 2] = J[:, 1, 1] * J[:, 3, 3] - J[:, 1, 3] * J[:, 3, 1]
#     @. dr_dx[:, 2, 3] = J[:, 1, 3] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 3]
#     @. dr_dx[:, 3, 1] = J[:, 2, 1] * J[:, 3, 2] - J[:, 2, 2] * J[:, 3, 1]
#     @. dr_dx[:, 3, 2] = J[:, 1, 2] * J[:, 3, 1] - J[:, 1, 1] * J[:, 3, 2]
#     @. dr_dx[:, 3, 3] = J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1]
#     detJ = [det(dr_dx[i, :, :]) for i in axes(dr_dx, 1)]
#     return dr_dx, detJ
# end

function metric_terms_optimised(mapping::PolynomialCurvilinearMapping{2, T, P},
                                coords::Vector{Coord{2, T}}, E::NTuple{2, Matrix{T}},
                                ref_elem::RefElemData) where {T, P}
    metric_terms_exact(mapping, coords)
end

# @inline function check_discrete_metric_invariance(Λ_q::Matrix{T}, Eone::Matrix{T},
#                                                   ref_elem::RefElemData) where {T}
#     LHS = hcat(ref_elem.D...) * Λ_q
#     RHS = ref_elem.H_inv * (hcat(ref_elem.E...) * Λ_q - Eone)
#     return isapprox(LHS, RHS, atol = 1e-11)
# end

# """
# Solve the optimisation problem to compute metric terms at vol. quadrature nodes that satisfy
# the discrete metric invariant property.
# """
# function metric_terms_optimised(mapping::PolynomialCurvilinearMapping{3, T, P},
#                                 coords::Vector{Coord{3, T}}, E::NTuple{3, Matrix{T}},
#                                 ref_elem::RefElemData) where {T, P}
#     dr_dx, _ = metric_terms_exact(mapping, coords)

#     # Eq. 34 in https://doi.org/10.1016/j.jcp.2017.12.015
#     dr_dx_out = zeros(T, size(dr_dx))
#     for d in 1:3
#         c = sum(E[d], dims = 2)
#         m_targ = reshape(@view(dr_dx[:, :, d]), :)
#         m = m_targ .- ref_elem.Qt_inv * (ref_elem.Qt * m_targ .- c)
#         @assert check_discrete_metric_invariance(m, c, ref_elem)
#         dr_dx_out[:, :, d] .= reshape(m, size(dr_dx_out[:, :, d]))
#     end
#     detJ = [det(dr_dx_out[i, :, :]) for i in axes(dr_dx_out, 1)]
#     return dr_dx_out, detJ
# end

function metric_terms_exact(mapping::PolynomialCurvilinearMapping{2, T},
                            coords::Vector{Coord{2, T}}) where {T}
    J = comp_to_phys_deriv(mapping, coords)
    detJ = (J[:, 1, 1] .* J[:, 2, 2]) .- (J[:, 1, 2] .* J[:, 2, 1])
    metric_terms = zeros(T, length(coords), 2, 2)
    metric_terms[:, 1, 1] = J[:, 2, 2]
    metric_terms[:, 2, 2] = J[:, 1, 1]
    metric_terms[:, 1, 2] = -J[:, 1, 2]
    metric_terms[:, 2, 1] = -J[:, 2, 1]
    metric_terms .*= sign.(detJ)                            #TODO: Check this
    return metric_terms, detJ
end

function metric_terms_exact(mapping::PolynomialCurvilinearMapping{3, T},
                            coords::Vector{Coord{3, T}}) where {T}
    J = comp_to_phys_deriv(mapping, coords)
    dr_dx = zeros(T, length(coords), 3, 3)
    @. dr_dx[:, 1, 1] = J[:, 2, 2] * J[:, 3, 3] - J[:, 2, 3] * J[:, 3, 2]
    @. dr_dx[:, 1, 2] = J[:, 1, 3] * J[:, 3, 2] - J[:, 1, 2] * J[:, 3, 3]
    @. dr_dx[:, 1, 3] = J[:, 1, 2] * J[:, 2, 3] - J[:, 1, 3] * J[:, 2, 2]
    @. dr_dx[:, 2, 1] = J[:, 2, 3] * J[:, 3, 1] - J[:, 2, 1] * J[:, 3, 3]
    @. dr_dx[:, 2, 2] = J[:, 1, 1] * J[:, 3, 3] - J[:, 1, 3] * J[:, 3, 1]
    @. dr_dx[:, 2, 3] = J[:, 1, 3] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 3]
    @. dr_dx[:, 3, 1] = J[:, 2, 1] * J[:, 3, 2] - J[:, 2, 2] * J[:, 3, 1]
    @. dr_dx[:, 3, 2] = J[:, 1, 2] * J[:, 3, 1] - J[:, 1, 1] * J[:, 3, 2]
    @. dr_dx[:, 3, 3] = J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1]
    detJ = [det(dr_dx[i, :, :]) for i in axes(dr_dx, 1)]
    return dr_dx, detJ
end

function metric_terms_optimised(mapping::PolynomialCurvilinearMapping{2, T},
                                coords::Vector{Coord{2, T}}, E::Array{T}, Qt::Matrix{T},
                                Qtinv::Matrix{T}) where {T}
    metric_terms_exact(mapping, coords)
end

"""
Solve the optimisation problem to compute metric terms at vol. quadrature nodes that satisfy
the discrete metric invariant property.
"""
function metric_terms_optimised(mapping::PolynomialCurvilinearMapping{3, T},
                                coords::Vector{Coord{3, T}}, E::Array{T}, Qt::Matrix{T},
                                Qtinv::Matrix{T}) where {T}
    dr_dx, _ = metric_terms_exact(mapping, coords)

    # Eq. 34 in https://doi.org/10.1016/j.jcp.2017.12.015
    dr_dx_out = zeros(T, size(dr_dx))
    for m in 1:3
        c = sum(E[:, :, m], dims = 2)
        @assert isapprox(sum(c), 0.0, atol = 100 * eps(T))
        m_targ = reshape(@view(dr_dx[:, :, m]), :)
        dr_dx_out[:, :, m] .= reshape(m_targ .- Qtinv * (Qt * m_targ .- c),
                                      size(dr_dx_out[:, :, m]))
    end
    detJ = [det(dr_dx_out[i, :, :]) for i in axes(dr_dx_out, 1)]
    return dr_dx_out, detJ
end

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

basis_functions(::Type{MonomialType}, coords, degree) = monomial_basis.(coords, degree)
basis_functions(::Type{BernsteinType}, coords, degree) = bernstein_basis.(coords, degree)
polyval(::Type{MonomialType}, coord, params) = polyval_monomial(coord, params)
function polyval(::Type{BernsteinType}, coord, params, rescale = true)
    polyval_bernstein(coord, params, rescale)
end

@inline bernstein(n::Int, i::Int, x::T) where {T} = binomial(n, i) * x^i * (1 - x)^(n - i)
@inline bernstein_poly(n::Int, x::T) where {T} = [bernstein(n, i, x) for i in 0:n]
"""
Tensor product Bernstein polynomial basis with highest degree `deg` in each dimension.
"""
@inline function bernstein_basis(coord::Coord{N, T}, deg::Int) where {N, T}
    reduce(kron, bernstein_poly.(deg, coord))
end

"""
Evaluate univariate Bernstein polynomial at given coordinates. Rescale from [-1, 1] to [0, 1] if needed.
"""
function polyval_bernstein(coord::Coord{N, T1},
                           coeffs::Union{NTuple{N, Vector{T2}}, Vector{Vector{T2}}},
                           rescale) where {N, T1, T2}
    if (rescale)
        basis = bernstein_basis((coord .+ 1) / 2, Int(sqrt(length(coeffs[1])) - 1))
    else
        basis = bernstein_basis(coord, Int(sqrt(length(coeffs[1])) - 1))
    end
    return map(c -> sum(basis .* c), coeffs)
end

"""
Return vector of monomial terms of given `degree` of arbitrary no. of variables. The order of terms
is lexicographic in the following sense:
[1, x, y, z, x^2, xy, xz, y^2, yz, z^2, x^3, x^2y, x^2z, xy^2, xyz, xz^2, y^3, ...]
"""
function monomial_basis(coord::Union{Vector{T}, NTuple{N, T}, Coord{N, T}},
                        degree::Int) where {N, T}
    exponents = multiexponents(length(coord) + 1, degree)       # Add 1 for bias term
    _x = [one(T), coord...]
    basis = exponents |> Map(exponent -> prod(_x .^ exponent)) |> collect
    return basis
end

@doc raw"""
Evaluate an N-dim polynomial of the form:
``P(x_1, x_2, \cdots, x_n) = \begin{bmatrix}{P_1(x_1, x_2, \cdots, x_n) \\ P_2(x_1, x_2, \cdots, x_n)
                             \\ \vdots \\ P_M(x_1, x_2, \cdots, x_n)}
                             \end{bmatrix}``
"""
function polyval_monomial(coord::Coord{N, T1},
                          coeffs::Union{NTuple{N, Vector{T2}}, Vector{Vector{T2}}}) where {T1, T2, N
                                                                                           }
    basis = monomial_basis(coord, _get_degree(length(coeffs[1]), N))
    return map(c -> sum(basis .* c), coeffs)
end

function _get_degree(n_terms::Int, dim::Int)
    if dim == 2
        return Int((-3 + sqrt(8 * n_terms + 1)) / 2)
    elseif dim == 3
        for i in 1:20
            if 6 * n_terms == (i + 1) * (i + 2) * (i + 3)
                return i
            end
        end
    else
        error("Invalid dimension. Only dims 2 and 3 supported.")
    end
end
