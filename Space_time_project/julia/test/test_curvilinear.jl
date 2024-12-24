include("../src/element_type.jl")
using .ElementType

using Transducers: Map
using Test

ATOL = 1e-12
_n_terms(n::Int, deg::Int) = binomial(deg + (n + 1) - 1, deg)
function test_poly_2D(x, y, deg)
    [1, x, y, x^2, x * y, y^2, x^3, x^2 * y, x * y^2, y^3, x^4, x^3 * y, x^2 * y^2, x * y^3,
        y^4][1:_n_terms(2, deg)]
end

function test_poly_2D_der_x(x, y, deg)
    [0, 1, 0, 2 * x, y, 0, 3 * x^2, 2 * x * y, y^2, 0, 4 * x^3, 3 * x^2 * y, 2 * x * y^2,
        y^3, 0][1:_n_terms(2, deg)]
end

function test_poly_2D_der_y(x, y, deg)
    [0, 0, 1, 0, x, 2 * y, 0, x^2, 2 * x * y, 3 * y^2, 0, x^3, 2 * x^2 * y, 3 * x * y^2,
        4 * y^3][1:_n_terms(2, deg)]
end

function test_poly_3D(x, y, z, deg)
    [1, x, y, z, x^2, x * y, x * z, y^2, y * z, z^2, x^3, x^2 * y, x^2 * z, x * y^2, x * y * z,
        x * z^2, y^3, y^2 * z, y * z^2, z^3, x^4, x^3 * y, x^3 * z, x^2 * y^2, x^2 * y * z,
        x^2 * z^2, x * y^3, x * y^2 * z, x * y * z^2, x * z^3, y^4, y^3 * z, y^2 * z^2, y * z^3, z^4
    ][1:_n_terms(3, deg)]
end

function test_poly_3D_der_x(x, y, z, deg)
    [0, 1, 0, 0, 2 * x, y, z, 0, 0, 0, 3 * x^2, 2 * x * y, 2 * x * z, y^2, y * z, z^2, 0, 0, 0, 0,
        4 * x^3, 3 * x^2 * y, 3 * x^2 * z, 2 * x * y^2, 2 * x * y * z, 2 * x * z^2, y^3, y^2 * z,
        y * z^2, z^3, 0, 0, 0, 0, 0][1:_n_terms(3, deg)]
end

function test_poly_3d_der_y(x, y, z, deg)
    [0, 0, 1, 0, 0, x, 0, 2 * y, z, 0, 0, x^2, 0, 2 * x * y, x * z, 0, 3 * y^2, 2 * y * z, z^2,
        0, 0, x^3, 0, 2 * x^2 * y, x^2 * z, 0, 3 * x * y^2, 2 * x * y * z, x * z^2, 0, 4 * y^3,
        3 * y^2 * z, 2 * y * z^2, z^3, 0][1:_n_terms(3, deg)]
end

function test_poly_3d_der_z(x, y, z, deg)
    [0, 0, 0, 1, 0, 0, x, 0, y, 2 * z, 0, 0, x^2, 0, x * y, 2 * x * z, 0, y^2, 2 * y * z, 3 * z^2,
        0, 0, x^3, 0, x^2 * y, 2 * x^2 * z, 0, x * y^2, 2 * x * y * z, 3 * x * z^2, 0, y^3,
        2 * y^2 * z, 3 * y * z^2, 4 * z^3][1:_n_terms(3, deg)]
end

@testset verbose=true "monomial_basis" begin
    degree_max = 4
    coords_2D = [rand(2) for _ in 1:5]
    coords_3D = [rand(3) for _ in 1:5]
    for coord in coords_2D
        for deg in 1:degree_max
            result = monomial_basis(coord, deg)
            @test length(result) == Int((deg + 1) * (deg + 2) / 2)
            @test isapprox(result, test_poly_2D(coord..., deg), atol = ATOL)
        end
    end
    for coord in coords_3D
        for deg in 1:degree_max
            result = monomial_basis(coord, deg)
            @test length(result) == Int((deg + 1) * (deg + 2) * (deg + 3) / 6)
            @test isapprox(result, test_poly_3D(coord..., deg), atol = ATOL)
        end
    end
end

@testset verbose=true "polyval_ND" begin
    degree_max = 4
    coords_2D = Coords([rand(2) for _ in 1:5])
    coords_3D = Coords([rand(3) for _ in 1:5])
    coeffs = rand(_n_terms(3, degree_max))
    expected_2D(x, y, deg) = sum(coeffs[1:_n_terms(2, deg)] .* test_poly_2D(x, y, deg))
    expected_3D(x, y, z, deg) = sum(coeffs[1:_n_terms(3, deg)] .* test_poly_3D(x, y, z, deg))
    for coord in coords_2D
        for deg in 1:degree_max
            result_1 = polyval_ND(coord, coeffs[1:_n_terms(2, deg)])
            expected = expected_2D(coord..., deg)
            @test isapprox(result_1, expected, atol = ATOL)
        end
    end
    for coord in coords_3D
        for deg in 1:degree_max
            result_1 = polyval_ND(coord, coeffs[1:_n_terms(3, deg)])
            expected = expected_3D(coord..., deg)
            @test isapprox(result_1, expected, atol = ATOL)
        end
    end
end

@testset verbose=true "polyder_2D" begin
    degree_max = 4
    coeffs = rand(_n_terms(2, degree_max))
    coords = Coords([rand(2) for _ in 1:5])
    for coord in coords
        for deg in 1:degree_max
            result = polyval_ND(coord, polyder_2D(coeffs[1:_n_terms(2, deg)], 1))
            expected = sum(test_poly_2D_der_x(coord..., deg) .* coeffs[1:_n_terms(2, deg)])
            @test isapprox(result, expected, atol = ATOL)

            result = polyval_ND(coord, polyder_2D(coeffs[1:_n_terms(2, deg)], 2))
            expected = sum(test_poly_2D_der_y(coord..., deg) .* coeffs[1:_n_terms(2, deg)])
            @test isapprox(result, expected, atol = ATOL)
        end
    end
end

@testset verbose=true "polyder_3D" begin
    degree_max = 4
    coeffs = rand(_n_terms(3, degree_max))
    coords = Coords([rand(3) for _ in 1:5])
    for coord in coords
        for deg in 1:degree_max
            result = polyval_ND(coord, polyder_3D(coeffs[1:_n_terms(3, deg)], 1))
            expected = sum(test_poly_3D_der_x(coord..., deg) .* coeffs[1:_n_terms(3, deg)])
            @test isapprox(result, expected, atol = ATOL)

            result = polyval_ND(coord, polyder_3D(coeffs[1:_n_terms(3, deg)], 2))
            expected = sum(test_poly_3d_der_y(coord..., deg) .* coeffs[1:_n_terms(3, deg)])
            @test isapprox(result, expected, atol = ATOL)

            result = polyval_ND(coord, polyder_3D(coeffs[1:_n_terms(3, deg)], 3))
            expected = sum(test_poly_3d_der_z(coord..., deg) .* coeffs[1:_n_terms(3, deg)])
            @test isapprox(result, expected, atol = ATOL)
        end
    end
end

@testset verbose=true "comp_to_phys" begin
    degree_max = 4                                  #TODO: How to check higher degrees?
    for _ in 1:10
        for deg in 1:degree_max
            n_points = _n_terms(2, deg)
            comp_coords = Coords([rand(2) for _ in 1:n_points])
            coeffs_x, coeffs_y = rand(n_points), rand(n_points)
            poly_coords = [test_poly_2D(coord..., deg) for coord in comp_coords]
            phys_x = poly_coords |> Map(coord -> sum(coord .* coeffs_x)) |> collect
            phys_y = poly_coords |> Map(coord -> sum(coord .* coeffs_y)) |> collect
            phys_coords = Coords([phys_x phys_y]')
            mapping = get_polynomial_curvilinear_mapping(comp_coords, phys_coords, deg)
            phys_approx = comp_to_phys(mapping, comp_coords)
            @test isapprox(phys_coords, phys_approx, atol = ATOL)
        end
    end
end
