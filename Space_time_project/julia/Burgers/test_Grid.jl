using Test
@testset "polynomial exactness" begin
    for i in 1:n_cells(grid)
        tol = 1e-10
        coordinates = grid.xyz_q[i]
        x = get_i_coordinates(coordinates, 1)
        t = get_i_coordinates(coordinates, 2)
        y = (x.^(order - 1)) .* t
        dydx = (order - 1) .* (x .^ (order - 2)) .* t
        dydt = x.^(order - 1)
        Dx = grid.VOL[i][1]
        Dt = grid.VOL[i][2]
        @testset "cell $i" begin
            @test isapprox(maximum(abs.(Dx * y - dydx )), 0, atol = tol) || println("Failed at cell $i for Dx")
            @test isapprox(maximum(abs.(Dt * y - dydt )), 0, atol = tol) || println("Failed at cell $i for Dt")
        end
    end
end


"""
This was testing if the Vyom's new implementation of the face fluxes, grid.FAC[cv][lfγv] * Diagonal(Ntγv),
is the same term as my previous implementation, (inv(Jv) * ref.H_inv) * Rγv' * (Ntγv  .* ref.H_face).
note that now we no longer multiply J on both side ( like before )
everything is now on physical domain
"""

@testset begin
    for i in eachindex(grid.face_interfaces)
        interface = grid.face_interfaces[i]
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        Rγk, Rγv = Matrix(ref.R[lfγk]), Matrix(ref.R[lfγv])


        # normal vector of physical element
        maskγk, maskγv = @views ref.f_mask[lfγk], ref.f_mask[lfγv]
        normalγk, normalγv = @views grid.geometric_terms.N_f[ck][:, maskγk], grid.geometric_terms.N_f[cv][:, maskγv]
        Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
        Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]

        Jk = Diagonal(abs.(grid.geometric_terms.J_q[ck]))
        Jv = Diagonal(abs.(grid.geometric_terms.J_q[cv]))
        @test isapprox(maximum(abs.(grid.FAC[ck][lfγk] * Diagonal(Nxγk) .- (inv(Jk) * ref.H_inv) * Rγk' * (Nxγk  .* ref.H_face))), 0, atol = 1e-12)
        @test isapprox(maximum(abs.(grid.FAC[ck][lfγk] * Diagonal(Ntγk) .- (inv(Jk) * ref.H_inv) * Rγk' * (Ntγk  .* ref.H_face))), 0, atol = 1e-12)
        @test isapprox(maximum(abs.(grid.FAC[cv][lfγv] * Diagonal(Nxγv) .- (inv(Jv) * ref.H_inv) * Rγv' * (Nxγv  .* ref.H_face))), 0, atol = 1e-12)
        @test isapprox(maximum(abs.(grid.FAC[cv][lfγv] * Diagonal(Ntγv) .- (inv(Jv) * ref.H_inv) * Rγv' * (Ntγv  .* ref.H_face))), 0, atol = 1e-12)

    end
end

@testset "test (-1,-1)" begin
    for i in 1:n_cells(grid)
        @test length(grid.topology.face_face_neighbours[i,:]) == 4
        for j in 1:4
            @test grid.topology.face_face_neighbours[i,j] != FaceIndex((-1,-1))
        end
    end
end

@testset "check correctness of P1, P2 for periodicity functionality, check coordinate matchs" begin
    for i in 1:length(grid.face_interfaces)
        interface = grid.face_interfaces[i]
        c1, lf1 = interface.face_1
        c2, lf2 = interface.face_2
        P1, P2 = interface.P1, interface.P2
        elem1, elem2 = grid.cells[c1].ref_data[], grid.cells[c2].ref_data[]
        R_1, R_2 = elem1.R[lf1], elem2.R[lf2]
        cell1_face_x = R_1 * SBPLite.get_i_coordinates(grid.xyz[c1], 1)
        cell1_face_y = R_1 * SBPLite.get_i_coordinates(grid.xyz[c1], 2)
        cell2_face_x = R_2 * SBPLite.get_i_coordinates(grid.xyz[c2], 1)
        cell2_face_y = R_2 * SBPLite.get_i_coordinates(grid.xyz[c2], 2)
        dx = abs(cell1_face_x[end] - cell1_face_x[1])
        dy = abs(cell1_face_y[end] - cell1_face_y[1])
        if dx > 1e-3 && dy > 1e-3
            @test cell1_face_x ≈ cell2_face_x[P2] atol=1e-3
            @test cell1_face_y ≈ cell2_face_y[P2] atol=1e-3
            @test cell1_face_x[P1] ≈ cell2_face_x atol=1e-3
            @test cell1_face_y[P1] ≈ cell2_face_y atol=1e-3
        elseif dx ≈ 0
            @test cell1_face_y ≈ cell2_face_y[P2] atol=1e-3
            @test cell1_face_y[P1] ≈ cell2_face_y atol=1e-3
        elseif dy ≈ 0
            @test cell1_face_x ≈ cell2_face_x[P2] atol=1e-3
            @test cell1_face_x[P1] ≈ cell2_face_x atol=1e-3
        end
    end
end


"""
for example, I have two elements

    3 6 9 | 7 8 9
    2 5 8 | 4 5 6
    1 4 7 | 1 2 3

    R3 * left = 7 8 9
    R4 * right = 1 4 7 

    always in a correct order
"""

@testset "check correctness of P1, P2 for periodicity functionality, check coordinate matchs for quadraleteral element" begin
    for i in 1:length(grid.face_interfaces)
        interface = grid.face_interfaces[i]
        c1, lf1 = interface.face_1
        c2, lf2 = interface.face_2
        P1, P2 = interface.P1, interface.P2
        elem1, elem2 = grid.cells[c1].ref_data[], grid.cells[c2].ref_data[]
        R_1, R_2 = elem1.R[lf1], elem2.R[lf2]
        cell1_face_x = R_1 * SBPLite.get_i_coordinates(grid.xyz[c1], 1)
        cell1_face_y = R_1 * SBPLite.get_i_coordinates(grid.xyz[c1], 2)
        cell2_face_x = R_2 * SBPLite.get_i_coordinates(grid.xyz[c2], 1)
        cell2_face_y = R_2 * SBPLite.get_i_coordinates(grid.xyz[c2], 2)
        dx = abs(cell1_face_x[end] - cell1_face_x[1])
        dy = abs(cell1_face_y[end] - cell1_face_y[1])
        if dx > 1e-3 && dy > 1e-3
            @test cell1_face_x ≈ cell2_face_x atol=1e-3
            @test cell1_face_y ≈ cell2_face_y atol=1e-3
        elseif dx ≈ 0
            @test cell1_face_y ≈ cell2_face_y atol=1e-3
        elseif dy ≈ 0
            @test cell1_face_x ≈ cell2_face_x atol=1e-3
        end
    end
end



@testset "check correctness of P1, P2 for periodicity functionality, check coordinate matchs, for boundary only" begin
    for i in get_face_set(grid, "TOP_INFLOW_1")
        
        cell_id, face_id = i
        neighbors_cells = map(x -> x.idx[1], grid.topology.face_face_neighbours[cell_id,:])
        neighbor_id, neighbor_face_id = -1, -1
        for (j1, j2) in get_face_set(grid, "BOTTOM_INFLOW_1")
            if j1 in neighbors_cells
                neighbor_id = j1
                neighbor_face_id = j2
            end
        end
        interface = find_FaceInterface(grid, i, FaceIndex((neighbor_id, neighbor_face_id)))
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        elem = grid.cells[ck].ref_data[]
        Rγk, Rγv = Matrix(elem.R[lfγk]), Matrix(elem.R[lfγv])
    
    
        ck_face_x = Rγk * get_i_coordinates(grid.xyz[ck], 1)
        cv_face_x = Rγv * get_i_coordinates(grid.xyz[cv], 1)
        ck_face_y = Rγk * get_i_coordinates(grid.xyz[ck], 2)
        cv_face_y = Rγv * get_i_coordinates(grid.xyz[cv], 2)
        @test ck_face_x ≈ cv_face_x[Pv] atol=1e-6
        @test ck_face_x[Pk] ≈ cv_face_x atol=1e-6
        @test all(x -> isapprox(x, ck_face_y[1], atol = 1e-6), ck_face_y)
        @test all(x -> isapprox(x, cv_face_y[1], atol = 1e-6), cv_face_y)
    end

    for i in get_face_set(grid, "TOP_INFLOW_2")
        
        cell_id, face_id = i
        neighbors_cells = map(x -> x.idx[1], grid.topology.face_face_neighbours[cell_id,:])
        neighbor_id, neighbor_face_id = -1, -1
        for (j1, j2) in get_face_set(grid, "BOTTOM_INFLOW_2")
            if j1 in neighbors_cells
                neighbor_id = j1
                neighbor_face_id = j2
            end
        end
        interface = find_FaceInterface(grid, i, FaceIndex((neighbor_id, neighbor_face_id)))
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        elem = grid.cells[ck].ref_data[]
        Rγk, Rγv = Matrix(elem.R[lfγk]), Matrix(elem.R[lfγv])
    
    
        ck_face_x = Rγk * get_i_coordinates(grid.xyz[ck], 1)
        cv_face_x = Rγv * get_i_coordinates(grid.xyz[cv], 1)
        ck_face_y = Rγk * get_i_coordinates(grid.xyz[ck], 2)
        cv_face_y = Rγv * get_i_coordinates(grid.xyz[cv], 2)
        @test ck_face_x ≈ cv_face_x[Pv] atol=1e-6
        @test ck_face_x[Pk] ≈ cv_face_x atol=1e-6
        @test all(x -> isapprox(x, ck_face_y[1], atol = 1e-6), ck_face_y)
        @test all(x -> isapprox(x, cv_face_y[1], atol = 1e-6), cv_face_y)
    end

    for i in get_face_set(grid, "BOTTOM_INFLOW_1") 
        cell_id, face_id = i
        neighbors_cells = map(x -> x.idx[1], grid.topology.face_face_neighbours[cell_id,:])
        neighbor_id, neighbor_face_id = -1, -1
        for (j1, j2) in get_face_set(grid, "TOP_INFLOW_1")
            if j1 in neighbors_cells
                neighbor_id = j1
                neighbor_face_id = j2
            end
        end
        interface = find_FaceInterface(grid, i, FaceIndex((neighbor_id, neighbor_face_id)))
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        elem = grid.cells[ck].ref_data[]
        Rγk, Rγv = Matrix(elem.R[lfγk]), Matrix(elem.R[lfγv])
    
    
        ck_face_x = Rγk * get_i_coordinates(grid.xyz[ck], 1)
        cv_face_x = Rγv * get_i_coordinates(grid.xyz[cv], 1)
        ck_face_y = Rγk * get_i_coordinates(grid.xyz[ck], 2)
        cv_face_y = Rγv * get_i_coordinates(grid.xyz[cv], 2)
        @test ck_face_x ≈ cv_face_x[Pv] atol=1e-6
        @test ck_face_x[Pk] ≈ cv_face_x atol=1e-6
        @test all(x -> isapprox(x, ck_face_y[1], atol = 1e-6), ck_face_y)
        @test all(x -> isapprox(x, cv_face_y[1], atol = 1e-6), cv_face_y)
    end

    for i in get_face_set(grid, "BOTTOM_INFLOW_2") 
        cell_id, face_id = i
        neighbors_cells = map(x -> x.idx[1], grid.topology.face_face_neighbours[cell_id,:])
        neighbor_id, neighbor_face_id = -1, -1
        for (j1, j2) in get_face_set(grid, "TOP_INFLOW_2")
            if j1 in neighbors_cells
                neighbor_id = j1
                neighbor_face_id = j2
            end
        end
        interface = find_FaceInterface(grid, i, FaceIndex((neighbor_id, neighbor_face_id)))
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        elem = grid.cells[ck].ref_data[]
        Rγk, Rγv = Matrix(elem.R[lfγk]), Matrix(elem.R[lfγv])
    
    
        ck_face_x = Rγk * get_i_coordinates(grid.xyz[ck], 1)
        cv_face_x = Rγv * get_i_coordinates(grid.xyz[cv], 1)
        ck_face_y = Rγk * get_i_coordinates(grid.xyz[ck], 2)
        cv_face_y = Rγv * get_i_coordinates(grid.xyz[cv], 2)
        @test ck_face_x ≈ cv_face_x[Pv] atol=1e-6
        @test ck_face_x[Pk] ≈ cv_face_x atol=1e-6
        @test all(x -> isapprox(x, ck_face_y[1], atol = 1e-6), ck_face_y)
        @test all(x -> isapprox(x, cv_face_y[1], atol = 1e-6), cv_face_y)
    end

    for i in get_face_set(grid, "LEFT_INFLOW")
        cell_id, face_id = i
        neighbors_cells = map(x -> x.idx[1], grid.topology.face_face_neighbours[cell_id,:])
        neighbor_id, neighbor_face_id = -1, -1
        for (j1, j2) in get_face_set(grid, "RIGHT_INFLOW")
            if j1 in neighbors_cells
                neighbor_id = j1
                neighbor_face_id = j2
            end
        end
        interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((neighbor_id, neighbor_face_id)))
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        elem = grid.cells[ck].ref_data[]
        Rγk, Rγv = Matrix(elem.R[lfγk]), Matrix(elem.R[lfγv])
    
    
        ck_face_x = Rγk * get_i_coordinates(grid.xyz[ck], 1)
        cv_face_x = Rγv * get_i_coordinates(grid.xyz[cv], 1)
        ck_face_y = Rγk * get_i_coordinates(grid.xyz[ck], 2)
        cv_face_y = Rγv * get_i_coordinates(grid.xyz[cv], 2)
        @test ck_face_y ≈ cv_face_y[Pv] atol=1e-6
        @test ck_face_y[Pk] ≈ cv_face_y atol=1e-6
        @test all(x -> isapprox(x, ck_face_x[1], atol = 1e-6), ck_face_x)
        @test all(x -> isapprox(x, cv_face_x[1], atol = 1e-6), cv_face_x)
    end

end


@testset "check correctness of P1, P2 for periodicity functionality, check coordinate matchs, for boundary only, for quadraleteral element" begin
    for i in get_face_set(grid, "TOP_INFLOW_1")
        
        cell_id, face_id = i
        neighbors_cells = map(x -> x.idx[1], grid.topology.face_face_neighbours[cell_id,:])
        neighbor_id, neighbor_face_id = -1, -1
        for (j1, j2) in get_face_set(grid, "BOTTOM_INFLOW_1")
            if j1 in neighbors_cells
                neighbor_id = j1
                neighbor_face_id = j2
            end
        end
        interface = find_FaceInterface(grid, i, FaceIndex((neighbor_id, neighbor_face_id)))
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        elem = grid.cells[ck].ref_data[]
        Rγk, Rγv = Matrix(elem.R[lfγk]), Matrix(elem.R[lfγv])
    
        ck_face_x = Rγk * get_i_coordinates(grid.xyz[ck], 1)
        cv_face_x = Rγv * get_i_coordinates(grid.xyz[cv], 1)
        ck_face_y = Rγk * get_i_coordinates(grid.xyz[ck], 2)
        cv_face_y = Rγv * get_i_coordinates(grid.xyz[cv], 2)
        @test ck_face_x ≈ cv_face_x atol=1e-6
        @test all(x -> isapprox(x, ck_face_y[1], atol = 1e-6), ck_face_y)
        @test all(x -> isapprox(x, cv_face_y[1], atol = 1e-6), cv_face_y)
    end

    for i in get_face_set(grid, "BOTTOM_INFLOW_1") 
        cell_id, face_id = i
        neighbors_cells = map(x -> x.idx[1], grid.topology.face_face_neighbours[cell_id,:])
        neighbor_id, neighbor_face_id = -1, -1
        for (j1, j2) in get_face_set(grid, "TOP_INFLOW_1")
            if j1 in neighbors_cells
                neighbor_id = j1
                neighbor_face_id = j2
            end
        end
        interface = find_FaceInterface(grid, i, FaceIndex((neighbor_id, neighbor_face_id)))
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        elem = grid.cells[ck].ref_data[]
        Rγk, Rγv = Matrix(elem.R[lfγk]), Matrix(elem.R[lfγv])
    
    
        ck_face_x = Rγk * get_i_coordinates(grid.xyz[ck], 1)
        cv_face_x = Rγv * get_i_coordinates(grid.xyz[cv], 1)
        ck_face_y = Rγk * get_i_coordinates(grid.xyz[ck], 2)
        cv_face_y = Rγv * get_i_coordinates(grid.xyz[cv], 2)
        @test ck_face_x ≈ cv_face_x atol=1e-6
        @test all(x -> isapprox(x, ck_face_y[1], atol = 1e-6), ck_face_y)
        @test all(x -> isapprox(x, cv_face_y[1], atol = 1e-6), cv_face_y)
    end


    for i in get_face_set(grid, "LEFT_INFLOW")
        cell_id, face_id = i
        neighbors_cells = map(x -> x.idx[1], grid.topology.face_face_neighbours[cell_id,:])
        neighbor_id, neighbor_face_id = -1, -1
        for (j1, j2) in get_face_set(grid, "RIGHT_INFLOW")
            if j1 in neighbors_cells
                neighbor_id = j1
                neighbor_face_id = j2
            end
        end
        interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((neighbor_id, neighbor_face_id)))
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        elem = grid.cells[ck].ref_data[]
        Rγk, Rγv = Matrix(elem.R[lfγk]), Matrix(elem.R[lfγv])
    
    
        ck_face_x = Rγk * get_i_coordinates(grid.xyz[ck], 1)
        cv_face_x = Rγv * get_i_coordinates(grid.xyz[cv], 1)
        ck_face_y = Rγk * get_i_coordinates(grid.xyz[ck], 2)
        cv_face_y = Rγv * get_i_coordinates(grid.xyz[cv], 2)
        @test ck_face_y ≈ cv_face_y atol=1e-6
        @test all(x -> isapprox(x, ck_face_x[1], atol = 1e-6), ck_face_x)
        @test all(x -> isapprox(x, cv_face_x[1], atol = 1e-6), cv_face_x)
    end

end