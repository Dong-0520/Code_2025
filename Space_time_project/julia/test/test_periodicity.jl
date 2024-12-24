using Test

@testset "test (-1,-1)" begin
    for i in 1:n_cells(grid)
        @test length(grid.topology.face_face_neighbours[i,:]) == 3
        for j in 1:3
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