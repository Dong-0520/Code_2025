using DifferentialEquations, LinearAlgebra, BlockArrays, SparseArrays
using Trixi
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics                                                                                                                                                                                                                                                                          

include("../src/SBPLite.jl")
using .SBPLite
include("plotting_helper.jl")

order = 3
# ref = TriRefElemOmega(order)
# ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 6" => ref)

ref = QuadRefElemGL(order)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Quadrilateral 4" => ref)

function identity(x)
    return x
end
 

# grid = read_mesh(joinpath(@__DIR__, "mesh/BurgersMeshSlab_tB.msh"), ref_elems_data, identity)
# grid = read_mesh(joinpath(@__DIR__, "BurgersMeshSlab_tB001.msh"), ref_elems_data, identity)
# grid = read_mesh(joinpath(@__DIR__, "mesh/BurgersMeshSlab_tB0001.msh"), ref_elems_data, identity)
# grid = read_mesh(joinpath(@__DIR__, "mesh/BurgersMeshSlab_tB00001.msh"), ref_elems_data, identity)
grid = read_mesh(joinpath(@__DIR__, "2d_grid_periodic.msh"), ref_elems_data, identity)

# grid = read_mesh(joinpath(@__DIR__, "2d_grid_test.msh"), ref_elems_data, identity)
grid = read_mesh(joinpath(@__DIR__, "2d_grid.msh"), ref_elems_data, identity)

include("test_Grid.jl")

@save joinpath(@__DIR__, "mesh/BurgersMeshSlab_tB0005.jld2") grid
@save joinpath(@__DIR__, "mesh/BurgersMeshSlab_tB0001.jld2") grid
# @save joinpath(@__DIR__, "mesh/BurgersMeshSlab_tB.jld2") grid
@load joinpath(@__DIR__, "mesh/BurgersMeshSlab_tB001.jld2") grid
@load joinpath(@__DIR__, "mesh/BurgersMeshSlab_tB.jld2") grid

@save joinpath(@__DIR__, "2d_grid_quad.jld2") grid 







# continuous case, dont forget change flux at shock
function analytic_u(x::Coord)
    x1, t1 = x[1], x[2]
    return sinpi(4 * x1)
end
bottomBC1(x::Coord) = analytic_u(x)
bottomBC2(x::Coord) = analytic_u(x)
leftBC(x::Coord) = analytic_u(x)
rightBC(x::Coord) = analytic_u(x)

function get_interior_interfaces_periodicBC(grid)
    """
    This function is used to get the interior interfaces of the grid
    only works for periodic BC because the left and right BC is considered as interior interfaces
    """
    interior_interfaces = Set{FaceInterface}()
    for interface in grid.face_interfaces
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        _, Pv = interface.P1, interface.P2
        refk = grid.cells[ck].ref_data[]
        refv = grid.cells[cv].ref_data[]
        Rγk, Rγv = Matrix(refk.R[lfγk]), Matrix(refv.R[lfγv])

        ck_face_x = map(x -> x[1], Rγk * grid.xyz_q[ck])
        cv_face_x = map(x -> x[1], Rγv * grid.xyz_q[cv])
        ck_face_y = map(x -> x[2], Rγk * grid.xyz_q[ck])
        cv_face_y = map(x -> x[2], Rγv * grid.xyz_q[cv])
        if all(x -> isapprox(x, 0, atol = 1e-12), ck_face_x .- cv_face_x[Pv]) && all(x -> isapprox(x, 0, atol = 1e-12), ck_face_y .- cv_face_y[Pv])
            push!(interior_interfaces, interface)
        end
    end

    for (cell_id, face_id) in get_face_set(grid, "LEFT_INFLOW")
        neighbors_cells = map(x -> x.idx[1], grid.topology.face_face_neighbours[cell_id,:])
        neighbor_id, neighbor_face_id = -1, -1
        for (j1, j2) in get_face_set(grid, "RIGHT_INFLOW")
            if j1 in neighbors_cells
                neighbor_id = j1
                neighbor_face_id = j2
            end
        end
        interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((neighbor_id, neighbor_face_id)))
        push!(interior_interfaces, interface)
    end
    return interior_interfaces
end

function set_IC_for_slab(grid; u = Matrix{Float64}(undef, 2, 2), slab = 1)
    """
    return the initial condition and left/right boundary condtion for the next slab
    grid is the main slab grid, the coordiante is built upon the first slab
    u contains the solution from last slab
    initial condition is the last solution of the last slab
    left/right boundary condition will be evaluated by tghe coordinate of the grid plus dt
    """
    IC = Dict()
    # yT = maximum( map(x -> x[2], grid.xyz_q[collect(get_face_set(grid, "TOP_INFLOW_1"))[1][1]]) )
    # yB = minimum( map(x -> x[2], grid.xyz_q[collect(get_face_set(grid, "BOTTOM_INFLOW_1"))[1][1]]) )
    # dy = yT - yB

    if slab == 1
        for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_1")
            IC[(cell_id, face_id)] = bottomBC1.( grid.cells[cell_id].ref_data[].R[face_id] * grid.xyz_q[cell_id])
        end
        for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_2")
            IC[(cell_id, face_id)] = bottomBC2.( grid.cells[cell_id].ref_data[].R[face_id] * grid.xyz_q[cell_id])
        end
    else
        for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_1")
            neighbors_cells = map(x -> x.idx[1], grid.topology.face_face_neighbours[cell_id,:])
            neighbor_id, neighbor_face_id = -1, -1
            for (j1, j2) in get_face_set(grid, "TOP_INFLOW_1")
                if j1 in neighbors_cells
                    neighbor_id = j1
                    neighbor_face_id = j2
                end
            end
            interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((neighbor_id, neighbor_face_id)))
            # to make sure the ck is the element on the bottom boundary
            if FaceIndex((cell_id, face_id)) == interface.face_1
                ck, lfγk = interface.face_1
                cv, lfγv = interface.face_2
            else
                ck, lfγk = interface.face_2
                cv, lfγv = interface.face_1
            end
            Pk, Pv = interface.P1, interface.P2
            Rγk, Rγv = Matrix(grid.cells[ck].ref_data[].R[lfγk]), Matrix(grid.cells[cv].ref_data[].R[lfγv])

            IC[(cell_id, face_id)] = (Rγv * u[cv, :])[Pv]
        end

        for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_2")
            neighbors_cells = map(x -> x.idx[1], grid.topology.face_face_neighbours[cell_id,:])
            neighbor_id, neighbor_face_id = -1, -1
            for (j1, j2) in get_face_set(grid, "TOP_INFLOW_2")
                if j1 in neighbors_cells
                    neighbor_id = j1
                    neighbor_face_id = j2
                end
            end
            interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((neighbor_id, neighbor_face_id)))
            # to make sure the ck is the element on the bottom boundary
            if FaceIndex((cell_id, face_id)) == interface.face_1
                ck, lfγk = interface.face_1
                cv, lfγv = interface.face_2
            else
                ck, lfγk = interface.face_2
                cv, lfγv = interface.face_1
            end
            Pk, Pv = interface.P1, interface.P2
            Rγk, Rγv = Matrix(grid.cells[ck].ref_data[].R[lfγk]), Matrix(grid.cells[cv].ref_data[].R[lfγv])

            IC[(cell_id, face_id)] = (Rγv * u[cv, :])[Pv]
        end


    end
    return IC
end


function mysign(x)
    x >= 0 ? 1 : -1
end


function find_interfaces_align_shock(grid)
    interfaces_set = Set{FaceInterface}()
    for (cell_id, face_id) in get_face_set(grid, "SHOCK")
        neighbor_id, neighbor_face_id = Tuple(findall(x -> x == FaceIndex((cell_id, face_id)), grid.topology.face_face_neighbours)[1])
        interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((neighbor_id, neighbor_face_id)))
        push!(interfaces_set, interface)
    end
    return interfaces_set
end


function get_set_up(grid::Grid; u = Matrix{Float64}(undef, 2, 2), slab = 1)   
    interfaces_align_shock = find_interfaces_align_shock(grid)
    p = (grid, interfaces_align_shock, get_interior_interfaces_periodicBC(grid), set_IC_for_slab(grid, u = u, slab = slab))
    return p
end

function space_time_RK4(U, t, pseudo_dt, p; diss = false, test = false)
    c1, c2, c3, c4 = 0.0, 0.40128709, 0.56449983, 0.87678807
    b1, b2, b3, b4 = 0.20334721, 0.19932974, 0.28339585, 0.31392720
    a21 = 0.40128709
    a31, a32 = 0.28224991, 0.28224991
    a41, a42, a43 = 0.25972925, 0.25479937, 0.36225945

    # k1 = RHS(U, p, t)
    # k2 = RHS(U + 0.5 * pseudo_dt * k1, p, t + 0.5 * pseudo_dt)
    # k3 = RHS(U + 0.5 * pseudo_dt * k2, p, t + 0.5 * pseudo_dt)
    # k4 = RHS(U + pseudo_dt * k3, p, t + pseudo_dt)
    # return U + (pseudo_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


    if diss && !test
        k1 = RHS_diss_periodic(U, p)
        k2 = RHS_diss_periodic(U + a21 * pseudo_dt * k1, p)
        k3 = RHS_diss_periodic(U + a31 * pseudo_dt * k1 + a32 * pseudo_dt * k2, p)
        k4 = RHS_diss_periodic(U + a41 * pseudo_dt * k1 + a42 * pseudo_dt * k2 + a43 * pseudo_dt * k3, p)

    elseif test
        k1 = RHS_test(U, p)
        k2 = RHS_test(U + a21 * pseudo_dt * k1, p)
        k3 = RHS_test(U + a31 * pseudo_dt * k1 + a32 * pseudo_dt * k2, p)
        k4 = RHS_test(U + a41 * pseudo_dt * k1 + a42 * pseudo_dt * k2 + a43 * pseudo_dt * k3, p)
    else
        k1 = RHS_periodic(U, p)
        k2 = RHS_periodic(U + a21 * pseudo_dt * k1, p)
        k3 = RHS_periodic(U + a31 * pseudo_dt * k1 + a32 * pseudo_dt * k2, p)
        k4 = RHS_periodic(U + a41 * pseudo_dt * k1 + a42 * pseudo_dt * k2 + a43 * pseudo_dt * k3, p)
    end

    return U + pseudo_dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4)
end



function space_time_solve(U0::Array{Float64, 2}, p::Any; pseudo_dt = 1.0, num_of_pseudo_time_step = 100, diss = false, test = false)
    num_of_cells = n_cells(p[1])
    num_of_nodes = Int(length(p[1].xyz_q[1]))
    all_U = ones(Float64, (num_of_pseudo_time_step+1, num_of_cells, num_of_nodes))
    all_U[1, :, :] = U0
    diffs = zeros(num_of_pseudo_time_step)

    for pseudo_step in 1:num_of_pseudo_time_step
        curr_U = deepcopy(all_U[pseudo_step, :, :, :])

        # the if statement below is for furthur ease of implementation
        if pseudo_step > 50
            pseudo_dt = pseudo_dt
            next_U = space_time_RK4(curr_U, 0.0, pseudo_dt, p, diss = diss, test = test)
            max_diff = maximum(abs.(next_U - curr_U))
            println("Pseudo time step: $pseudo_step, Maximum difference: ", max_diff, "\n")
            diffs[pseudo_step] = max_diff
            all_U[pseudo_step+1, :, :, :] = next_U

        elseif pseudo_step > 150
            pseudo_dt = pseudo_dt
            next_U = space_time_RK4(curr_U, 0.0, pseudo_dt, p, diss = diss, test = test)
            max_diff = maximum(abs.(next_U - curr_U))
            println("Pseudo time step: $pseudo_step, Maximum difference: ", max_diff, "\n")
            diffs[pseudo_step] = max_diff
            all_U[pseudo_step+1, :, :, :] = next_U
        else
            next_U = space_time_RK4(curr_U, 0.0, pseudo_dt, p, diss = diss, test = test)
            max_diff = maximum(abs.(next_U - curr_U))
            println("Pseudo time step: $pseudo_step, Maximum difference: ", max_diff, "\n")
            diffs[pseudo_step] = max_diff
            all_U[pseudo_step+1, :, :, :] = next_U
        end

        if max_diff < 1e-13
            println("\n Solution converged at pseudo time step: $pseudo_step \n")
            return all_U[1:pseudo_step, :, :], diffs[1:pseudo_step]
        end

    end
    print("\n Solution not converged, pseudo_dt = $pseudo_dt \n")
    return all_U, diffs
end


function heaviside_fnc(x)
    if x > 0 + 1e-15
        return 1
    elseif isapprox(x, 0, atol = 1e-15)
        return 1
    else
        return 0
    end

    # return 1 / (1 + exp(-2 * 200 * x))
    
end



function RHS_periodic(u, p)
    grid, interfaces_align_shock, interior_interfaces, IC = p
    result = zeros(n_cells(grid), nodes_per_cell)

    # volume discretization
    for cell_id in 1:n_cells(grid)
        
        # J = Diagonal(abs.(grid.geometric_terms.J_q[cell_id]))
        Dx = grid.VOL[cell_id][1]
        Dt = grid.VOL[cell_id][2]
        

        local_sol = u[cell_id, :]
        ux = 1/3 * Dx * (local_sol .^ 2) .+ 1/3 * local_sol .* (Dx * local_sol)
        ut = Dt * local_sol 

        result[cell_id, :] = (-ux .- ut)
    end

    # interface discretization
    for interface in interior_interfaces
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        refk = grid.cells[ck].ref_data[]
        refv = grid.cells[cv].ref_data[]
        Rγk, Rγv = Matrix(refk.R[lfγk]), Matrix(refv.R[lfγv])

        uk, uv = u[ck, :], u[cv, :]
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        # normal vector of physical element
        maskγk, maskγv = @views refk.f_mask[lfγk], refv.f_mask[lfγv]
        normalγk, normalγv = @views grid.geometric_terms.N_f[ck][:, maskγk], grid.geometric_terms.N_f[cv][:, maskγv]
        Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
        Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]
        
        if all( x -> x >= 0, Nxγk .* (uk_face - uk_adj))
            spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4
            spatial_flux_v = (uv_face.^2 + uv_adj.^2) ./ 4
        else
            spatial_flux_k = (uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6
            spatial_flux_v = (uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 
        end 
        rk_spatial = grid.FAC[ck][lfγk] * Diagonal(Nxγk) * (0.5 .* uk_face.^2 .- spatial_flux_k)
        rv_spatial = grid.FAC[cv][lfγv] * Diagonal(Nxγv) * (0.5 .* uv_face.^2 .- spatial_flux_v)

        # temporal term
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        rk_temporal = grid.FAC[ck][lfγk] * Diagonal(Ntγk) * (0.5 .* uk_face .- 0.5 .* uk_adj)
        rv_temporal = grid.FAC[cv][lfγv] * Diagonal(Ntγv) * (0.5 .* uv_face .- 0.5 .* uv_adj)
        
        result[ck, :] += (rk_spatial .+ rk_temporal )
        result[cv, :] += (rv_spatial .+ rv_temporal )
    end


    for (cell_id, face_id) in union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2"))
        # cell = get_cells(grid, cell_id)
        # elem = get_ref_elem_data(grid, cell)
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        # Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = IC[(cell_id, face_id)]

        result[cell_id, :] += (grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj))
    end

    return result
end

function RHS_diss_periodic(u, p)
    grid, interfaces_align_shock, interior_interfaces, IC = p
    result = zeros(n_cells(grid), nodes_per_cell)

    # volume discretization
    for cell_id in 1:n_cells(grid)
        
        # J = Diagonal(abs.(grid.geometric_terms.J_q[cell_id]))
        Dx = grid.VOL[cell_id][1]
        Dt = grid.VOL[cell_id][2]
        

        local_sol = u[cell_id, :]
        ux = 1/3 * Dx * (local_sol .^ 2) .+ 1/3 * local_sol .* (Dx * local_sol)
        ut = Dt * local_sol 

        result[cell_id, :] = (-ux .- ut)
    end

    # interface discretization
    for interface in interior_interfaces
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        refk = grid.cells[ck].ref_data[]
        refv = grid.cells[cv].ref_data[]
        Rγk, Rγv = Matrix(refk.R[lfγk]), Matrix(refv.R[lfγv])

        uk, uv = u[ck, :], u[cv, :]
        uk_face, uv_face = Rγk * uk, Rγv * uv
        # uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        # normal vector of physical element
        maskγk, maskγv = @views refk.f_mask[lfγk], refv.f_mask[lfγv]
        normalγk, normalγv = @views grid.geometric_terms.N_f[ck][:, maskγk], grid.geometric_terms.N_f[cv][:, maskγv]
        Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
        Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]
        
        if all( x -> x >= 0, Nxγk .* (uk_face - uk_adj))
            spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4
            spatial_flux_v = (uv_face.^2 + uv_adj.^2) ./ 4
        else
            spatial_flux_k = (uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 .+ (mysign.(Nxγk)) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)
            spatial_flux_v = (uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 .+ (mysign.(Nxγv)) .* max.(abs.(uv_face), abs.(uv_adj)) .* (uv_face .- uv_adj)
        end 

        # spatial_flux_k = (uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 .+ (mysign.(Nxγk)) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)
        # spatial_flux_v = (uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 .+ (mysign.(Nxγv)) .* max.(abs.(uv_face), abs.(uv_adj)) .* (uv_face .- uv_adj)

        # function smooth_nondiff(x)
        #     return 1 / (1 + exp(-100 * x))
        # end
        # indicator_ave = smooth_nondiff.(Nxγk .* (uk_face - uk_adj))
        # indicator_upwind = smooth_nondiff.(-Nxγk .* (uk_face - uk_adj))
        # spatial_flux_k = ((uk_face.^2 + uk_adj.^2) ./ 4) .* indicator_ave .+ ((uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 .+ (mysign.(Nxγk)) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)) .* indicator_upwind
        # spatial_flux_v = ((uv_face.^2 + uv_adj.^2) ./ 4) .* indicator_ave .+ ((uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 .+ (mysign.(Nxγv)) .* max.(abs.(uv_face), abs.(uv_adj)) .* (uv_face .- uv_adj)) .* indicator_upwind

        rk_spatial = grid.FAC[ck][lfγk] * Diagonal(Nxγk) * (0.5 .* uk_face.^2 .- spatial_flux_k)
        rv_spatial = grid.FAC[cv][lfγv] * Diagonal(Nxγv) * (0.5 .* uv_face.^2 .- spatial_flux_v)

        # temporal term
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        rk_temporal = grid.FAC[ck][lfγk] * Diagonal(Ntγk) * (0.5 .* uk_face .- 0.5 .* uk_adj)
        rv_temporal = grid.FAC[cv][lfγv] * Diagonal(Ntγv) * (0.5 .* uv_face .- 0.5 .* uv_adj)
        
        result[ck, :] += (rk_spatial .+ rk_temporal )
        result[cv, :] += (rv_spatial .+ rv_temporal )
    end


    for (cell_id, face_id) in union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2"))
        # cell = get_cells(grid, cell_id)
        # elem = get_ref_elem_data(grid, cell)
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        # Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = IC[(cell_id, face_id)]

        result[cell_id, :] += (grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj))
    end

    return result
end

function RHS_test(u, p)
    grid, interfaces_align_shock, interior_interfaces, IC = p
    result = zeros(n_cells(grid), nodes_per_cell)

    # volume discretization
    for cell_id in 1:n_cells(grid)
        
        # J = Diagonal(abs.(grid.geometric_terms.J_q[cell_id]))
        Dx = grid.VOL[cell_id][1]
        Dt = grid.VOL[cell_id][2]
        

        local_sol = u[cell_id, :]
        ux = 1/3 * Dx * (local_sol .^ 2) .+ 1/3 * local_sol .* (Dx * local_sol)
        ut = Dt * local_sol 

        result[cell_id, :] = (-ux .- ut)
    end

    # interface discretization
    # for interface in setdiff(interior_interfaces, interfaces_align_shock)
    for interface in interior_interfaces
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        refk = grid.cells[ck].ref_data[]
        refv = grid.cells[cv].ref_data[]
        Rγk, Rγv = Matrix(refk.R[lfγk]), Matrix(refv.R[lfγv])

        uk, uv = u[ck, :], u[cv, :]
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        # normal vector of physical element
        maskγk, maskγv = @views refk.f_mask[lfγk], refv.f_mask[lfγv]
        normalγk, normalγv = @views grid.geometric_terms.N_f[ck][:, maskγk], grid.geometric_terms.N_f[cv][:, maskγv]
        Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
        Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]
        
        if all( x -> x >= 0, Nxγk .* (uk_face - uk_adj))
            spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4
            spatial_flux_v = (uv_face.^2 + uv_adj.^2) ./ 4
        else
            spatial_flux_k = (uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 .+ (mysign.(Nxγk)) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)
            spatial_flux_v = (uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 .+ (mysign.(Nxγv)) .* max.(abs.(uv_face), abs.(uv_adj)) .* (uv_face .- uv_adj)
        end 

        rk_spatial = grid.FAC[ck][lfγk] * Diagonal(Nxγk) * (0.5 .* uk_face.^2 .- spatial_flux_k)
        rv_spatial = grid.FAC[cv][lfγv] * Diagonal(Nxγv) * (0.5 .* uv_face.^2 .- spatial_flux_v)

        # temporal term
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        rk_temporal = grid.FAC[ck][lfγk] * Diagonal(Ntγk) * (0.5 .* uk_face .- 0.5 .* uk_adj)
        rv_temporal = grid.FAC[cv][lfγv] * Diagonal(Ntγv) * (0.5 .* uv_face .- 0.5 .* uv_adj)
        
        result[ck, :] += (rk_spatial .+ rk_temporal )
        result[cv, :] += (rv_spatial .+ rv_temporal )
    end

    # for interface in interfaces_align_shock 
    #     ck, lfγk = interface.face_1
    #     cv, lfγv = interface.face_2
    #     Pk, Pv = interface.P1, interface.P2
    #     refk = grid.cells[ck].ref_data[]
    #     refv = grid.cells[cv].ref_data[]
    #     Rγk, Rγv = Matrix(refk.R[lfγk]), Matrix(refv.R[lfγv])

    #     uk, uv = u[ck, :], u[cv, :]
    #     uk_face, uv_face = Rγk * uk, Rγv * uv
    #     uk_adj, uv_adj = uv_face, uk_face # 对于 quad element, 这个是不需要的。see test_Grid for details


    #     # normal vector of physical element
    #     maskγk, maskγv = @views refk.f_mask[lfγk], refv.f_mask[lfγv]
    #     normalγk, normalγv = @views grid.geometric_terms.N_f[ck][:, maskγk], grid.geometric_terms.N_f[cv][:, maskγv]
    #     Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
    #     Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]
        
    #     spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4
    #     spatial_flux_v = (uv_face.^2 + uv_adj.^2) ./ 4
    #     rk_spatial = grid.FAC[ck][lfγk] * Diagonal(Nxγk) * (0.5 .* uk_face.^2 .- spatial_flux_k)
    #     rv_spatial = grid.FAC[cv][lfγv] * Diagonal(Nxγv) * (0.5 .* uv_face.^2 .- spatial_flux_v)

    #     # temporal term
    #     uk_face, uv_face = Rγk * uk, Rγv * uv
    #     uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

    #     rk_temporal = grid.FAC[ck][lfγk] * Diagonal(Ntγk) * (0.5 .* uk_face .- 0.5 .* uk_adj)
    #     rv_temporal = grid.FAC[cv][lfγv] * Diagonal(Ntγv) * (0.5 .* uv_face .- 0.5 .* uv_adj)
        
    #     result[ck, :] += (rk_spatial .+ rk_temporal )
    #     result[cv, :] += (rv_spatial .+ rv_temporal )
    # end


    # for (cell_id, face_id) in union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2"))
    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_1")
        # cell = get_cells(grid, cell_id)
        # elem = get_ref_elem_data(grid, cell)
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        # Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = IC[(cell_id, face_id)]

        result[cell_id, :] += (grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj))
    end

    return result
end

function initial_guess(grid)
    nodes_per_cell = length(grid.xyz_q[1])
    U0 = zeros(Float64, n_cells(grid), nodes_per_cell)
    for cell_id in 1:n_cells(grid)
        for node_id in 1:nodes_per_cell
            U0[cell_id, node_id] = analytic_u(grid.xyz_q[cell_id][node_id])
        end
    end
    return U0
    
end

function L2_norm_of_RHS(grid, u)
    result = 0.0
    for i in 1:n_cells(grid)
        JP = Diagonal(abs.(grid.geometric_terms.J_q[i])) * grid.cells[i].ref_data[].H
        result += u[i, :]' * JP * u[i, :]
    end

    return sqrt(result)
end


zlimit = [-1.1, 1.1]
nodes_per_cell = length(grid.xyz_q[1])


# ---------------------- slab 1 ---------------------- #
p1 = get_set_up(grid, slab = 1)
IC1 = p1[end]

U0 = initial_guess(grid)
@time all_U, diffs = space_time_solve(U0, p1, pseudo_dt = 0.0001, num_of_pseudo_time_step = 2000, diss = true, test = true)
gif(observe_iteration(grid, all_U[1:10:end, :, :], zlimit, show_true_sol = false), joinpath(@__DIR__, "numerical_test_discts.gif"), fps=5)
u1 = all_U[end, :, :]
observe_numerical_sol(grid, u1, zlimit, show_true_sol = false)

#---------------------- slab 2 ---------------------- #
p2 = get_set_up(grid, u = u1, slab = 2)
IC2 = p2[end]

# check if the IC2 is correct
p = scatter()
for (cell_id, face_id) in union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2"))
    coord = grid.xyz_q[cell_id]
    R = grid.cells[cell_id].ref_data[].R[face_id]
    coords = R * coord
    x_coords = map(x -> x[1], coords)
    scatter!(p, x_coords, IC2[(cell_id, face_id)], label = "", color = "red")
    scatter!(p, x_coords, IC1[(cell_id, face_id)], label = "", color = "blue")

end
p

@time all_U2, diffs2 = space_time_solve(u1, p2, pseudo_dt = 0.0001, num_of_pseudo_time_step = 2500, diss = true, test = true)
gif(observe_iteration(grid, all_U2[1:50:end, :, :], zlimit, show_true_sol = false), joinpath(@__DIR__, "numerical_test_discts.gif"), fps=5)
plot([L2_norm_of_RHS(grid, RHS_diss_periodic(all_U2[i, :, :], p2)) for i in 1:length(all_U2[:,1,1]) ], label = "", xlabel = "Pseudo time step", ylabel = "Max abs RHS", title = "Convergence of the numerical solution")

observe_numerical_sol(grid, all_U2[end, :, :], zlimit, show_true_sol = false)

#---------------------- solver part ---------------------- #
# No dissipation
function RHS_forSolver_withoutdiss(du, u, p)

    grid, interfaces_align_shock, interior_interfaces, IC = p

    # volume discretization
    for cell_id in 1:n_cells(grid)
        
        Dx = grid.VOL[cell_id][1]
        Dt = grid.VOL[cell_id][2]
        

        local_sol = u[cell_id, :]
        ux = 1/3 * Dx * (local_sol .^ 2) .+ 1/3 * local_sol .* (Dx * local_sol)
        ut = Dt * local_sol 


        du[cell_id, :] = -ux .- ut
    end

    # interface discretization
    for interface in interior_interfaces
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        refk = grid.cells[ck].ref_data[]
        refv = grid.cells[cv].ref_data[]
        Rγk, Rγv = Matrix(refk.R[lfγk]), Matrix(refv.R[lfγv])

        uk, uv = u[ck, :], u[cv, :]
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        # normal vector of physical element
        maskγk, maskγv = @views refk.f_mask[lfγk], refv.f_mask[lfγv]
        normalγk, normalγv = @views grid.geometric_terms.N_f[ck][:, maskγk], grid.geometric_terms.N_f[cv][:, maskγv]
        Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
        Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]
        
        # if all( x -> x >= 0, Nxγk .* (uk_face - uk_adj))
        #     spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4
        #     spatial_flux_v = (uv_face.^2 + uv_adj.^2) ./ 4
        # else
        #     spatial_flux_k = (uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6
        #     spatial_flux_v = (uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 
        # end 
        # 替代布尔判断的代码
        indicator = Nxγk .* (uk_face - uk_adj)
        spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4 .* (indicator .>= 0) + (uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 .* (indicator .< 0)

        # 同样的逻辑适用于 spatial_flux_v
        indicator_v = Nxγv .* (uv_face - uv_adj)
        spatial_flux_v = (uv_face.^2 + uv_adj.^2) ./ 4 .* (indicator_v .>= 0) + (uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 .* (indicator_v .< 0)

        rk_spatial = grid.FAC[ck][lfγk] * Diagonal(Nxγk) * (0.5 .* uk_face.^2 .- spatial_flux_k)
        rv_spatial = grid.FAC[cv][lfγv] * Diagonal(Nxγv) * (0.5 .* uv_face.^2 .- spatial_flux_v)

        # temporal term
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        rk_temporal = grid.FAC[ck][lfγk] * Diagonal(Ntγk) * (0.5 .* uk_face .- 0.5 .* uk_adj)
        rv_temporal = grid.FAC[cv][lfγv] * Diagonal(Ntγv) * (0.5 .* uv_face .- 0.5 .* uv_adj)
        
        du[ck, :] += rk_spatial .+ rk_temporal
        du[cv, :] += rv_spatial .+ rv_temporal
    end

    for (cell_id, face_id) in union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2"))

        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        # Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = IC[(cell_id, face_id)]

        du[cell_id, :] += grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj)
    end
    return nothing
end

using BenchmarkTools
U0_for_solver = all_U2[argmin([L2_norm_of_RHS(grid, RHS_diss_periodic(all_U2[i, :, :], p2)) for i in 1:length(all_U2[:,1,1]) ]), :, :]
prob = NonlinearProblem(RHS_forSolver_withoutdiss, U0_for_solver, p2)
@benchmark sol1 = solve(prob, NewtonRaphson(); maxiters=5000, nlsolve_kwargs=(autodiff=false,))
@benchmark sol2 = solve(prob, LevenbergMarquardt(), abstol=1e-8, reltol=1e-8, maxiters=5000)
@benchmark sol3 = solve(prob, TrustRegion(), abstol=1e-8, reltol=1e-8, maxiters=5000)
@benchmark sol4 = solve(prob, PseudoTransient(), abstol=1e-8, reltol=1e-8, maxiters=5000)



result_plot1 = observe_numerical_sol(grid, sol1.u, zlimit, show_true_sol = false)
save(joinpath(@__DIR__, "numerical_result", "LevenMar_tB001,residual=$(round(L2_norm_of_RHS(grid, sol1.resid), digits=2)).png"), result_plot1)
result_plot2 = observe_numerical_sol(grid, sol2.u, zlimit, show_true_sol = false)
save(joinpath(@__DIR__, "numerical_result", "TrustRegion_tB001,residual=$(round(L2_norm_of_RHS(grid, sol2.resid), digits=2)).png"), result_plot2)
result_plot3 = observe_numerical_sol(grid, sol3.u, zlimit, show_true_sol = false)
save(joinpath(@__DIR__, "numerical_result", "PseudoTransient_tB001,residual=$(round(L2_norm_of_RHS(grid, sol3.resid), digits=2)).png"), result_plot3)

save(joinpath(@__DIR__, "numerical_result", "LevenMar_tB001,residual.png"), observe_numerical_sol(grid, sol1.resid, zlimit, show_true_sol = false))
save(joinpath(@__DIR__, "numerical_result", "TrustRegion_tB001,residual.png"), observe_numerical_sol(grid, sol2.resid, zlimit, show_true_sol = false))
save(joinpath(@__DIR__, "numerical_result", "PseudoTransient_tB001,residual.png"), observe_numerical_sol(grid, sol3.resid, zlimit, show_true_sol = false))



# dissipation
function RHS_forSolver_diss(du, u, p)

    grid, interfaces_align_shock, interior_interfaces, IC = p

    # volume discretization
    for cell_id in 1:n_cells(grid)
        
        Dx = grid.VOL[cell_id][1]
        Dt = grid.VOL[cell_id][2]
        

        local_sol = u[cell_id, :]
        ux = 1/3 * Dx * (local_sol .^ 2) .+ 1/3 * local_sol .* (Dx * local_sol)
        ut = Dt * local_sol 


        du[cell_id, :] = -ux .- ut
    end

    # interface discretization
    for interface in interior_interfaces
        ck, lfγk = interface.face_1
        cv, lfγv = interface.face_2
        Pk, Pv = interface.P1, interface.P2
        refk = grid.cells[ck].ref_data[]
        refv = grid.cells[cv].ref_data[]
        Rγk, Rγv = Matrix(refk.R[lfγk]), Matrix(refv.R[lfγv])

        uk, uv = u[ck, :], u[cv, :]
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        # normal vector of physical element
        maskγk, maskγv = @views refk.f_mask[lfγk], refv.f_mask[lfγv]
        normalγk, normalγv = @views grid.geometric_terms.N_f[ck][:, maskγk], grid.geometric_terms.N_f[cv][:, maskγv]
        Nxγk, Ntγk = normalγk[1, :], normalγk[2, :]
        Nxγv, Ntγv = normalγv[1, :], normalγv[2, :]
        
        # if all( x -> x >= 0, Nxγk .* (uk_face - uk_adj))
        #     spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4
        #     spatial_flux_v = (uv_face.^2 + uv_adj.^2) ./ 4
        # else
            # spatial_flux_k = (uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 .+ (mysign.(Nxγk)) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)
            # spatial_flux_v = (uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 .+ (mysign.(Nxγv)) .* max.(abs.(uv_face), abs.(uv_adj)) .* (uv_face .- uv_adj)
        # end 
        # 替代布尔判断的代码
        indicator = Nxγk .* (uk_face - uk_adj)
        spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4 .* (indicator .>= 0) + ((uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 .+ mysign.(Nxγk) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)) .* (indicator .< 0)

        # 同样的逻辑适用于 spatial_flux_v
        indicator_v = Nxγv .* (uv_face - uv_adj)
        spatial_flux_v = (uv_face.^2 + uv_adj.^2) ./ 4 .* (indicator_v .>= 0) + ((uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 .+ mysign.(Nxγv) .* max.(abs.(uv_face), abs.(uv_adj)) .* (uv_face .- uv_adj)) .* (indicator_v .< 0)

        rk_spatial = grid.FAC[ck][lfγk] * Diagonal(Nxγk) * (0.5 .* uk_face.^2 .- spatial_flux_k)
        rv_spatial = grid.FAC[cv][lfγv] * Diagonal(Nxγv) * (0.5 .* uv_face.^2 .- spatial_flux_v)

        # temporal term
        uk_face, uv_face = Rγk * uk, Rγv * uv
        uk_adj, uv_adj = uv_face[Pv], uk_face[Pk]

        rk_temporal = grid.FAC[ck][lfγk] * Diagonal(Ntγk) * (0.5 .* uk_face .- 0.5 .* uk_adj)
        rv_temporal = grid.FAC[cv][lfγv] * Diagonal(Ntγv) * (0.5 .* uv_face .- 0.5 .* uv_adj)
        
        du[ck, :] += rk_spatial .+ rk_temporal
        du[cv, :] += rv_spatial .+ rv_temporal
    end

    for (cell_id, face_id) in union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2"))

        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        mask = @views ref.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        # Nxγ = normal[1, :]
        Ntγ = normal[2, :]

        u_face = R * u[cell_id, :]
        u_adj = IC[(cell_id, face_id)]

        du[cell_id, :] += grid.FAC[cell_id][face_id] * Diagonal(Ntγ) * (u_face .- u_adj)
    end
    return nothing
end

using Symbolics
U0_for_solver = all_U2[argmin([L2_norm_of_RHS(grid, RHS_diss_periodic(all_U2[i, :, :], p2)) for i in 1:length(all_U2[:,1,1]) ]), :, :]
f_diss! = (du, u) -> RHS_forSolver_diss(du, u, p2)
du0_diss = similar(U0_for_solver)
jac_sparsity_diss = Symbolics.jacobian_sparsity(f_diss!, du0_diss, U0_for_solver)

# 生成稀疏雅可比结构（布尔矩阵）
jac_sparsity_diss = Symbolics.jacobian_sparsity(f_diss!, du0_diss, U0_for_solver)

# 将 Bool 稀疏结构转换为数值类型的稀疏矩阵
sparse_jacobian_diss = Float64.(sparse(jac_sparsity_diss))

# 使用 NonlinearFunction 和 jac_prototype
nonlinearFunction_with_jac_diss = NonlinearFunction(RHS_forSolver_diss; jac_prototype=sparse_jacobian_diss)
prob_with_jac_diss = NonlinearProblem(nonlinearFunction_with_jac_diss, U0_for_solver, p2; abstol=1e-10, reltol=1e-10)

# 求解问题
sol1_diss = solve(prob_with_jac_diss, NewtonRaphson(linsolve = KrylovJL_GMRES()), maxiters=10)
sol1_diss_falseAutodiff = solve(prob_with_jac_diss, NewtonRaphson(; autodiff=false))

sol2_diss = solve(prob_with_jac_diss, LevenbergMarquardt())
sol3_diss = solve(prob_with_jac_diss, TrustRegion())
sol4_diss = solve(prob_with_jac_diss, PseudoTransient())

result_plot1_diss = observe_numerical_sol(grid, sol1_diss_falseAutodiff.u, zlimit, show_true_sol = false)
save(joinpath(@__DIR__, "numerical_result", "NewtonRaphson_tB,residual=$(round(L2_norm_of_RHS(grid,sol1_diss_falseAutodiff.resid), digits=2)),diss.png"), result_plot1_diss)
result_plot2_diss = observe_numerical_sol(grid, sol2_diss.u, zlimit, show_true_sol = false)
save(joinpath(@__DIR__, "numerical_result", "LevenMar_tB,residual=$(round(L2_norm_of_RHS(grid, sol2_diss.resid), digits=2)),diss.png"), result_plot2_diss)
result_plot3_diss = observe_numerical_sol(grid, sol3_diss.u, zlimit, show_true_sol = false)
save(joinpath(@__DIR__, "numerical_result", "TrustRegion_tB,residual=$(round(L2_norm_of_RHS(grid, sol3_diss.resid), digits=2)),diss.png"), result_plot3_diss)
result_plot4_diss = observe_numerical_sol(grid, sol4_diss.u, zlimit, show_true_sol = false)
save(joinpath(@__DIR__, "numerical_result", "PseudoTransient_tB,residual=$(round(L2_norm_of_RHS(grid, sol4_diss.resid), digits=2)),diss.png"), result_plot4_diss)

#---------------------- developing new solver ---------------------- #

sparse_jacobian = sparse(jac_sparsity)

sol5 = solve(prob, NewtonRaphson(), jac_prototype=sparse_jacobian)

using SparseDiffTools, LinearSolve

# 自动生成稀疏雅可比的线性求解器
linsolve = Factorize()  # 或 KrylovJL_GMRES()
jac_sparsity = Symbolics.jacobian_sparsity(f!, du0, U0_for_solver)

sol5 = solve(prob, NewtonRaphson(linsolve=KrylovJL_GMRES(), adkwargs=(; jac_prototype=sparse_jacobian)), abstol=1e-8, reltol=1e-8, maxiters=5)

NonlinearProblem(RHS_forSolver_diss, U0_for_solver, p2)
nonlinearFunction_with_jac = NonlinearFunction(RHS_forSolver_diss; jac_prototype = jac_sparsity)
prob_with_jac = NonlinearProblem(nonlinearFunction_with_jac, U0_for_solver, p2; abstol = 1e-10, reltol = 1e-10)
sol5 = solve(prob_with_jac, NewtonRaphson(), maxiters=5)


using NonlinearSolve, LinearSolve, SparseArrays, Symbolics
ref = TriangleDiagELGL(order, Int(2 * order))
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 6" => ref)

ref_elems = Dict{String, SBPLite.RefElemData}(RefQuadrilateral => QuadRefElemLGL(2))

nel = (10, 10)
LL = (0.0, 0.0);
LR = (1.0, 0.0);
UR = (1.0, 1.0);
UL = (0.0, 1.0);

# grid = read_mesh(mesh_file, ref_elems)
grid = generate_grid(QuadraticQuadrilateral, ref_elems, nel, LL, LR, UR, UL)