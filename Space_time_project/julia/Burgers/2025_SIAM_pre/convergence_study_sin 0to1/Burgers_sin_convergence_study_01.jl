using LinearAlgebra, SparseArrays
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics
using SparseConnectivityTracer, ADTypes
using NonlinearSolve                                                                                                                                                                                                                                                          

include("../../../src/SBPLite.jl")
using .SBPLite
include("../../plotting_helper.jl")

order = 4
ref = TriangleDiagELGL(order, 2 * order - 1)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)


# This is random larger mesh from 0 to 0.15
grid = read_mesh(joinpath(@__DIR__, "Burgers_sin_12.msh"), ref_elems_data, Base.identity)
zlimit = [-1.1, 1.1]
nodes_per_cell = length(grid.xyz_q[1])

# ---------------------- slab 1 ---------------------- #
p1 = get_set_up(grid, slab = 1)
U0 = initial_guess(grid)
@time all_U, diffs = space_time_solve(U0, p1, pseudo_dt = 0.001, num_of_pseudo_time_step = 1000, diss = true, test = true)
# gif(observe_iteration(grid, all_U[1:10:end], zlimit, show_true_sol = false), fps = 5)
u1 = all_U[argmin(diffs)]
sol_slab1 = solver_wrapper(u1, p1, maximum_iteration = 100, solver = NewtonRaphson())
u1 = sol_slab1.u
#---------------------- slab 2 ---------------------- #
p2 = get_set_up(grid, slab = 2, u = u1)
all_U2, diffs2 = space_time_solve(u1, p2, pseudo_dt = 0.0001, num_of_pseudo_time_step = 4000, diss = true, test = true)
@time all_U2, diffs2 = space_time_solve(all_U2[end], p2, pseudo_dt = 0.00001, num_of_pseudo_time_step = 2000, diss = true, test = true)

gif(observe_iteration(grid, all_U2[1:20:end], zlimit, show_true_sol = false), fps = 5)

sol_slab2 = solver_wrapper(all_U2[end], p2, maximum_iteration = 100, solver = NewtonRaphson())

u2 = sol_slab2.u
observe_numerical_sol(grid, u2, zlimit, show_true_sol = false)

#---------------------- convergence study ---------------------- #
x_coords = get_xCoords_for_plotting_sol(grid)
u = solve_u(x_coords, 0.1, tol = 8e-12)
num_u = get_final_u_for_plotting_sol(grid, u4)
Pnorms = get_Pnorm_for_finalTimeInterface(grid)

plot_analytic_num_error(u, x_coords, num_u)
# error_u = Dict()
# for k in keys(u)
#     error_u[k] = num_u[k] - u[k]
# end
# plot_dict_u(error_u, x_coords)

L2_error = compute_L2_error_atFianlInterface(u, num_u, Pnorms)


grid4 = read_mesh(joinpath(@__DIR__, "Burgers_sin_4.msh"), ref_elems_data, Base.identity)
grid5 = read_mesh(joinpath(@__DIR__, "Burgers_sin_5.msh"), ref_elems_data, Base.identity)
grid6 = read_mesh(joinpath(@__DIR__, "Burgers_sin_6.msh"), ref_elems_data, Base.identity)
grid9 = read_mesh(joinpath(@__DIR__, "Burgers_sin_9.msh"), ref_elems_data, Base.identity)
grid10 = read_mesh(joinpath(@__DIR__, "Burgers_sin_10.msh"), ref_elems_data, Base.identity)
grid15 = read_mesh(joinpath(@__DIR__, "Burgers_sin_15.msh"), ref_elems_data, Base.identity)
grid20 = read_mesh(joinpath(@__DIR__, "Burgers_sin_20.msh"), ref_elems_data, Base.identity)
dxs_ave_length = [get_average_length_of_sides(grid4), get_average_length_of_sides(grid5), get_average_length_of_sides(grid6), get_average_length_of_sides(grid9), get_average_length_of_sides(grid10), get_average_length_of_sides(grid15), get_average_length_of_sides(grid20)]
errors = [0.001247630484234621, 0.003819816045301901, 0.0038198160453019473, 0.004666942519959309, 0.004666942519959333, 0.009570746071266758, 0.013335902892093356]
fit(log10.(dxs_ave_length[[1,2,6]]), log10.(errors[[1,2,6]]),1)[1]




p = scatter(log10.(dxs_ave_length), log10.(errors), 
           seriestype = :scatter, label = "data points")
# Add number labels to points
for i in 1:length(dxs_ave_length)
    annotate!(p, 
             [(log10(dxs_ave_length[i]), log10(errors[i]), text("$i", :black, :top, 8))])
end
display(p)
fit(log10.(dxs_ave_length[[1,2,3]]), log10.(errors[[1,2,3]]),1)[1]



# entropy datas
x_coords = get_xCoords_for_plotting_sol(grid)

num_u1 = get_final_u_for_plotting_sol(grid, u1)
num_u2 = get_final_u_for_plotting_sol(grid, u2)
num_u3 = get_final_u_for_plotting_sol(grid, u3)
num_u4 = get_final_u_for_plotting_sol(grid, u4)
Pnorms = get_Pnorm_for_finalTimeInterface(grid)
entropys = [compute_L2_norm_of_Initialcondition(p1),compute_L2_norm_atFinalInterface(num_u1, Pnorms), compute_L2_norm_atFinalInterface(num_u2, Pnorms), compute_L2_norm_atFinalInterface(num_u3, Pnorms), compute_L2_norm_atFinalInterface(num_u4, Pnorms)]


# order 2
# mesh 2 = [ 0.5, 0.49999994718779567, 0.49999920628289946, 0.49996210289240395, 0.4888040457178535]
# mesh 8 = [ 0.5, 0.4999694622730587, 0.4996412588197546, 0.498600835493074, 0.48970901817126367]
# mesh 10 = [0.49999999999999994, 0.4998807924643948, 0.4996397113873916, 0.4981568057890972, 0.489418849877316]







# ---------------------- compute convergence rate ---------------------- #
grid2 = read_mesh(joinpath(@__DIR__, "Burgers_sin_2.msh"), ref_elems_data, Base.identity)
grid4 = read_mesh(joinpath(@__DIR__, "Burgers_sin_4.msh"), ref_elems_data, Base.identity)
grid5 = read_mesh(joinpath(@__DIR__, "Burgers_sin_5.msh"), ref_elems_data, Base.identity)
grid6 = read_mesh(joinpath(@__DIR__, "Burgers_sin_6.msh"), ref_elems_data, Base.identity)
grid8 = read_mesh(joinpath(@__DIR__, "Burgers_sin_8.msh"), ref_elems_data, Base.identity)
grid9 = read_mesh(joinpath(@__DIR__, "Burgers_sin_9.msh"), ref_elems_data, Base.identity)
grid10 = read_mesh(joinpath(@__DIR__, "Burgers_sin_10.msh"), ref_elems_data, Base.identity)
dxs_ave_length = [get_average_length_of_sides(grid2), get_average_length_of_sides(grid4), get_average_length_of_sides(grid5), get_average_length_of_sides(grid6), get_average_length_of_sides(grid8), get_average_length_of_sides(grid9), get_average_length_of_sides(grid10)]
errors = [0.0004401726359496635, 0.0012476304842346994, 0.003819816045301837, 0.003819816045301836, 0.006658577234714387, 0.004666942519959153, 0.005839954573884097]


grid6 = read_mesh(joinpath(@__DIR__, "Burgers_sin_6.msh"), ref_elems_data, Base.identity)
grid8 = read_mesh(joinpath(@__DIR__, "Burgers_sin_8.msh"), ref_elems_data, Base.identity)
grid9 = read_mesh(joinpath(@__DIR__, "Burgers_sin_9.msh"), ref_elems_data, Base.identity)
grid10 = read_mesh(joinpath(@__DIR__, "Burgers_sin_10.msh"), ref_elems_data, Base.identity)
grid11 = read_mesh(joinpath(@__DIR__, "Burgers_sin_11.msh"), ref_elems_data, Base.identity)
grid12 = read_mesh(joinpath(@__DIR__, "Burgers_sin_12.msh"), ref_elems_data, Base.identity)
dxs_ave_length = [get_average_length_of_sides(grid6), get_average_length_of_sides(grid8), get_average_length_of_sides(grid9), get_average_length_of_sides(grid10), get_average_length_of_sides(grid11), get_average_length_of_sides(grid12)]
errors = [0.002261348385514659, 0.00531671638890342, 0.005839954573884054, 0.005839954573883938, 0.005839954573883909, 0.0052534599124164855]


plot(log10.(dxs_ave_length), log10.(errors), seriestype = :scatter, label = "data points")
fit(log10.(dxs_ave_length[1:3]), log10.(errors[1:3]),1)[1]

# ---------------------- functions you need to run ---------------------- #
function analytic_u(x::Coord)
    x1, t1 = x[1], x[2]
    return sinpi(2 * x1)
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
            interface = nothing 
            for (j1, j2) in get_face_set(grid, "TOP_INFLOW_1")
                if j1 in neighbors_cells
                    try
                        interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((j1, j2)))
                        break
                    catch e
                        continue
                    end
                end
            end
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
            interface = nothing 
            for (j1, j2) in get_face_set(grid, "TOP_INFLOW_2")
                if j1 in neighbors_cells
                    try
                        interface = find_FaceInterface(grid, FaceIndex((cell_id, face_id)), FaceIndex((j1, j2)))
                        break
                    catch e
                        continue
                    end
                end
            end
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
    all_U = Vector{Matrix{Float64}}()
    push!(all_U, U0)
    diffs = Vector{Float64}()

    for pseudo_step in 1:num_of_pseudo_time_step
        curr_U = deepcopy(all_U[pseudo_step])
        next_U = space_time_RK4(curr_U, 0.0, pseudo_dt, p, diss = diss, test = test)
        max_diff = maximum(abs.(next_U - curr_U))
        push!(all_U, next_U)
        push!(diffs, max_diff)

        if max_diff < 1e-12
            println("\n Solution converged at pseudo time step: $pseudo_step \n")
            return all_U, diffs
        elseif max_diff > 100
            println("Solution diverges at pseudo time step: $pseudo_step \n")
            return all_U, diffs
        else
            println("Pseudo time step: $pseudo_step, max diff: ", max_diff, "\n")
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
        
        # 替代布尔判断的代码
        indicator = Nxγk .* (uk_face - uk_adj)
        spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4 .* (indicator .>= 0) + ((uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 .+ mysign.(Nxγk) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)) .* (indicator .< 0)

        # 同样的逻辑适用于 spatial_flux_v
        # indicator_v = Nxγv .* (uv_face - uv_adj)
        # spatial_flux_v = (uv_face.^2 + uv_adj.^2) ./ 4 .* (indicator_v .>= 0) + ((uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 .+ mysign.(Nxγv) .* max.(abs.(uv_face), abs.(uv_adj)) .* (uv_face .- uv_adj)) .* (indicator_v .< 0)

        rk_spatial = grid.FAC[ck][lfγk] * Diagonal(Nxγk) * (0.5 .* uk_face.^2 .- spatial_flux_k)
        rv_spatial = grid.FAC[cv][lfγv] * Diagonal(Nxγv) * (0.5 .* uv_face.^2 .- spatial_flux_k[Pk])

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

function RHS_test(u, p)
    grid, interfaces_align_shock, interior_interfaces, IC = p
    interior_interfaces = collect(interior_interfaces)
    result = zeros(n_cells(grid), nodes_per_cell)

    # volume discretization
    Threads.@threads for cell_id in 1:n_cells(grid)
        
        Dx = grid.VOL[cell_id][1]
        Dt = grid.VOL[cell_id][2]
        

        local_sol = u[cell_id, :]
        ux = 1/3 * Dx * (local_sol .^ 2) .+ 1/3 * local_sol .* (Dx * local_sol)
        ut = Dt * local_sol 

        result[cell_id, :] = (-ux .- ut)
    end

    Threads.@threads for i in 1:length(interior_interfaces)
        interface = interior_interfaces[i]
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

    
        # 替代布尔判断的代码
        indicator = Nxγk .* (uk_face - uk_adj)
        spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4 .* (indicator .>= 0) + ((uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 .+ mysign.(Nxγk) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)) .* (indicator .< 0)
        # spatial_flux_k = (uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6
        # spatial_flux_k = (uk_face.^2 + uk_adj.^2) ./ 4 .* (indicator .>= 0) + ((uk_face.^2 + uk_face .* uk_adj + uk_adj.^2) ./ 6 ) .* (indicator .< 0) .+ mysign.(Nxγk) .* max.(abs.(uk_face), abs.(uk_adj)) .* (uk_face .- uk_adj)

        # 同样的逻辑适用于 spatial_flux_v
        # indicator_v = Nxγv .* (uv_face - uv_adj)
        # spatial_flux_v = (uv_face.^2 + uv_adj.^2) ./ 4 .* (indicator_v .>= 0) + ((uv_face.^2 + uv_face .* uv_adj + uv_adj.^2) ./ 6 .+ mysign.(Nxγv) .* max.(abs.(uv_face), abs.(uv_adj)) .* (uv_face .- uv_adj)) .* (indicator_v .< 0)

        rk_spatial = grid.FAC[ck][lfγk] * Diagonal(Nxγk) * (0.5 .* uk_face.^2 .- spatial_flux_k)
        rv_spatial = grid.FAC[cv][lfγv] * Diagonal(Nxγv) * (0.5 .* uv_face.^2 .- spatial_flux_k[Pk])


        rk_temporal = grid.FAC[ck][lfγk] * Diagonal(Ntγk) * (0.5 .* uk_face .- 0.5 .* uk_adj)
        rv_temporal = grid.FAC[cv][lfγv] * Diagonal(Ntγv) * (0.5 .* uv_face .- 0.5 .* uv_adj)

        
        result[ck, :] += ( rk_spatial .+ rk_temporal )
        result[cv, :] += ( rv_spatial .+ rv_temporal )
    end


    bottom_faces = collect(union(get_face_set(grid, "BOTTOM_INFLOW_1"), get_face_set(grid, "BOTTOM_INFLOW_2")))
    Threads.@threads for i in 1:length(bottom_faces)
        cell_id, face_id = bottom_faces[i]
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

using Roots


function get_xCoords_for_plotting_sol(grid)
    """
    this function return the x coordinates for plotting numerical solution at each final time of slabs
    """
    TOP_INFLOW = collect(union(get_face_set(grid, "TOP_INFLOW_1"), get_face_set(grid, "TOP_INFLOW_2")))
    ref = grid.cells[TOP_INFLOW[1][1]].ref_data[]
    coords = Dict()
    for (cell_id, face_id) in TOP_INFLOW
        R = ref.R[face_id]
        # mask = ref.f_mask[face_id]
        x = map(x -> x[1], R * grid.xyz_q[cell_id])
        coords[cell_id] = x
    end
    return coords

end

function get_final_u_for_plotting_sol(grid, u)
    TOP_INFLOW = collect(union(get_face_set(grid, "TOP_INFLOW_1"), get_face_set(grid, "TOP_INFLOW_2")))
    ref = grid.cells[TOP_INFLOW[1][1]].ref_data[]
    num_sol = Dict()
    for (cell_id, face_id) in TOP_INFLOW
        R = ref.R[face_id]
        # mask = ref.f_mask[face_id]
        temp_u = map(x -> x[1], R * u[cell_id, :])
        num_sol[cell_id] = temp_u
    end
    return num_sol
    
end

function get_Pnorm_for_finalTimeInterface(grid)
    """
    works for straight interfaces
    """
    TOP_INFLOW = collect(union(get_face_set(grid, "TOP_INFLOW_1"), get_face_set(grid, "TOP_INFLOW_2")))
    ref = grid.cells[TOP_INFLOW[1][1]].ref_data[]
    Pnorm = Dict()
    for (cell_id, face_id) in TOP_INFLOW
        R = ref.R[face_id]
        face_coords = R * grid.xyz_q[cell_id]
        @assert face_coords[1][2] ≈ face_coords[end][2]
        Pnorm[cell_id] = (abs(face_coords[1][1] - face_coords[end][1]) / 2) * ref.H_face
        @assert sum(diag(Pnorm[cell_id])) ≈ abs(face_coords[1][1] - face_coords[end][1])
    end
    return Pnorm
end



function solve_single_point(x::Float64, t::Float64)
    # Function that returns another function with x,t fixed
    F(u) = u - sin(4π*x - 4π*u*t)
    return find_zero(F, (-1.0, 1.0), Roots.Bisection(), atol=1e-10)
end

function solve_u(xs_dict, t; u_init::Float64 = 0.0, tol = 1e-13)
    U = Dict()
    for cell_id in collect(keys(xs_dict))
        xs = xs_dict[cell_id]
        temp_U = []
        for x in xs
            if x ≈ 0.25 && xs[2] > 0.25
                push!(temp_U, solve_single_point(x+tol, t))
            elseif x ≈ 0.25 && xs[2] < 0.25
                push!(temp_U, solve_single_point(x-tol, t))
            else
                push!(temp_U, solve_single_point(x, t))
            end
        end
        # temp_U = [solve_single_point(x, t) for x in xs]
        U[cell_id] = temp_U
    end
    return U
end

function compute_L2_error_atFianlInterface(analytic_u, num_u, Pnorms)
    error = 0.0
    for cell_id in collect(keys(analytic_u))
        local_error = analytic_u[cell_id] - num_u[cell_id]
        error += local_error' * Pnorms[cell_id] * local_error
    end
    return sqrt(error)
    
end

function compute_L2_norm_atFinalInterface(num_u, Pnorms)
    norm = 0.0
    for cell_id in collect(keys(num_u))
        local_norm = num_u[cell_id]
        norm += local_norm' * Pnorms[cell_id] * local_norm
    end
    return sqrt(norm)
end

function compute_L2_norm_of_Initialcondition(p)
    """
    p is the set up for the problem for the first slab, p[end] store the cellid, faceid and u0
    """
    IC = p[end]
    grid = p[1]
    norm = 0.0
    for (cell_id, face_id) in collect(keys(IC))

        # compute local Pγ
        ref = grid.cells[cell_id].ref_data[]
        R = ref.R[face_id]
        face_coords = R * grid.xyz_q[cell_id]
        @assert face_coords[1][2] ≈ face_coords[end][2]
        Pγ = (abs(face_coords[1][1] - face_coords[end][1]) / 2) * ref.H_face
        @assert sum(diag(Pγ)) ≈ abs(face_coords[1][1] - face_coords[end][1])
        u_face = IC[(cell_id, face_id)]
        norm += u_face' * Pγ * u_face
    end
    return sqrt(norm)
end

function plot_analytic_num_error(analytic_u, x_coords, u_numeric)
    """
    Plot analytic solution, numeric solution and error in horizontal layout
    """
    # Create layout with 3 horizontally arranged subplots
    error_u = Dict(k => num_u[k] - u[k] for k in keys(u))
    p = plot(layout=(1,3), size=(1200,400))
    
    # Analytic solution
    for cell_id in collect(keys(analytic_u))
        scatter!(p[1], x_coords[cell_id], analytic_u[cell_id], 
                label="", color="red", title="Analytic Solution")
    end
    
    # Numeric solution
    for cell_id in collect(keys(u_numeric))
        scatter!(p[2], x_coords[cell_id], u_numeric[cell_id], 
                label="", color="blue", title="Numeric Solution")
    end
    
    # Error
    for cell_id in collect(keys(error_u))
        scatter!(p[3], x_coords[cell_id], error_u[cell_id], 
                label="", color="green", title="Error")
    end
    
    return p
end


using Polynomials

function get_sidelength_perCell(grid::Grid, cell::AbstractCell)

    vertices_IDS = vertices(cell)
    vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
    vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
    vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

    v1x = vertice1[1]
    v1y = vertice1[2]
    v2x = vertice2[1]
    v2y = vertice2[2]
    v3x = vertice3[1]
    v3y = vertice3[2]

    l1 = sqrt((v1x - v2x)^2 + (v1y - v2y)^2)
    l2 = sqrt((v2x - v3x)^2 + (v2y - v3y)^2)
    l3 = sqrt((v3x - v1x)^2 + (v3y - v1y)^2)

    return l1, l2, l3
    
end

function get_average_length_of_sides(grid)
    num_of_cells = n_cells(grid)
    total_length = 0.0
    for i in 1:num_of_cells
        l1, l2, l3 = get_sidelength_perCell(grid, grid.cells[i])
        total_length += l1 + l2 + l3
    end
    return total_length / (3 * num_of_cells)

end

function solver_wrapper(U0_for_solver, pi; maximum_iteration = 1000, solver = NewtonRaphson())
    f! = (du, u) -> RHS_forSolver_diss(du, u, pi)
    du0 = similar(U0_for_solver)
    jac_sparsity_diss = ADTypes.jacobian_sparsity(f!, du0, U0_for_solver, TracerSparsityDetector())
    nonlinearFunction_with_jac_diss = NonlinearFunction(RHS_forSolver_diss; jac_prototype=jac_sparsity_diss)
    prob_with_jac_diss = NonlinearProblem(nonlinearFunction_with_jac_diss, U0_for_solver, pi; abstol=1e-13, reltol=1e-15)
    sol = solve(prob_with_jac_diss, solver,show_trace = Val(true), maxiters = maximum_iteration)
    return sol
end