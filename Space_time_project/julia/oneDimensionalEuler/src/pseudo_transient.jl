function set_IC(grid; WL = [3, 2, 10], WR = [3, -1, 5], γ = 1.4)

    IC = Dict()

    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_1")
        temp_u = zeros(length(grid.xyz_q[cell_id]), 3)
        temp_u[:, 1] .= WL[1]
        temp_u[:, 2] .= WL[1] * WL[2]
        temp_u[:, 3] .= WL[3] / (γ - 1) + 0.5  * WL[1] * WL[2]^2
        IC[(cell_id, face_id)] = temp_u
    end
    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_2")
        temp_u = zeros(length(grid.xyz_q[cell_id]), 3)
        temp_u[:, 1] .= WR[1]
        temp_u[:, 2] .= WR[1] * WR[2]
        temp_u[:, 3] .= WR[3] / (γ - 1) + 0.5  * WR[1] * WR[2]^2
        IC[(cell_id, face_id)] = temp_u
    end

    return IC


end


function space_time_RK4(U, pseudo_dt, p; diss = false, test = false, vol_diss_factor = 0.1, interface_diss_factor = 1.0)
    c1, c2, c3, c4 = 0.0, 0.40128709, 0.56449983, 0.87678807
    b1, b2, b3, b4 = 0.20334721, 0.19932974, 0.28339585, 0.31392720
    a21 = 0.40128709
    a31, a32 = 0.28224991, 0.28224991
    a41, a42, a43 = 0.25972925, 0.25479937, 0.36225945

    k1 = RHS(U, p, vol_diss_factor = vol_diss_factor, interface_diss_factor = interface_diss_factor)
    k2 = RHS(U + a21 * pseudo_dt * k1, p, vol_diss_factor = vol_diss_factor, interface_diss_factor = interface_diss_factor)
    k3 = RHS(U + a31 * pseudo_dt * k1 + a32 * pseudo_dt * k2, p, vol_diss_factor = vol_diss_factor, interface_diss_factor = interface_diss_factor)
    k4 = RHS(U + a41 * pseudo_dt * k1 + a42 * pseudo_dt * k2 + a43 * pseudo_dt * k3, p, vol_diss_factor = vol_diss_factor, interface_diss_factor = interface_diss_factor)

    result = U + pseudo_dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4)
    # if any(x -> x<0 , result[:,:,1]) || any(x -> x<0 , result[:,:,3])
    #     print("Negative value in space_time_RK4")
    #     return abs.(result)
    # end


    return positive_filtering(result)
end


function space_time_solve(U0::Array{Float64, 3}, p::Any, all_U::Vector{Array{Float64, 3}}, diffs::Vector{Float64}; pseudo_dt = 0.0001, num_of_pseudo_time_step = 100, num_of_variable = 3,  vol_diss_factor = 0.1, interface_diss_factor = 1.0)

    num_of_cells = n_cells(p[1])
    num_of_nodes = Int(length(p[1].xyz_q[1]))
    # all_U = ones(Float64, (num_of_pseudo_time_step+1, num_of_cells, num_of_nodes, num_of_variable))
    # all_U[1, :, :, :] = pseudo_initial_condtion(p[1])
    # all_U = Vector{Array{Float64, 3}}()
    # push!(all_U, U0)
    # diffs = zeros(num_of_pseudo_time_step)

    for pseudo_step in 1:num_of_pseudo_time_step
        # print(pseudo_dt, "\n")
        curr_U = deepcopy(all_U[pseudo_step])
        next_U = space_time_RK4(curr_U, pseudo_dt, p, vol_diss_factor = vol_diss_factor, interface_diss_factor = interface_diss_factor)
        
        max_diff = maximum(abs.(next_U - curr_U))
        println("Pseudo time step: $pseudo_step, Maximum difference: ", max_diff, "\n")
        diffs[pseudo_step] = max_diff
        push!(all_U, next_U)
        push!(diffs, max_diff)

        if max_diff > 100 || isnan(max_diff)
            print("\n Unstable at pseudo time step: $pseudo_step, pseudo_dt = $pseudo_dt \n")
            return all_U, diffs
        elseif max_diff < 1e-13
            # u = next_u
            # break
            print("\n Congratulation! Solution converged, pseudo_dt = $pseudo_dt \n")
            return all_U, diffs
        end

    end
    print("\n Solution not converged, pseudo_dt = $pseudo_dt \n")
    # return all_U, diffs
end