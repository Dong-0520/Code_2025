

# 这里记住，所有的analytic solution 都假设了是 [0, 1] × [0, 1] 的区域
# mesh 也假定了 沿 1，1 advection vector 的方向
# 如果要改analytical solution，这里两个bottom和left，以及后面的 analytical_solution_for_grid 都要改

# assume the left analytic_solution is -cos( π (x - t) ) - 1
# assume the right analytic_solution is cos( π (x - t) ) + 1
ax, at = (1.0, 1.0)

bottomBC1(x::Coord) = -cospi(x[1]) - 1
# bottomBC2(x::Coord) = cospi(x[1]) + 1
leftBC(x::Coord) = -cospi(0 - ax * x[2]) - 1


function RHS(u, p, t)
    grid, interfaces_align_shock, interior_interfaces, IC = p
    result = Matrix{Float64}(undef, n_cells(grid), nodes_per_cell)

    for cell_id in 1:n_cells(grid)
        Dx = grid.VOL[cell_id][1]
        Dt = grid.VOL[cell_id][2]
        

        local_sol = @view u[cell_id, :]
        ux = ax * Dx * local_sol
        ut = at * Dt * local_sol

        @. result[cell_id, :] = -ux - ut
    end

    for interface in grid.face_interfaces
        c1, lf1 = interface.face_1
        c2, lf2 = interface.face_2
        P1, P2 = interface.P1, interface.P2
        cell1, cell2 = get_cells(grid, [c1, c2])
        elem1, elem2 = get_ref_elem_data(grid, cell1), get_ref_elem_data(grid, cell2)
        R_1, R_2 = elem1.R[lf1], elem2.R[lf2]
        mask1, mask2 = @views elem1.f_mask[lf1], elem2.f_mask[lf2]

        # normal vector of physical element
        normal_1, normal_2 = @views grid.geometric_terms.N_f[c1][:, mask1], grid.geometric_terms.N_f[c2][:, mask2]

        an1, an2 = normal_1 .* a, normal_2 .* a
        lambda_1, lambda_2 = an1[1, :] + an1[2, :], an2[1, :] + an2[2, :]

        u1_face, u2_face = R_1 * u[c1, :], R_2 * u[c2, :]
        u1_adj, u2_adj = u2_face[P2], u1_face[P1]
        flux_1 = 0.5 * ((lambda_1 .+ abs.(lambda_1)) .* u1_face .+ (lambda_1 .- abs.(lambda_1)) .* u1_adj)
        flux_2 = 0.5 * ((lambda_2 .+ abs.(lambda_2)) .* u2_face .+ (lambda_2 .- abs.(lambda_2)) .* u2_adj)
        result[c1, :] += elem1.H_inv * Matrix(R_1)' * elem1.H_face * (lambda_1 .* u1_face .- flux_1)
        result[c2, :] += elem2.H_inv * Matrix(R_2)' * elem2.H_face * (lambda_2 .* u2_face .- flux_2)
    end

    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_1")
        cell = get_cells(grid, cell_id)
        elem = get_ref_elem_data(grid, cell)
        R = elem.R[face_id]
        mask = @views elem.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        an = normal .* a
        gamma = an[1, :] + an[2, :]

        u_face = R * u[cell_id, :]
        u_adj = bottomBC1.(R * grid.xyz_q[cell_id])
        flux = gamma .* u_adj
        result[cell_id, :] += elem.H_inv * Matrix(R)' * elem.H_face * (gamma .* u_face .- flux)
    end

    for (cell_id, face_id) in get_face_set(grid, "BOTTOM_INFLOW_2")
        cell = get_cells(grid, cell_id)
        elem = get_ref_elem_data(grid, cell)
        R = elem.R[face_id]
        mask = @views elem.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        an = normal .* a
        gamma = an[1, :] + an[2, :]

        u_face = R * u[cell_id, :]
        u_adj = bottomBC2.(R * grid.xyz_q[cell_id])
        flux = gamma .* u_adj
        result[cell_id, :] += elem.H_inv * Matrix(R)' * elem.H_face * (gamma .* u_face .- flux)
    end

    for (cell_id, face_id) in get_face_set(grid, "LEFT_INFLOW")
        cell = get_cells(grid, cell_id)
        elem = get_ref_elem_data(grid, cell)
        R = elem.R[face_id]
        mask = @views elem.f_mask[face_id]
        normal = grid.geometric_terms.N_f[cell_id][:, mask]
        an = normal .* a
        gamma = an[1, :] + an[2, :]

        u_face = R * u[cell_id, :]
        u_adj = leftBC.(R * grid.xyz_q[cell_id])
        flux = gamma .* u_adj
        result[cell_id, :] += elem.H_inv * Matrix(R)' * elem.H_face * (gamma .* u_face .- flux)
    end

    return result
end



function pseudo_time_step_RK4(u, p, t, pseudo_dt)
    c1, c2, c3, c4 = 0.0, 0.40128709, 0.56449983, 0.87678807
    b1, b2, b3, b4 = 0.20334721, 0.19932974, 0.28339585, 0.31392720
    a21 = 0.40128709
    a31, a32 = 0.28224991, 0.28224991
    a41, a42, a43 = 0.25972925, 0.25479937, 0.36225945

    RK1 = RHS(u, p, t)
    RK2 = RHS(u + a21 * pseudo_dt * RK1, p, t + c2 * pseudo_dt)
    RK3 = RHS(u + a31 * pseudo_dt * RK1 + a32 * pseudo_dt * RK2, p, t + c3 * pseudo_dt)
    RK4 = RHS(u + a41 * pseudo_dt * RK1 + a42 * pseudo_dt * RK2 + a43 * pseudo_dt * RK3, p, t + c4 * pseudo_dt)

    return u + pseudo_dt * (b1 * RK1 + b2 * RK2 + b3 * RK3 + b4 * RK4)  
end

function get_set_up(grid::Grid, ax, at)
    Dx_all, Dt_all, invJ_all = get_Dx_Dt_invJ_for_all_cell_curvilinear(grid)
    a = [ax, at]
    p = (grid, Dx_all, Dt_all, invJ_all, a)
    return p
end


nodes_per_cell = length(grid.xyz_q[1])
u = Matrix{Float64}(undef, n_cells(grid), nodes_per_cell)
curr_u = Matrix{Float64}(undef, n_cells(grid), nodes_per_cell)
next_u = Matrix{Float64}(undef, n_cells(grid), nodes_per_cell)
fill!(u, 0.0)
fill!(curr_u, 0.0)
fill!(next_u, 0.0)
num_of_pseudo_time_step = 1500
pseudo_dt = 5.0
p = get_set_up(grid, ax, at)
for pseudo_step in 1:num_of_pseudo_time_step
    curr_u = deepcopy(u)
    next_u = pseudo_time_step_RK4(curr_u, p, 0.0, pseudo_dt)
    println("Pseudo time step: $pseudo_step, Maximum difference: ", maximum(abs.(next_u - curr_u)), "\n")
    max_diff = maximum(abs.(next_u - curr_u))
    if max_diff >5
        println("Divergence detected \n")
        pseudo_dt -= 0.01
        u = Matrix{Float64}(undef, n_cells(grid), nodes_per_cell)
        curr_u = Matrix{Float64}(undef, n_cells(grid), nodes_per_cell)
        next_u = Matrix{Float64}(undef, n_cells(grid), nodes_per_cell)
        fill!(u, 0.0)
        fill!(curr_u, 0.0)
        fill!(next_u, 0.0)
    elseif max_diff < 1e-10
        u = next_u
        break
    else
        u = next_u
    end
end


plot_solution(grid, u)

RHS(u, p, 0.0)


# Or plot by PlotlyJS

using PlotlyJS

using PlotlyJS

function plot_solution_interactive(grid, u)
    nodes_per_cell = length(grid.xyz_q[1])
    
    # 定义存储 x, y, z 的数组
    x = []
    y = []
    z = []
    
    # 遍历所有的单元格
    for cell_id in 1:n_cells(grid)
        for node_id in 1:nodes_per_cell
            push!(x, grid.xyz_q[cell_id][node_id][1])
            push!(y, grid.xyz_q[cell_id][node_id][2])
            push!(z, u[cell_id, node_id])
        end
    end
    
    # 使用 PlotlyJS 创建交互式3D散点图
    scatter_trace = PlotlyJS.scatter3d(
        x = x,
        y = y,
        z = z,
        mode = "markers",
        marker = attr(
            size = 2,
            color = z,                # 使用 z 值进行颜色渐变
            colorscale = "Viridis",   # 使用 Viridis 渐变色表
            opacity = 0.8
        )
    )

    layout = PlotlyJS.Layout(
        title = "3D Interactive Solution",
        scene = PlotlyJS.attr(
            xaxis_title = "X Axis",
            yaxis_title = "Y Axis",
            zaxis_title = "Z Axis"
        )
    )

    # 使用 PlotlyJS 的 plot 函数
    PlotlyJS.plot(scatter_trace, layout)
end


# 调用绘图函数
p=plot_solution_interactive(grid, u)
PlotlyJS.savefig(p, "solution.png")

# convrgence study for future use
function analytical_solution_left(x)
    -cospi(x[1] - ax * x[2]) - 1
end

function analytical_solution_right(x)
    cospi(x[1] - ax * x[2]) + 1
end

function analytic_solution_for_grid(grid)
    for cell_id in 1:n_cells(grid)
        cell = get_cells(grid, cell_id)
        elem = get_ref_elem_data(grid, cell)
        coords = grid.xyz_q[cell_id]
        if all(x -> isapprox(x[1] - x[2], 0.25, atol=1e-5) || x[1] - x[2] < 0.25, coords)
            u[cell_id, :] = analytical_solution_left.(coords)
        end
        if all(x -> isapprox(x[1] - x[2], 0.25, atol=1e-5) || x[1] - x[2] > 0.25, coords)
            u[cell_id, :] = analytical_solution_right.(coords)
        end
    end
    u
end

analytical_u = analytic_solution_for_grid(grid)

plot_solution_interactive(grid,abs.(analytical_u- u))
