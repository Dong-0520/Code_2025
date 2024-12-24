function plot_cell_withShock(grid, cells)
    num_of_cells = length(cells)
    num_of_nodes = length(grid.xyz_q[1])
    x = Matrix{Float64}(undef, num_of_cells, num_of_nodes)
    y = Matrix{Float64}(undef, num_of_cells, num_of_nodes)

    @inbounds for i in 1:num_of_cells
        for node_id in 1:num_of_nodes
            x[i, node_id] = grid.xyz_q[cells[i]][node_id][1]
            y[i, node_id] = grid.xyz_q[cells[i]][node_id][2]
        end
    end

    p = scatter()
    all_x = Float64[]
    all_y = Float64[]
    
    @inbounds for i in 1:num_of_cells
        scatter!(p, x[i, :], y[i, :], markersize=3, label="")

        # 计算每个 cell 的中心位置，用于放置标签
        x_center = mean(x[i, :])
        y_center = mean(y[i, :])

        # 收集所有 x 和 y 值以计算范围
        append!(all_x, x[i, :])
        append!(all_y, y[i, :])
        
        # 使用 annotate! 添加文本标签
        annotate!(x_center, y_center, text(string(cells[i]), :black, 8))
    end

    # 使用收集的 x 和 y 数据计算范围
    x_min, x_max = extrema(all_x)
    y_min, y_max = extrema(all_y)

    # 设置红线的 x 范围和对应的 y 值
    x_vals = range(x_min, stop=x_max, length=100)
    y_vals = 2 * x_vals / 3

    # 画出红线
    plot!(x_vals, y_vals, color=:red, linewidth=1)

    # 设置图标题
    title!("Cells $cells")
    display(p)
end

function plot_cell(grid, cells)
    num_of_cells = length(cells)
    num_of_nodes = length(grid.xyz_q[1])
    x = Matrix{Float64}(undef, num_of_cells, num_of_nodes)
    y = Matrix{Float64}(undef, num_of_cells, num_of_nodes)

    @inbounds for i in 1:num_of_cells
        for node_id in 1:num_of_nodes
            x[i, node_id] = grid.xyz_q[cells[i]][node_id][1]
            y[i, node_id] = grid.xyz_q[cells[i]][node_id][2]
        end
    end

    p = scatter()
    all_x = Float64[]
    all_y = Float64[]
    
    @inbounds for i in 1:num_of_cells
        scatter!(p, x[i, :], y[i, :], markersize=3, label="")

        # 计算每个 cell 的中心位置，用于放置标签
        x_center = mean(x[i, :])
        y_center = mean(y[i, :])

        # 收集所有 x 和 y 值以计算范围
        append!(all_x, x[i, :])
        append!(all_y, y[i, :])
        
        # 使用 annotate! 添加文本标签
        annotate!(x_center, y_center, text(string(cells[i]), :black, 8))

        # 为每个点添加标签
        for node_id in 1:num_of_nodes
            annotate!(x[i, node_id], y[i, node_id], text(string(node_id), :blue, 10))
        end
    end


    # 设置图标题
    title!("Cells $cells")
    display(p)
end


function observe_iteration(grid, all_U, zlimit)
    """
    Draw a gif with x, y coordinates from grid, taking `all_U[i,:,:,1]` for plotting `ρ`.
    """
    num_of_cells = n_cells(grid)
    num_of_nodes = length(grid.xyz_q[1])
    num_of_pseudo_time_step = size(all_U, 1)

    # Pre-allocate x and y arrays
    x = Matrix{Float64}(undef, num_of_cells, num_of_nodes)
    y = Matrix{Float64}(undef, num_of_cells, num_of_nodes)

    @inbounds for cell_id in 1:num_of_cells
        for node_id in 1:num_of_nodes
            x[cell_id, node_id] = grid.xyz_q[cell_id][node_id][1]
            y[cell_id, node_id] = grid.xyz_q[cell_id][node_id][2]
        end
    end

    @inbounds anim = @animate for i in ProgressBar(1:num_of_pseudo_time_step)
        p = scatter3d()

        # Precompute analytical solution for each cell to avoid recomputation
        analytic_values = [analytic_u.(grid.xyz_q[cell_id]) for cell_id in 1:num_of_cells]

        for cell_id in 1:num_of_cells
            # Plot numerical and analytical solutions for this cell
            z1 = all_U[i, cell_id, :]
            scatter3d!(p, x[cell_id, :], y[cell_id, :], z1, markersize=1, label="", zlims=zlimit, color=:blue)
            scatter3d!(p, x[cell_id, :], y[cell_id, :], analytic_values[cell_id], markersize=1, label="", zlims=zlimit, color=:orange)
        end

        title!("Pseudo time step: $i")
    end

    return anim
end

function observe_iteration(grid, all_U, zlimit; show_true_sol = true)
    """
    Draw a gif with x, y coordinates from grid, taking `all_U[i,:,:,1]` for plotting `ρ`.
    """
    num_of_cells = n_cells(grid)
    num_of_nodes = length(grid.xyz_q[1])
    num_of_pseudo_time_step = size(all_U, 1)

    # Pre-allocate x and y arrays
    x = Matrix{Float64}(undef, num_of_cells, num_of_nodes)
    y = Matrix{Float64}(undef, num_of_cells, num_of_nodes)

    @inbounds for cell_id in 1:num_of_cells
        for node_id in 1:num_of_nodes
            x[cell_id, node_id] = grid.xyz_q[cell_id][node_id][1]
            y[cell_id, node_id] = grid.xyz_q[cell_id][node_id][2]
        end
    end

    @inbounds anim = @animate for i in ProgressBar(1:num_of_pseudo_time_step)
        p = scatter3d()

        # Precompute analytical solution for each cell to avoid recomputation
        analytic_values = [analytic_u.(grid.xyz_q[cell_id]) for cell_id in 1:num_of_cells]

        for cell_id in 1:num_of_cells
            # Plot numerical and analytical solutions for this cell
            z1 = all_U[i, cell_id, :]
            scatter3d!(p, x[cell_id, :], y[cell_id, :], z1, markersize=1, label="", zlims=zlimit, color=:blue)
            if show_true_sol
                scatter3d!(p, x[cell_id, :], y[cell_id, :], analytic_values[cell_id], markersize=1, label="", zlims=zlimit, color=:orange)
            end
        end

        title!("Pseudo time step: $i")
    end

    return anim
end



function observe_numerical_sol(grid::Grid, all_U::Array{Float64, 3}, zlimit::Vector{Float64})
    """
    plot the final iteration only, which is the numerical solution
    """
    num_of_cells = n_cells(grid)
    num_of_nodes = Int(length(grid.xyz_q[1]))
    p = scatter3d()
        
    for cell_id in 1:num_of_cells
        x = []
        y = []
        for node_id in 1:num_of_nodes
            push!(x, grid.xyz_q[cell_id][node_id][1])
            push!(y, grid.xyz_q[cell_id][node_id][2])
        end
        z1 = all_U[end, cell_id, :]
        scatter3d!(p, x, y, z1, markersize=1, label = "", zlims = zlimit)
        scatter3d!(p, x, y, analytic_u.(grid.xyz_q[cell_id]), markersize=1, label = "", zlims = zlimit, color = :red, size = (800, 800))
    end
    title!("Numerical solution and analytic solution")
end

function observe_numerical_sol(grid::Grid, u::Union{Array{Float64, 2}, Adjoint{Float64, Matrix{Float64}}}, zlimit::Vector{Float64}; show_true_sol = true)
    num_of_cells = n_cells(grid)
    num_of_nodes = Int(length(grid.xyz_q[1]))
    p = scatter3d()
        
    for cell_id in 1:num_of_cells
        x = []
        y = []
        for node_id in 1:num_of_nodes
            push!(x, grid.xyz_q[cell_id][node_id][1])
            push!(y, grid.xyz_q[cell_id][node_id][2])
        end
        z1 = u[cell_id, :]
        if show_true_sol
            scatter3d!(p, x, y, z1, markersize=1, label = "", zlims = zlimit)
            scatter3d!(p, x, y, analytic_u.(grid.xyz_q[cell_id]), markersize=1, label = "", zlims = zlimit, color = :red, size = (800, 800))
        else
            scatter3d!(p, x, y, z1, markersize=1, label = "", zlims = zlimit, color = :red, size = (800, 800))
        end
    end
    title!("Numerical solution and analytic solution")
end