


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

function plot_grid(grid)
    num_of_cells = n_cells(grid)
    num_of_nodes = length(grid.xyz_q[1])
    x = Matrix{Float64}(undef, num_of_cells, num_of_nodes)
    y = Matrix{Float64}(undef, num_of_cells, num_of_nodes)

    @inbounds for i in 1:num_of_cells
        for node_id in 1:num_of_nodes
            x[i, node_id] = grid.xyz_q[i][node_id][1]
            y[i, node_id] = grid.xyz_q[i][node_id][2]
        end
    end

    p = scatter()
    
    @inbounds for i in 1:num_of_cells
        # Close the loop by appending the first node to the end
        x_closed = vcat(x[i, :], x[i, 1])
        y_closed = vcat(y[i, :], y[i, 1])
        plot!(p, x_closed, y_closed, seriestype = :shape, fillalpha = 0.3, label="", color = :blue)
    end

    display(p)
end


function observe_iteration(grid, all_U, zlimit)
    """
    Draw a gif with x, y coordinates from grid, taking `all_U[i,:,:,1]` for plotting `ρ`.
    """
    num_of_cells = n_cells(grid)
    num_of_nodes = length(grid.xyz_q[1])
    num_of_pseudo_time_step = size(all_U)[1]

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
            z1 = all_U[i][cell_id, :]
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
    num_of_pseudo_time_step = size(all_U)[1]
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

        

        for cell_id in 1:num_of_cells
            # Plot numerical and analytical solutions for this cell
            z1 = all_U[i][cell_id, :]
            scatter3d!(p, x[cell_id, :], y[cell_id, :], z1, markersize=1, label="", zlims=zlimit, color=:blue)
            if show_true_sol
                # Precompute analytical solution for each cell to avoid recomputation
                analytic_values = [analytic_u.(grid.xyz_q[cell_id]) for cell_id in 1:num_of_cells]
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


function plot_mesh(grid; xlimit = [-1.1, 1.1], ylimit = [-1.1, 1.1], size = (800, 800))
    num_of_cells = n_cells(grid)
    num_of_nodes = Int(length(grid.xyz_q[1]))
    p = scatter(xlabel = "x", ylabel = "y")

    # 计算网格范围
    x_min, x_max = xlimit
    y_min, y_max = ylimit
    x_range = x_max - x_min
    y_range = y_max - y_min

    # 自适应 size 让图片紧凑
    # width = 800 + 400 * (x_range / y_range)
    # height = 600
    # size = (round(Int, width), round(Int, height))

    # 画网格
    for cell_id in 1:num_of_cells
        vertices_IDS = vertices(grid.cells[cell_id])
        vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
        vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
        vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

        v1x, v1y = vertice1[1], vertice1[2]
        v2x, v2y = vertice2[1], vertice2[2]
        v3x, v3y = vertice3[1], vertice3[2]

        plot!(p, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
              label="", color=:blue, xlims=xlimit, ylims=ylimit, size=size)

        x = [grid.xyz_q[cell_id][node_id][1] for node_id in 1:num_of_nodes]
        y = [grid.xyz_q[cell_id][node_id][2] for node_id in 1:num_of_nodes]
        scatter!(p, x, y, markersize=1, label="", markercolor=:red)
    end

    return p
end

function plot_one_variable(grid, W; xlimit = [-0.51, 0.51], ylimit = [-0.01, 0.151], size = (2400, 800), variable = 1)

    if variable == 1
        println("plotting ρ \n")
        colorbar_title = "ρ"
    elseif variable == 2
        println("plotting u \n")
        colorbar_title = "u"
    elseif variable == 3
        println("plotting p \n")
        colorbar_title = "p"
    else
        error("variable not found \n")
    end
    num_of_cells = n_cells(grid)
    num_of_nodes = Int(length(grid.xyz_q[1]))
    p = scatter(xlabel = "x", ylabel = "y")

    u = W[:, :, variable]
    
    # Calculate color range for density
    u_avg = [mean(u[cell_id, :]) for cell_id in 1:num_of_cells]
    u_min, u_max = minimum(u_avg), maximum(u_avg)

    # Plot triangles
    for cell_id in 1:num_of_cells
        vertices_IDS = vertices(grid.cells[cell_id])
        vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
        vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
        vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

        v1x, v1y = vertice1[1], vertice1[2]
        v2x, v2y = vertice2[1], vertice2[2]
        v3x, v3y = vertice3[1], vertice3[2]

        # Calculate triangle center
        center_x = (v1x + v2x + v3x) / 3
        center_y = (v1y + v2y + v3y) / 3

        # Calculate color based on average density
        cell_avg_u = mean(u[cell_id, :])
        normalized_u = (cell_avg_u - u_min)/(u_max - u_min)
        
        # Plot filled triangle
        plot!(p, Shape([v1x, v2x, v3x], [v1y, v2y, v3y]), 
              color=cgrad(:viridis)[normalized_u],
              alpha=0.7,
              label="")

        # Plot edges
        plot!(p, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
              label="", color=:black, linewidth=0.5,
              xlims=xlimit, ylims=ylimit, size=size)

        # Add cell ID at triangle center
        annotate!(p, [(center_x, center_y, text("$cell_id", :black, 8))])
    end

    plot!(p, colorbar=true, colorbar_title=colorbar_title)
    return p
end


