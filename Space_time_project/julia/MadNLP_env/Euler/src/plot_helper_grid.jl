function plot_u_interactive(grid::Grid, us)
    # 返回交互式3D图
    plotlyjs()
    p = scatter3d(markersize=1, markercolor=:blue)

    if typeof(us) != Array{<:AbstractArray}
        us = [us]  # 确保 us 是一个数组
    end


    for u in us
        for cell_id in 1:n_cells(grid)
            x = [x[1] for x in grid.xyz_q[cell_id]]
            t = [x[2] for x in grid.xyz_q[cell_id]]
            u_cell = u[cell_id, :]
            scatter3d!(p, x, t, u_cell, 
                    markersize=1, 
                    markercolor=:blue, 
                    label="")
        end
    end


    xlabel!(p, "x")
    ylabel!(p, "t")
    zlabel!(p, "u")
    title!(p, "Solution u")
    
    return p
end


function plot_u_2D(grid::Grid, u)
    p = plot()
    for cell_id in 1:n_cells(grid)
        x = [x[1] for x in grid.xyz_q[cell_id]]
        t = [x[2] for x in grid.xyz_q[cell_id]]
        u_cell = u[cell_id, :]
        # 画二维颜色图
        scatter!(p, x, t, zcolor=u_cell, markersize=5, label="", color=:viridis)
    end
    p
    
end

function plot_mesh(grid::Grid; xlimit = [-1.1, 1.1], ylimit = [-1.1, 1.1], plot_size = (800, 800), show_cell_id = true)
    # num_of_cells = n_cells(grid)
    # num_of_nodes = Int(length(grid.xyz_q[1]))
    num_of_cells = length(grid.VOL)
    num_of_nodes = size(grid.VOL[1][1], 1)
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

        # 计算三角形中心
        center_x = (v1x + v2x + v3x) / 3
        center_y = (v1y + v2y + v3y) / 3

        plot!(p, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
              label="", color=:blue, xlims=xlimit, ylims=ylimit, size=plot_size)

        # x = [grid.xyz_q[cell_id][node_id][1] for node_id in 1:num_of_nodes]
        # y = [grid.xyz_q[cell_id][node_id][2] for node_id in 1:num_of_nodes]

        x = [grid.xyz_q[cell_id][node_id][1] for node_id in 1:num_of_nodes]
        y = [grid.xyz_q[cell_id][node_id][2] for node_id in 1:num_of_nodes]

        
        scatter!(p, x, y, markersize=1, label="", markercolor=:red)

        if show_cell_id
            annotate!(p, [(center_x, center_y, text("$cell_id", :black, 8))])
        end
    end

    return p
end

function plot_interactive_solution_by_U(grid, U::Array{Float64, 3}; size = (2400, 800), variable = 1)
    """
    using plot. 
    """
    plotly()
    if variable == 1
        println("Plotting ρ")
        vari = U[:, :, 1]
    elseif variable == 2
        println("Plotting u")
        vari = U[:, :, 2] ./ U[:, :, 1]
    else
        println("Plotting p")
        u = U[:, :, 2] ./ U[:, :, 1]
        ρ = U[:, :, 1]
        E = U[:, :, 3]
        # p = (E - 0.5 * ρ * u^2) * (γ - 1)
        vari = (E .- 0.5 * ρ .* (u.^2)) * (γ - 1)
    end

    num_of_cells = n_cells(grid)
    num_of_nodes = length(grid.xyz_q[1])
    # Pre-allocate x and y arrays
    x = Matrix{Float64}(undef, num_of_cells, num_of_nodes)
    y = Matrix{Float64}(undef, num_of_cells, num_of_nodes)

    @inbounds for cell_id in 1:num_of_cells
        for node_id in 1:num_of_nodes
            x[cell_id, node_id] = grid.xyz_q[cell_id][node_id][1]
            y[cell_id, node_id] = grid.xyz_q[cell_id][node_id][2]
        end
    end

    @inbounds p = scatter3d()

    for cell_id in 1:num_of_cells
        # Plot numerical and analytical solutions for this cell
        z1 = vari[cell_id, :]
        scatter3d!(p, x[cell_id, :], y[cell_id, :], z1, markersize=1, label="", color=:blue)
    end

    return p
end

function plot_interactive_solution_by_W(grid, W::Array{Float64, 3}; variable = 1)
    """
    using plot. 
    """
    plotly()
    if variable == 1
        println("Plotting ρ")
        plot_title = "ρ"
        vari = W[:, :, 1]
    elseif variable == 2
        println("Plotting u")
        plot_title = "u"
        vari = W[:, :, 2]
    else
        println("Plotting p")
        plot_title = "p"
        vari = W[:, :, 3]
    end

    num_of_cells = n_cells(grid)
    num_of_nodes = length(grid.xyz_q[1])
    # Pre-allocate x and y arrays
    x = Matrix{Float64}(undef, num_of_cells, num_of_nodes)
    y = Matrix{Float64}(undef, num_of_cells, num_of_nodes)

    @inbounds for cell_id in 1:num_of_cells
        for node_id in 1:num_of_nodes
            x[cell_id, node_id] = grid.xyz_q[cell_id][node_id][1]
            y[cell_id, node_id] = grid.xyz_q[cell_id][node_id][2]
        end
    end

    @inbounds p = scatter3d()

    for cell_id in 1:num_of_cells
        # Plot numerical and analytical solutions for this cell
        z1 = vari[cell_id, :]
        scatter3d!(p, x[cell_id, :], y[cell_id, :], z1, markersize=1, label="", color=:blue)
    end

    xaxis!(p, "x")
    yaxis!(p, "t")
    zaxis!(p, plot_title)

    return p
    
end