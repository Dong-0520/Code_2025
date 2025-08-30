


function plot_u_interactive(grid::SimpleGrid, us)
    # 返回交互式3D图
    plotlyjs()
    p = scatter3d(markersize=1, markercolor=:blue)
    if typeof(us) != Array{<:AbstractArray}
        us = [us]  # 确保 us 是一个数组
    end

    for u in us
        for cell_id in 1:size(grid.D̄t, 1)
            # x = [x[1] for x in grid.xyz_q[cell_id]]
            # t = [x[2] for x in grid.xyz_q[cell_id]]
            x = [grid.xyz_SBP[cell_id][node_id][1] for node_id in 1:size(grid.ref.rst, 2)]
            t = [grid.xyz_SBP[cell_id][node_id][2] for node_id in 1:size(grid.ref.rst, 2)]
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


function plot_mesh(grid::SimpleGrid; xlimit = [-1.1, 1.1], ylimit = [-1.1, 1.1], plot_size = (800, 800), show_cell_id = true)
    # num_of_cells = n_cells(grid)
    # num_of_nodes = Int(length(grid.xyz_q[1]))
    num_of_cells = size(grid.D̄t, 1)
    num_of_nodes = size(grid.D̄t, 2) ÷ 3
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

        x = [grid.xyz_SBP[cell_id][node_id][1] for node_id in 1:num_of_nodes]
        y = [grid.xyz_SBP[cell_id][node_id][2] for node_id in 1:num_of_nodes]

        
        scatter!(p, x, y, markersize=1, label="", markercolor=:red)

        if show_cell_id
            annotate!(p, [(center_x, center_y, text("$cell_id", :black, 8))])
        end
    end

    return p
end


function plot_one_variable_by_W(grid::SimpleGrid, W; xlimit = [-0.51, 0.51], ylimit = [-0.01, 0.151], plot_size = (2400, 800), variable = 1, show_cell_id = true)

    if variable == 1
        println("plotting ρ \n")
        plot_title = "Density"
    elseif variable == 2
        println("plotting u \n")
        plot_title = "Velocity"
    elseif variable == 3
        println("plotting p \n")
        plot_title = "Pressure"
    else
        error("variable not found \n")
    end

    num_of_cells = size(grid.D̄t, 1)

    # 计算颜色范围
    u = W[:, :, variable]
    u_avg = [mean(u[cell_id, :]) for cell_id in 1:num_of_cells]
    u_min, u_max = minimum(u_avg), maximum(u_avg)
    # 创建带有明确边距和 colorbar 设置的图
    p = scatter(xlabel = "x", ylabel = "t",
                right_margin = 10Plots.mm,
                left_margin = 20Plots.mm,
                bottom_margin = 10Plots.mm,
                top_margin = 4Plots.mm,      # 减小顶部边距
                size = plot_size,
                framestyle = :box,
                legend = false,
                title = plot_title,
                titlefontsize = 14,
                titlelocation = :center,     # 居中标题
                titlefont = font("Times", 14)) # 设置标题字体

    # 创建一个虚拟的散点图来设置 colorbar
    scatter!(p, [NaN], [NaN],
            marker_z = [u_min],
            clims = (u_min, u_max),
            colorbar = true,
            colorbar_title = "",
            c = :viridis)

    # ...existing triangle plotting code...

    # 绘制三角形

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

        # 计算颜色值
        cell_avg_u = mean(u[cell_id, :])
        normalized_u = (cell_avg_u - u_min)/(u_max - u_min)
        
        # 绘制填充三角形
        plot!(p, Shape([v1x, v2x, v3x], [v1y, v2y, v3y]), 
              color = cgrad(:viridis)[normalized_u],
              alpha = 0.7,
              label = "")

        # 绘制边界
        plot!(p, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
              color = :black, 
              linewidth = 0.5,
              label = "")

        # 添加单元编号
        if show_cell_id
            annotate!(p, [(center_x, center_y, text("$cell_id", :black, 8))])
        end
    end

    scatter!(p, [NaN], [NaN],
        marker_z = [u_min],
        clims = (u_min, u_max),
        colorbar = true,
        c = :viridis)

    xaxis!(p, "x")
    yaxis!(p, "t")
    zaxis!(p, plot_title)
    return p
end