
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

# function plot_one_variable_by_W(grid, W; xlimit = [-0.51, 0.51], ylimit = [-0.01, 0.151], size = (2400, 800), variable = 1, show_cell_id = true)
#     if !isnothing(get(ENV, "GKSwstype", nothing))
#         ENV["GKSwstype"] = "100"
#     end
#     if variable == 1
#         println("plotting ρ \n")
#         colorbar_title = "ρ"
#     elseif variable == 2
#         println("plotting u \n")
#         colorbar_title = "u"
#     elseif variable == 3
#         println("plotting p \n")
#         colorbar_title = "p"
#     else
#         error("variable not found \n")
#     end
#     num_of_cells = n_cells(grid)
#     num_of_nodes = Int(length(grid.xyz_q[1]))
#     # p = scatter(xlabel = "x", ylabel = "t")
#     p = scatter(xlabel = "x", ylabel = "t",
#             right_margin = 20Plots.mm,  # 添加右边距
#             size = size,
#             framestyle = :box)

#     u = W[:, :, variable]
    
#     # Calculate color range for density
#     u_avg = [mean(u[cell_id, :]) for cell_id in 1:num_of_cells]
#     u_min, u_max = minimum(u_avg), maximum(u_avg)

#     # Plot triangles
#     for cell_id in 1:num_of_cells
#         vertices_IDS = vertices(grid.cells[cell_id])
#         vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
#         vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
#         vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

#         v1x, v1y = vertice1[1], vertice1[2]
#         v2x, v2y = vertice2[1], vertice2[2]
#         v3x, v3y = vertice3[1], vertice3[2]

#         # Calculate triangle center
#         center_x = (v1x + v2x + v3x) / 3
#         center_y = (v1y + v2y + v3y) / 3

#         # Calculate color based on average density
#         cell_avg_u = mean(u[cell_id, :])
#         normalized_u = (cell_avg_u - u_min)/(u_max - u_min)
        
#         # Plot filled triangle
#         plot!(p, Shape([v1x, v2x, v3x], [v1y, v2y, v3y]), 
#               color=cgrad(:viridis)[normalized_u],
#               alpha=0.7,
#               label="")

#         # Plot edges
#         plot!(p, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
#               label="", color=:black, linewidth=0.5,
#               xlims=xlimit, ylims=ylimit, size=size)

#         # Add cell ID at triangle center
#         if show_cell_id
#             annotate!(p, [(center_x, center_y, text("$cell_id", :black, 8))])
#         end
        
#     end

#     # plot!(p, colorbar=true, colorbar_title=colorbar_title)
#     plot!(p, 
#           colorbar = true,
#           colorbar_title = colorbar_title,
#           colorbar_titlefontsize = 12,
#           colorbar_tickfontsize = 10,
#           clims = (u_min, u_max))  # 设置colorbar的范围
#     return p
# end

function plot_one_variable_by_W(grid, W; xlimit = [-0.51, 0.51], ylimit = [-0.01, 0.151], size = (2400, 800), variable = 1, show_cell_id = true)

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

    # 计算颜色范围
    u = W[:, :, variable]
    u_avg = [mean(u[cell_id, :]) for cell_id in 1:n_cells(grid)]
    u_min, u_max = minimum(u_avg), maximum(u_avg)

    # 创建带有明确边距和 colorbar 设置的图
    p = scatter(xlabel = "x", ylabel = "t",
                right_margin = 30Plots.mm,
                bottom_margin = 10Plots.mm,
                top_margin = 4Plots.mm,      # 减小顶部边距
                size = size,
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
    for cell_id in 1:n_cells(grid)
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



function plot_one_variable_by_U(grid, U; xlimit = [-0.51, 0.51], ylimit = [-0.01, 0.151], size = (2400, 800), variable = 1)

    if variable == 1
        println("plotting ρ \n")
        colorbar_title = "ρ"
        u = U[:, :, variable]
    elseif variable == 2
        println("plotting u \n")
        colorbar_title = "u"
        u = U[:, :, variable] ./ U[:, :, 1]
    elseif variable == 3
        println("plotting p \n")
        colorbar_title = "p"
        u = (U[:, :, 3] .- 0.5 * U[:, :, 1] .* (U[:, :, 2].^2)) * (γ - 1)
    else
        error("variable not found \n")
    end
    num_of_cells = n_cells(grid)
    num_of_nodes = Int(length(grid.xyz_q[1]))
    p = scatter(xlabel = "x", ylabel = "y")


    
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


function observe_variable_iteration(grid, all_U, zlimit, jump; show_true_sol = true, variable = 1)
    """
    Draw a gif with x, y coordinates from grid, taking `all_U[i,:,:,1]` for plotting `ρ`.
    """
    num_of_cells = n_cells(grid)
    num_of_nodes = length(grid.xyz_q[1])
    num_of_pseudo_time_step = size(all_U)[1]
    # Pre-allocate x and y arrays
    x = Matrix{Float64}(undef, num_of_cells, num_of_nodes)
    y = Matrix{Float64}(undef, num_of_cells, num_of_nodes)
    all_vari = zeros(Float64,length(all_U[1:jump:end]), num_of_cells, num_of_nodes)
    if variable == 1
        println("Plotting ρ")
    elseif variable == 2
        println("Plotting u")
    elseif variable == 3
        println("Plotting p")
    end

    @inbounds for cell_id in 1:num_of_cells
        for node_id in 1:num_of_nodes
            x[cell_id, node_id] = grid.xyz_q[cell_id][node_id][1]
            y[cell_id, node_id] = grid.xyz_q[cell_id][node_id][2]
        end
    end

    @inbounds for i in 1:size(all_vari, 1)
        if variable == 1
            all_vari[i, :, :] = all_U[(i-1)*jump+1][:, :, variable]
        elseif variable == 2
            all_vari[i, :, :] = all_U[(i-1)*jump+1][:, :, variable] ./ all_U[(i-1)*jump+1][:, :, 1]
        else
            # p = (E - 0.5 * ρ * u^2) * (γ - 1)
            all_u = all_U[(i-1)*jump+1][:, :, 2] ./ all_U[(i-1)*jump+1][:, :, 1]
            all_vari[i, :, :] = (all_U[(i-1)*jump+1][:, :, 3] .- 0.5 * all_U[(i-1)*jump+1][:, :, 1] .* (all_u.^2)) * (γ - 1)
        end
    end

    @inbounds anim = @animate for i in ProgressBar(1:size(all_vari, 1))
        p = scatter3d()

        for cell_id in 1:num_of_cells
            # Plot numerical and analytical solutions for this cell
            z1 = all_vari[i, cell_id, :]
            scatter3d!(p, x[cell_id, :], y[cell_id, :], z1, markersize=1, label="", zlims=zlimit, color=:blue)
        end

        title!("Pseudo time step: $i")
    end

    return anim
end

function find_final_time(grid)
    final_t = 0.0
    for cell_id in 1:n_cells(grid)
        curr_t = [x[2] for x in grid.xyz_q[cell_id]]
        final_t = max(final_t, maximum(curr_t))
    end
    return final_t
    
end

function plot_solution_at_final_time_by_U(grid, U; variable = 1, threshold=1.0)

    final_t = find_final_time(grid)
    
    if variable == 1
        print("Plotting density at final time...\n")
        z = U[:, :, 1]
        label = "ρ"
    elseif variable == 2
        print("Plotting velocity at final time...\n")
        z = U[:, :, 2] ./ U[:, :, 1]
        label = "u"
    elseif variable == 3
        print("Plotting pressure at final time...\n")
        z = (U[:, :, 3] .- 0.5 .* U[:, :, 1] .* (U[:, :, 2] ./ U[:, :, 1]).^2) * 0.4
        label = "p"
    end

    plot_x = Array{Float64, 1}()
    plot_u = Array{Float64, 1}()

    for cell_id in 1:n_cells(grid)
        ref = grid.cells[cell_id].ref_data[]
        curr_x = [x[1] for x in grid.xyz_q[cell_id]]
        curr_t = [x[2] for x in grid.xyz_q[cell_id]]
        for lfid in 1:length(ref.R)
            Rγ = Matrix(ref.R[lfid])
            x_on_face = Rγ * curr_x
            t_on_face = Rγ * curr_t
            if all(x -> isapprox(x, final_t, atol=1e-8, rtol=1e-8), t_on_face)
                append!(plot_x, x_on_face)
                append!(plot_u, Rγ * z[cell_id, :])
            end
        end
    end


    # Sort data points by x-coordinate
    sorted_indices = sortperm(plot_x)
    plot_x = plot_x[sorted_indices]
    plot_u = plot_u[sorted_indices]

    # 创建基础图
    p = scatter(plot_x, plot_u, 
               markersize=3, 
               markercolor=:blue,
               markershape=:star5,
               label=nothing, 
               title="t=$final_t",
               xlabel="x",
               ylabel=label)
    
    # 添加智能连线
    i = 1
    while i < length(plot_u)
        # 找到连续段的结束位置
        j = i
        while j < length(plot_u) && abs(plot_u[j+1] - plot_u[j]) <= threshold
            j += 1
        end
        
        # 如果有至少2个点，就连线
        if j > i
            plot!(p, plot_x[i:j], plot_u[i:j], 
                  color=:blue, linewidth=2, label=nothing)
        end
        
        i = j + 1
    end
    
    return p
end