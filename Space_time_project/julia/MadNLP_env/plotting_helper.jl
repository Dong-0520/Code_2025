function plot_u_interactive(grid, us::Vector{Matrix{Float64}})
    # 返回交互式3D图
    plotlyjs()
    p = scatter3d(markersize=1)

    # Define a color palette for different solutions
    colors = [:blue, :red, :green, :purple, :orange, :cyan, :magenta, :yellow]
    
    for (i, u) in enumerate(us)
        # Use modulo to cycle through colors if there are more solutions than colors
        color = colors[mod1(i, length(colors))]
        
        for cell_id in 1:n_cells(grid)
            x = [x[1] for x in grid.xyz_q[cell_id]]
            t = [x[2] for x in grid.xyz_q[cell_id]]
            u_cell = u[cell_id, :]
            scatter3d!(p, x, t, u_cell, 
                    markersize=1, 
                    markercolor=color, 
                    label="")  # Only add label for first cell
        end
    end

    # ...existing code...
    return p
end
function plot_u_interactive(grid, u::Matrix{Float64})
    plot_u_interactive(grid, [u])
end

function plot_u_2D(grid, u)
    p = plot()
    for cell_id in 1:n_cells(grid)
        x = [x[1] for x in grid.xyz_q[cell_id]]
        t = [x[2] for x in grid.xyz_q[cell_id]]
        u_cell = u[cell_id, :]
        # 画二维颜色图
        scatter!(p, x, t, zcolor=u_cell, markersize=5, label="", color=:viridis)

        vertices_IDS = vertices(grid.cells[cell_id])
        vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
        vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
        vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

        v1x, v1y = vertice1[1], vertice1[2]
        v2x, v2y = vertice2[1], vertice2[2]
        v3x, v3y = vertice3[1], vertice3[2]

        plot!(p, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
              label="", color=:blue)
    end
    p
    
end


function plot_SBP_elements(grid, cell_ids::Array{Int, 1})


    num_of_cells = n_cells(grid)
    num_of_nodes = Int(length(grid.xyz_q[1]))
    p = scatter(xlabel = "x", ylabel = "y")

    for cell_id in cell_ids
        vertices_IDS = vertices(grid.cells[cell_id])
        vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
        vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
        vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

        v1x, v1y = vertice1[1], vertice1[2]
        v2x, v2y = vertice2[1], vertice2[2]
        v3x, v3y = vertice3[1], vertice3[2]

        plot!(p, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
              label="", color=:blue)

        x = [grid.xyz_q[cell_id][node_id][1] for node_id in 1:num_of_nodes]
        y = [grid.xyz_q[cell_id][node_id][2] for node_id in 1:num_of_nodes]
        scatter!(p, x, y, markersize=1, label="", markercolor=:red)
    end
    return p
end
function plot_SBP_elements(grid, cell_id::Int)
    return plot_SBP_elements(grid, [cell_id])
end

function plot_SBP_mesh(grid; size = (1600, 400))
    num_of_cells = n_cells(grid)
    num_of_nodes = Int(length(grid.xyz_q[1]))
    p = scatter(xlabel = "x", ylabel = "y", 
                size=size, aspect_ratio=:equal)

    for cell_id in 1:num_of_cells
        vertices_IDS = vertices(grid.cells[cell_id])
        vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
        vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
        vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

        v1x, v1y = vertice1[1], vertice1[2]
        v2x, v2y = vertice2[1], vertice2[2]
        v3x, v3y = vertice3[1], vertice3[2]

        plot!(p, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
              label="", color=:blue)

        x = [grid.xyz_q[cell_id][node_id][1] for node_id in 1:num_of_nodes]
        y = [grid.xyz_q[cell_id][node_id][2] for node_id in 1:num_of_nodes]
        scatter!(p, x, y, markersize=1, label="", markercolor=:red)
    end
    return p

end