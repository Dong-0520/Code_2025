#-------------------------------------------------
# plotting Helper
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

function plot_SBP_mesh(grid; size = (1600, 400), show_cell_ids = true)
    num_of_cells = n_cells(grid)
    num_of_nodes = Int(length(grid.xyz_q[1]))
    p = scatter(xlabel = "x", ylabel = "t", 
                size=size, aspect_ratio=:equal)

    for cell_id in 1:num_of_cells
        vertices_IDS = vertices(grid.cells[cell_id])
        vertice1 = grid.xyz_gmsh[vertices_IDS[1]]
        vertice2 = grid.xyz_gmsh[vertices_IDS[2]]
        vertice3 = grid.xyz_gmsh[vertices_IDS[3]]

        v1x, v1y = vertice1[1], vertice1[2]
        v2x, v2y = vertice2[1], vertice2[2]
        v3x, v3y = vertice3[1], vertice3[2]

        center_x = (v1x + v2x + v3x) / 3
        center_y = (v1y + v2y + v3y) / 3


        plot!(p, [v1x, v2x, v3x, v1x], [v1y, v2y, v3y, v1y], 
              label="", color=:blue, xlabel = "x", ylabel = "t")

        x = [grid.xyz_q[cell_id][node_id][1] for node_id in 1:num_of_nodes]
        y = [grid.xyz_q[cell_id][node_id][2] for node_id in 1:num_of_nodes]
        scatter!(p, x, y, markersize=1, label="", markercolor=:red)
        if show_cell_ids
            annotate!(p, [(center_x, center_y, text("$cell_id", :black, 8))])
        end
    end
    return p

end


function plot_SBP_mesh_simpleGrid(grid; size = (1600, 400))
    num_of_cells = length(grid.cells)
    num_of_nodes = Int(length(grid.xyz_SBP[1]))
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

        x = [grid.xyz_SBP[cell_id][node_id][1] for node_id in 1:num_of_nodes]
        y = [grid.xyz_SBP[cell_id][node_id][2] for node_id in 1:num_of_nodes]
        scatter!(p, x, y, markersize=1, label="", markercolor=:red)
        center_x = (v1x + v2x + v3x) / 3
        center_y = (v1y + v2y + v3y) / 3
        annotate!(p, [(center_x, center_y, text("$cell_id", :black, 8))])
    end
    return p
    
end

print(" \n plot helper included \n")