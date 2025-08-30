using gmsh_jll
include(gmsh_jll.gmsh_api)

# 初始化 Gmsh
gmsh.initialize()

meshNumber = 4
alg_number = 5

name = "mesh_example"
gmsh.model.add(name)

lc = meshNumber * 0.01
xL = 0
xR = 1.0
yB = 0.0
yT = 0.1

x_star = 0.2
a = 1.0

# add points
gmsh.model.geo.addPoint(xL, yB, 0.0, lc, 1)
gmsh.model.geo.addPoint(x_star, yB, 0.0, lc, 2)
gmsh.model.geo.addPoint(x_star + a * yT, yT, 0.0, lc, 3)
gmsh.model.geo.addPoint(xL, yT, 0.0, lc, 4)
gmsh.model.geo.addPoint(xR, yB, 0.0, lc, 5)
gmsh.model.geo.addPoint(xR, yT, 0.0, lc, 6)


# add lines
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

gmsh.model.geo.addLine(5, 2, 5)
gmsh.model.geo.addLine(6, 5, 6)
gmsh.model.geo.addLine(3, 6, 7)


# for i in 1:9
#     gmsh.model.geo.addLine(6 + i, 15 + i, 7 + i)
# end

gmsh.model.geo.synchronize()

# add curved loop
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addCurveLoop([2, 7, 6, 5], 2)

# add surface
surface_tag1 = gmsh.model.geo.addPlaneSurface([1])
gmsh.model.addPhysicalGroup(2, [surface_tag1], 1, "surface1")
surface_tag2 = gmsh.model.geo.addPlaneSurface([2])
gmsh.model.addPhysicalGroup(2, [surface_tag2], 2, "surface2")
# Add BCs
gmsh.model.addPhysicalGroup(1, [4], 3, "LEFT_INFLOW")
gmsh.model.addPhysicalGroup(1, [6], 4, "RIGHT_INFLOW")
gmsh.model.addPhysicalGroup(1, [1], 5, "BOTTOM_INFLOW_1")
gmsh.model.addPhysicalGroup(1, [5], 6, "BOTTOM_INFLOW_2")
gmsh.model.addPhysicalGroup(1, [3], 7, "TOP_INFLOW_1")
gmsh.model.addPhysicalGroup(1, [7], 8, "TOP_INFLOW_2")
gmsh.model.addPhysicalGroup(1, [2], 9, "SHOCK")
gmsh.model.geo.synchronize()




# set mesh algorithm
gmsh.option.setNumber("Mesh.MshFileVersion", 4.0)
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.Algorithm", alg_number)
# gmsh.model.mesh.setRecombine(2, surface_tag1)
# gmsh.model.mesh.setRecombine(2, surface_tag2)
gmsh.model.mesh.generate(2)
gmsh.model.geo.synchronize()

# save mesh
output_path = joinpath(@__DIR__, "$name.msh")
gmsh.write(output_path)

# run gmsh gui
if !("-nopopup" in ARGS)
    gmsh.fltk.run()
end

# finalize
gmsh.finalize()