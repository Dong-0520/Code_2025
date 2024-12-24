using gmsh_jll
include(gmsh_jll.gmsh_api)

# 初始化 Gmsh
gmsh.initialize()

meshNumber = 5

name = "BurgersMeshSlab_tB0005"
gmsh.model.add(name)

lc = meshNumber * 0.01
xL = 0.0
xR = 0.5
yB = 0.0
yT = (1 / (4 * pi)) + 0.005
x_star = 0.25
a = 0.0

# add points
gmsh.model.geo.addPoint(xL, yB, 0.0, lc, 1)
gmsh.model.geo.addPoint(x_star, yB, 0.0, lc, 2)
gmsh.model.geo.addPoint(x_star + a * yT, yT, 0.0, lc, 3)
gmsh.model.geo.addPoint(xL, yT, 0.0, lc, 4)
gmsh.model.geo.addPoint(xR, yB, 0.0, lc, 5)
gmsh.model.geo.addPoint(xR, yT, 0.0, lc, 6)

# for i in 1:9
#     gmsh.model.geo.addPoint(xL + i * (xR - xL) / 10, yB, 0.0, lc, 6 + i)
# end
# for i in 1:9
#     gmsh.model.geo.addPoint(xL + i * (xR - xL) / 10, yT, 0.0, lc, 15 + i)
# end

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

# set setPeriodic
# dont forget to check line tag to make sure they are 
# periodic to the correct line
dx = abs(xR - xL)
dy = abs(yT - yB)
to_right = [1.0, 0.0, 0.0, dx, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
to_up = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, dy, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

gmsh.model.mesh.setPeriodic(1, [6], [4], to_right)
gmsh.model.mesh.setPeriodic(1, [3], [1], to_up)
gmsh.model.mesh.setPeriodic(1, [7], [5], to_up)
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
gmsh.option.setNumber("Mesh.ElementOrder", 2)
gmsh.option.setNumber("Mesh.Algorithm", 5)
# gmsh.model.mesh.setRecombine(2, surface_tag1)
# gmsh.model.mesh.setRecombine(2, surface_tag2)
gmsh.model.mesh.generate(2)
gmsh.model.geo.synchronize()

# save mesh
output_path = joinpath(@__DIR__, "mesh/$name.msh")
gmsh.write(output_path)

# run gmsh gui
if !("-nopopup" in ARGS)
    gmsh.fltk.run()
end

# finalize
gmsh.finalize()