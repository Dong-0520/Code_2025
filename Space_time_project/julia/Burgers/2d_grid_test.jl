using gmsh_jll
include(gmsh_jll.gmsh_api)

# 初始化 Gmsh
gmsh.initialize()

meshNumber = 5

name = "2d_grid_test1"
gmsh.model.add(name)

lc = meshNumber * 0.01
xL = 0.0
xR = 0.5
yB = 0.0
yT = (1 / (4 * pi))
x_star = 0.25
a = 0.0

# add points
gmsh.model.geo.addPoint(xL, yB, 0.0, lc, 1)
gmsh.model.geo.addPoint(x_star, yB, 0.0, lc, 2)
gmsh.model.geo.addPoint(x_star + a * yT, yT, 0.0, lc, 3)
gmsh.model.geo.addPoint(xL, yT, 0.0, lc, 4)
gmsh.model.geo.addPoint(xR, yB, 0.0, lc, 5)
gmsh.model.geo.addPoint(xR, yT, 0.0, lc, 6)


# add lines
# gmsh.model.geo.addLine(1, 2, 1)
# gmsh.model.geo.addLine(2, 3, 2)
# gmsh.model.geo.addLine(3, 4, 3)
# gmsh.model.geo.addLine(4, 1, 4)

# gmsh.model.geo.addLine(5, 2, 5)
# gmsh.model.geo.addLine(6, 5, 6)
# gmsh.model.geo.addLine(3, 6, 7)
gmsh.model.geo.addLine(1, 5, 1)
gmsh.model.geo.addLine(5, 6, 2)
gmsh.model.geo.addLine(6, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)
gmsh.model.geo.addLine(2, 3, 5) # shock

gmsh.model.geo.synchronize()



# set setPeriodic
# dont forget to check line tag to make sure they are 
# periodic to the correct line
dx = abs(xR - xL)
dy = abs(yT - yB)
to_right = [1.0, 0.0, 0.0, dx, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
to_up = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, dy, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

gmsh.model.mesh.setPeriodic(1, [2], [4], to_right)
gmsh.model.mesh.setPeriodic(1, [3], [1], to_up)
gmsh.model.geo.synchronize()

# add curved loop
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)

# add surface
surface_tag1 = gmsh.model.geo.addPlaneSurface([1])
gmsh.model.addPhysicalGroup(2, [surface_tag1], 1, "surface1")

# make a quadraleteral mesh
gmsh.option.setNumber("Mesh.RecombineAll", 1)


# Add BCs
gmsh.model.addPhysicalGroup(1, [4], 2, "LEFT_INFLOW")
gmsh.model.addPhysicalGroup(1, [2], 3, "RIGHT_INFLOW")
gmsh.model.addPhysicalGroup(1, [1], 4, "BOTTOM_INFLOW_1")
gmsh.model.addPhysicalGroup(1, [3], 5, "TOP_INFLOW_1")
gmsh.model.addPhysicalGroup(1, [5], 6, "SHOCK")
gmsh.model.geo.synchronize()



# set mesh algorithm
gmsh.option.setNumber("Mesh.MshFileVersion", 4.0)
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3) # 2 or 3
# gmsh.option.setNumber("Mesh.Algorithm", 5)
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