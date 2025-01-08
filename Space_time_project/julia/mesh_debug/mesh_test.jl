using gmsh_jll
include(gmsh_jll.gmsh_api)

# 初始化 Gmsh
gmsh.initialize()

meshNumber = 10
alg_number = 2

name = "mesh_test1"
gmsh.model.add(name)

lc = 0.5
xL = 0.0
xR = 0.5
yB = 0.0
yT = 0.1

x_star = 0.25
a = 0.0

# add points
for i in 0:10
    xi = round(xL + i * (xR - xL) / 10, digits=2)
    gmsh.model.geo.addPoint(xi, yB, 0.0, lc, 1 + i)
end
for i in 0:10
    xi = round(xL + i * (xR - xL) / 10, digits=2)
    gmsh.model.geo.addPoint(xi, yT, 0.0, lc, 12 + i)
end


# add lines
for i in 2:10
    gmsh.model.geo.addLine(i, 11 + i, i)
end
gmsh.model.geo.addLine(12, 1, 1)
gmsh.model.geo.addLine(22, 11, 11)

gmsh.model.geo.addLine(1, 6, 12)
gmsh.model.geo.addLine(11, 6, 13)
gmsh.model.geo.addLine(17, 12, 14)
gmsh.model.geo.addLine(17, 22, 15)
# 也可以采取把line tag 标负号以反转方向，类似
# gmsh.model.geo.addCurveLoop([1, 12, -6, 14], 1)
# 之后再尝试吧

gmsh.model.geo.synchronize()

# set setPeriodic
# dont forget to check line tag to make sure they are 
# periodic to the correct line
dx = abs(xR - xL)
dy = abs(yT - yB)
to_right = [1.0, 0.0, 0.0, dx, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
to_up = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, dy, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

gmsh.model.mesh.setPeriodic(1, [11], [1], to_right)
gmsh.model.mesh.setPeriodic(1, [14], [12], to_up)
gmsh.model.mesh.setPeriodic(1, [15], [13], to_up)
gmsh.model.geo.synchronize()

# add curved loop
gmsh.model.geo.addCurveLoop([1, 12, 6, 14], 1)
gmsh.model.geo.addCurveLoop([13, 11, 15, 6], 2)

# add surface
surface_tag1 = gmsh.model.geo.addPlaneSurface([1])
gmsh.model.addPhysicalGroup(2, [surface_tag1], 1, "surface1")
surface_tag2 = gmsh.model.geo.addPlaneSurface([2])
gmsh.model.addPhysicalGroup(2, [surface_tag2], 2, "surface2")
# Add BCs
gmsh.model.addPhysicalGroup(1, [1], 3, "LEFT_INFLOW")
gmsh.model.addPhysicalGroup(1, [11], 4, "RIGHT_INFLOW")
gmsh.model.addPhysicalGroup(1, [12], 5, "BOTTOM_INFLOW_1")
gmsh.model.addPhysicalGroup(1, [13], 6, "BOTTOM_INFLOW_2")
gmsh.model.addPhysicalGroup(1, [14], 7, "TOP_INFLOW_1")
gmsh.model.addPhysicalGroup(1, [15], 8, "TOP_INFLOW_2")
gmsh.model.addPhysicalGroup(1, [6], 9, "SHOCK")
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



# run gmsh gui
if !("-nopopup" in ARGS)
    gmsh.fltk.run()
end

# save mesh
output_path = joinpath(@__DIR__, "$name.msh")
gmsh.write(output_path)

# finalize
gmsh.finalize()