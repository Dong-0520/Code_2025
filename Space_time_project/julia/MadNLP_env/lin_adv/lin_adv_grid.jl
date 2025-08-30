using gmsh_jll, Gmsh
include(gmsh_jll.gmsh_api)

# 初始化 Gmsh
gmsh.initialize()

meshNumber = 6

name = "lin_adv_grid_nonalign_$meshNumber"
gmsh.model.add(name)

lc = meshNumber * 0.1
xL = -0.5
xR = 0.5    
yB = 0.0
yT = 0.2
x_star = -0.12
z = 0.0

a = 1.25

# add points
gmsh.model.geo.addPoint(xL, yB, z, lc, 1)
gmsh.model.geo.addPoint(x_star, yB, z, lc, 2)
gmsh.model.geo.addPoint(xR, yB, z, lc, 3)
gmsh.model.geo.addPoint(xR, yT, z, lc, 4)
gmsh.model.geo.addPoint(x_star + a * (yT - yB), yT, z, lc, 5)
gmsh.model.geo.addPoint(xL, yT, z, lc, 6)

# add lines
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 5, 4)
gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(6, 1, 6)
gmsh.model.geo.addLine(2, 5, 7)

gmsh.model.geo.synchronize()


# add curved loop
gmsh.model.geo.addCurveLoop([1, 7, 5, 6], 1)
gmsh.model.geo.addCurveLoop([2, 3, 4, -7], 2)
# add surface
surface_tag1 = gmsh.model.geo.addPlaneSurface([1])
gmsh.model.addPhysicalGroup(2, [surface_tag1], 1)
gmsh.model.setPhysicalName(2, 1, "surface1")
surface_tag2 = gmsh.model.geo.addPlaneSurface([2])
gmsh.model.addPhysicalGroup(2, [surface_tag2], 2)
gmsh.model.setPhysicalName(2, 2, "surface2")

# make a quadraleteral mesh
# gmsh.option.setNumber("Mesh.RecombineAll", 1)


# Add BCs
gmsh.model.addPhysicalGroup(1, [6], 3)
gmsh.model.setPhysicalName(1, 3, "LEFT_INFLOW")

gmsh.model.addPhysicalGroup(1, [1], 4)
gmsh.model.setPhysicalName(1, 4, "BOTTOM_INFLOW_1")

gmsh.model.addPhysicalGroup(1, [2], 5)
gmsh.model.setPhysicalName(1, 5, "BOTTOM_INFLOW_2")

gmsh.model.addPhysicalGroup(1, [5], 6)
gmsh.model.setPhysicalName(1, 6, "TOP_INFLOW_1")

gmsh.model.addPhysicalGroup(1, [4], 7)
gmsh.model.setPhysicalName(1, 7, "TOP_INFLOW_2")

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