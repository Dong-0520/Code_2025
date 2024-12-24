# 2D grid with high res. on L shape using distance fields

using gmsh_jll
include(gmsh_jll.gmsh_api)

gmsh.initialize()

gmsh.model.add("2d_grid_L_shape_2")

lc = 0.1

# Create rectangle
gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc, 1)
gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc, 2)
gmsh.model.geo.addPoint(1.0, 2.0, 0.0, lc, 3)
gmsh.model.geo.addPoint(0.0, 2.0, 0.0, lc, 4)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

# Curve loops
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 6)

# Surfaces as physical groups
gmsh.model.geo.addPlaneSurface([6], 7)
gmsh.model.addPhysicalGroup(1, [7], 1, "surface_1")

# Define distances from boundary 1 and 2
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", [1])
gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
gmsh.model.mesh.field.add("Distance", 2)
gmsh.model.mesh.field.setNumbers(2, "CurvesList", [4])
gmsh.model.mesh.field.setNumber(2, "Sampling", 100)

# Define threshold field
gmsh.model.mesh.field.add("Threshold", 3)
gmsh.model.mesh.field.setNumber(3, "InField", 1)
gmsh.model.mesh.field.setNumber(3, "SizeMin", lc / 10)
gmsh.model.mesh.field.setNumber(3, "SizeMax", lc)
gmsh.model.mesh.field.setNumber(3, "DistMin", 0.05)
gmsh.model.mesh.field.setNumber(3, "DistMax", 0.15)
gmsh.model.mesh.field.add("Threshold", 4)
gmsh.model.mesh.field.setNumber(4, "InField", 2)
gmsh.model.mesh.field.setNumber(4, "SizeMin", lc / 10)
gmsh.model.mesh.field.setNumber(4, "SizeMax", lc)
gmsh.model.mesh.field.setNumber(4, "DistMin", 0.05)
gmsh.model.mesh.field.setNumber(4, "DistMax", 0.15)

gmsh.model.mesh.field.add("Min", 5)
gmsh.model.mesh.field.setNumbers(5, "FieldsList", [3, 4])

gmsh.model.mesh.field.setAsBackgroundMesh(5)

gmsh.model.geo.synchronize()
gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.model.mesh.generate(2)
gmsh.write("2d_grid_L_shape_2.msh")


if !("-nopopup" in ARGS)
    gmsh.fltk.run()
end

gmsh.finalize()
