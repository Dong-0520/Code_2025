# 2D grid with high res. on L shape using boxes

using gmsh_jll
include(gmsh_jll.gmsh_api)

gmsh.initialize()

gmsh.model.add("2d_grid_L_shape_1")

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
gmsh.model.addPhysicalGroup(2, [7], 1, "surface_1")

# Define boxes for L shape
gmsh.model.mesh.field.add("Box", 1)
gmsh.model.mesh.field.setNumber(1, "VIn", lc / 4)
gmsh.model.mesh.field.setNumber(1, "VOut", lc)
gmsh.model.mesh.field.setNumber(1, "XMin", 0.0)
gmsh.model.mesh.field.setNumber(1, "XMax", 0.2)
gmsh.model.mesh.field.setNumber(1, "YMin", 0.0)
gmsh.model.mesh.field.setNumber(1, "YMax", 2.0)
gmsh.model.mesh.field.setNumber(1, "Thickness", 0.2)

# Define boxes for L shape
gmsh.model.mesh.field.add("Box", 2)
gmsh.model.mesh.field.setNumber(2, "VIn", lc / 4)
gmsh.model.mesh.field.setNumber(2, "VOut", lc)
gmsh.model.mesh.field.setNumber(2, "XMin", 0.0)
gmsh.model.mesh.field.setNumber(2, "XMax", 1.0)
gmsh.model.mesh.field.setNumber(2, "YMin", 0.0)
gmsh.model.mesh.field.setNumber(2, "YMax", 0.4)
gmsh.model.mesh.field.setNumber(2, "Thickness", 0.2)

gmsh.model.mesh.field.add("Min", 3)
gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])

gmsh.model.mesh.field.setAsBackgroundMesh(3)

gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(1, [1], 1, "INFLOW_1")
gmsh.model.addPhysicalGroup(1, [4], 2, "INFLOW_2")

gmsh.option.setNumber("Mesh.ElementOrder", 3)
gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.model.mesh.generate(2)
gmsh.write("2d_grid_L_shape_1.msh")


if !("-nopopup" in ARGS)
    gmsh.fltk.run()
end

gmsh.finalize()
