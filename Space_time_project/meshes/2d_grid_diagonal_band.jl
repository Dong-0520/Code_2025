# 2D grid with 1 surface with higher resolution along the diagonal band

using gmsh_jll
include(gmsh_jll.gmsh_api)

gmsh.initialize()

gmsh.model.add("2d_grid_diagonal_band")

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

# Create diagonal line
gmsh.model.geo.addLine(2, 4, 5)

# Curve loops
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)

# Surfaces as physical groups
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.addPhysicalGroup(2, [1], 1, "surface_1")

gmsh.model.geo.synchronize()

# Distance to diagonal line
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", [5])
gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

# Threshold field (if distance to diagonal line is less than 0.1, use lc/10)
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", lc / 10)
gmsh.model.mesh.field.setNumber(2, "SizeMax", lc)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.05)
gmsh.model.mesh.field.setNumber(2, "DistMax", 0.15)

gmsh.model.mesh.field.setAsBackgroundMesh(2)


gmsh.model.geo.synchronize()

# Add BCs
gmsh.model.addPhysicalGroup(1, [1], 1, "INFLOW")
gmsh.model.addPhysicalGroup(1, [4], 2, "OUTFLOW")

gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.model.mesh.generate(2)
gmsh.write("2d_grid_diagonal_band.msh")


if !("-nopopup" in ARGS)
    gmsh.fltk.run()
end

gmsh.finalize()
