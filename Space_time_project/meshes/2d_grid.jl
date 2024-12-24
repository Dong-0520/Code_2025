using gmsh_jll
include(gmsh_jll.gmsh_api)

gmsh.initialize()

gmsh.model.add("2d_grid")

lc = 0.1

# Create rectangle
gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc, 1)
gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc, 2)
gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc, 3)
gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc, 4)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

# Curve loops
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)

# Surfaces as physical groups
ps = gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.addPhysicalGroup(2, [1], 1, "surface_1")

gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.model.geo.synchronize()

# Add BCs
gmsh.model.addPhysicalGroup(1, [1], 1, "INFLOW_1")
gmsh.model.addPhysicalGroup(1, [4], 2, "INFLOW_2")
gmsh.model.addPhysicalGroup(1, [2], 3, "OUTFLOW_1")
gmsh.model.addPhysicalGroup(1, [3], 4, "OUTFLOW_2")

gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.ElementOrder", 3)

gmsh.model.mesh.generate(2)
gmsh.write("2d_grid.msh")

if !("-nopopup" in ARGS)
    gmsh.fltk.run()
end

gmsh.finalize()
