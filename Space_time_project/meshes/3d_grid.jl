using gmsh_jll
include(gmsh_jll.gmsh_api)

gmsh.initialize()

gmsh.model.add("3d_grid")

lc = 0.1

# Create simple cuboid
p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc, 1)
p2 = gmsh.model.geo.addPoint(2.0, 0.0, 0.0, lc, 2)
p3 = gmsh.model.geo.addPoint(2.0, 1.0, 0.0, lc, 3)
p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc, 4)
p5 = gmsh.model.geo.addPoint(0.0, 0.0, 1.0, lc, 5)
p6 = gmsh.model.geo.addPoint(2.0, 0.0, 1.0, lc, 6)
p7 = gmsh.model.geo.addPoint(2.0, 1.0, 1.0, lc, 7)
p8 = gmsh.model.geo.addPoint(0.0, 1.0, 1.0, lc, 8)

l1 = gmsh.model.geo.addLine(p1, p2, 1)
l2 = gmsh.model.geo.addLine(p2, p3, 2)
l3 = gmsh.model.geo.addLine(p3, p4, 3)
l4 = gmsh.model.geo.addLine(p4, p1, 4)
l5 = gmsh.model.geo.addLine(p5, p6, 5)
l6 = gmsh.model.geo.addLine(p6, p7, 6)
l7 = gmsh.model.geo.addLine(p7, p8, 7)
l8 = gmsh.model.geo.addLine(p8, p5, 8)
l9 = gmsh.model.geo.addLine(p1, p5, 9)
l10 = gmsh.model.geo.addLine(p2, p6, 10)
l11 = gmsh.model.geo.addLine(p4, p8, 11)
l12 = gmsh.model.geo.addLine(p3, p7, 12)

# Curve loops
c1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4], 1)
c2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8], 2)
c3 = gmsh.model.geo.addCurveLoop([l9, -l8, -l11, l4], 3)
c4 = gmsh.model.geo.addCurveLoop([l10, l6, -l12, -l2], 4)
c5 = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9], 5)
c6 = gmsh.model.geo.addCurveLoop([-l3, l12, l7, -l11], 6)

# Surfaces
s1 = gmsh.model.geo.addPlaneSurface([c1], 1)
s2 = gmsh.model.geo.addPlaneSurface([c2], 2)
s3 = gmsh.model.geo.addPlaneSurface([c3], 3)
s4 = gmsh.model.geo.addPlaneSurface([c4], 4)
s5 = gmsh.model.geo.addPlaneSurface([c5], 5)
s6 = gmsh.model.geo.addPlaneSurface([c6], 6)

# Volume
sl = gmsh.model.geo.addSurfaceLoop([1, 6, 2, 5, 4, 3], 1)
v = gmsh.model.geo.addVolume([1])

gmsh.model.geo.synchronize()

# Add BCs
gmsh.model.addPhysicalGroup(2, [s1], 1, "INFLOW_1")
gmsh.model.addPhysicalGroup(2, [s3], 2, "INFLOW_2")
gmsh.model.addPhysicalGroup(2, [s5], 3, "INFLOW_3")

gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.ElementOrder", 3)

gmsh.model.mesh.generate(3)
gmsh.write("3d_grid.msh")

if !("-nopopup" in ARGS)
    gmsh.fltk.run()
end

gmsh.finalize()
