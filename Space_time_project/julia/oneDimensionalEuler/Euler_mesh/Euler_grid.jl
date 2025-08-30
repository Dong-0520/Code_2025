using gmsh_jll
include(gmsh_jll.gmsh_api)

# 初始化 Gmsh
gmsh.initialize()

meshNumber = 12

name = "Euler_grid_test2_$(meshNumber)"
gmsh.model.add(name)

lc = meshNumber * 0.01
xL = -0.5
xR = 0.5
yB = 0.0
yT = 0.15
x_star = 0.0
z = 0.0
γ = 1.4

ρL = 3.0
uL = 2.0
pL = 10.0
ρR = 3.0
uR = -1.0
pR = 5.0

# ρstarL = 5.013935153894693
# ρstarR = 7.67671462561582
# ustar = 0.7943931050164833
# pstar = 20.85590303477569

ρstarL = 5.013935083865458
ρstarR = 7.676714537106275
ustar = 0.7943931077390836
pstar = 20.855902598748525

function one_wave_speed(ρL, uL, pL, pstar)
    aL = sqrt(γ * pL / ρL)

    ratio1 = (γ + 1)/(2*γ)
    p_ratio = pstar / pL
    ratio2 = (γ - 1)/(2*γ)

    return uL - aL * sqrt( ratio1 * p_ratio + ratio2)
end

function three_wave_speed(ρR, uR, pR, pstar)
    aR = sqrt(γ * pR / ρR)

    ratio1 = (γ + 1)/(2*γ)
    p_ratio = pstar / pR
    ratio2 = (γ - 1)/(2*γ)

    return uR + aR * sqrt( ratio1 * p_ratio + ratio2)
end

S1 = one_wave_speed(ρL, uL, pL, pstar)
S2 = ustar
S3 = three_wave_speed(ρR, uR, pR, pstar)

# add points
gmsh.model.geo.addPoint(xL, yB, z, lc, 1)
gmsh.model.geo.addPoint(x_star, yB, z, lc, 2)
gmsh.model.geo.addPoint(xR, yB, z, lc, 3)
gmsh.model.geo.addPoint(xR, yT, z, lc, 4)

gmsh.model.geo.addPoint(x_star + S3 * yT , yT, z, lc, 5)
gmsh.model.geo.addPoint(x_star + S2 * yT, yT, z, lc, 6)
gmsh.model.geo.addPoint(x_star + S1 * yT, yT, z, lc, 7)

gmsh.model.geo.addPoint(xL, yT, z, lc, 8)

# add lines
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 7, 2)
gmsh.model.geo.addLine(7, 8, 3)
gmsh.model.geo.addLine(8, 1, 4)

gmsh.model.geo.addLine(3, 2, 5)
gmsh.model.geo.addLine(2, 5, 6)
gmsh.model.geo.addLine(5, 4, 7)
gmsh.model.geo.addLine(4, 3, 8)

gmsh.model.geo.addLine(5, 6, 9)
gmsh.model.geo.addLine(6, 2, 10)
gmsh.model.geo.addLine(7, 6, 11)

gmsh.model.geo.synchronize()


# add curved loop
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)
gmsh.model.geo.addCurveLoop([6, 9, 10], 3)
gmsh.model.geo.addCurveLoop([10, 2, 11], 4)

# add surface
surface_tag1 = gmsh.model.geo.addPlaneSurface([1])
gmsh.model.addPhysicalGroup(2, [surface_tag1], 1, "surface1")
surface_tag2 = gmsh.model.geo.addPlaneSurface([2])
gmsh.model.addPhysicalGroup(2, [surface_tag2], 2, "surface2")
surface_tag3 = gmsh.model.geo.addPlaneSurface([3])
gmsh.model.addPhysicalGroup(2, [surface_tag3], 3, "surface3")
surface_tag4 = gmsh.model.geo.addPlaneSurface([4])
gmsh.model.addPhysicalGroup(2, [surface_tag4], 4, "surface4")

# make a quadraleteral mesh
# gmsh.option.setNumber("Mesh.RecombineAll", 1)


# Add BCs
gmsh.model.addPhysicalGroup(1, [4], 5, "LEFT_INFLOW")
gmsh.model.addPhysicalGroup(1, [8], 6, "RIGHT_INFLOW")
gmsh.model.addPhysicalGroup(1, [1], 7, "BOTTOM_INFLOW_1")
gmsh.model.addPhysicalGroup(1, [5], 8, "BOTTOM_INFLOW_2")
gmsh.model.addPhysicalGroup(1, [2], 9, "ONE_WAVE")
gmsh.model.addPhysicalGroup(1, [10], 10, "CONTACT_WAVE")
gmsh.model.addPhysicalGroup(1, [6], 11, "THREE_WAVE")
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