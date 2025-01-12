using gmsh_jll
include(gmsh_jll.gmsh_api)

# 初始化 Gmsh
print(" This is test mesh file ")
gmsh.initialize()

meshNumber = 7.5
alg_number = 6

name = "BurgersMeshSlab_005_SimpleTriangle"
gmsh.model.add(name)

lc = meshNumber * 0.01
xL = 0.0
xR = 0.5
yB = 0.0
yT = 0.05

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


# #----------------- Set mesh size controls -----------------#
# # 对于shock附近把网格变密集
# # 禁用默认的网格尺寸控制
# gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
# gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
# gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

# # 创建距离场，用于计算到激波线的距离
# distance_field = gmsh.model.mesh.field.add("Distance")
# gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", [2])  # 2是激波线的tag
# gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 200)     # 增加采样点数量
# gmsh.model.mesh.field.setNumbers(distance_field, "NodesList", [2, 3]) # 添加激波线的端点

# # 创建阈值场来控制网格尺寸的过渡
# threshold_field = gmsh.model.mesh.field.add("Threshold")
# gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
# gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", lc/4)     # 激波附近的网格尺寸
# gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", lc)       # 远离激波的网格尺寸
# gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.01)   # 开始过渡的距离
# gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 0.05)   # 结束过渡的距离

# # 将阈值场设置为背景场
# gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

# # 设置全局尺寸限制
# gmsh.option.setNumber("Mesh.MeshSizeMin", lc/4)
# gmsh.option.setNumber("Mesh.MeshSizeMax", lc)



# set mesh algorithm
gmsh.option.setNumber("Mesh.MshFileVersion", 4.0)
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.Algorithm", alg_number)
# gmsh.model.mesh.setRecombine(2, surface_tag1)
# gmsh.model.mesh.setRecombine(2, surface_tag2)
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