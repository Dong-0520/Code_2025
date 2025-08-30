using gmsh_jll
include(gmsh_jll.gmsh_api)

gmsh.initialize()

alg_number = 6

name = "Burgers_sin_coarse_mesh"
gmsh.model.add(name)

# 🔥 大幅增加特征长度来减少单元数量
lc = 0.24  # 更大的特征长度

xL = 0.0
xR = 0.5
yB = 0.0
yT = 0.05
x_star = 0.225
a = 1.0

# 添加点
gmsh.model.geo.addPoint(xL, yB, 0.0, lc, 1)
gmsh.model.geo.addPoint(x_star, yB, 0.0, lc, 2)
gmsh.model.geo.addPoint(x_star + a * yT, yT, 0.0, lc, 3)
gmsh.model.geo.addPoint(xL, yT, 0.0, lc, 4)
gmsh.model.geo.addPoint(xR, yB, 0.0, lc, 5)
gmsh.model.geo.addPoint(xR, yT, 0.0, lc, 6)

# 添加线
gmsh.model.geo.addLine(1, 2, 1)  # 底边左
gmsh.model.geo.addLine(2, 3, 2)  # 中间斜线
gmsh.model.geo.addLine(3, 4, 3)  # 顶边左
gmsh.model.geo.addLine(4, 1, 4)  # 左边
gmsh.model.geo.addLine(2, 5, 5)  # 底边右
gmsh.model.geo.addLine(5, 6, 6)  # 右边
gmsh.model.geo.addLine(6, 3, 7)  # 顶边右

# 🔥 先同步几何，然后检查存在的实体
gmsh.model.geo.synchronize()

# 🔥 查看创建的线（调试用）
curves = gmsh.model.getEntities(1)
println("创建的线: ", curves)

# 🔥 简化或删除周期性设置（如果不需要的话）
# 周期性边界条件可能不是必需的，可以先注释掉
# dx = abs(xR - xL)
# dy = abs(yT - yB)
# to_right = [1.0, 0.0, 0.0, dx, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
# to_up = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, dy, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

# gmsh.model.mesh.setPeriodic(1, [6], [4], to_right)
# gmsh.model.mesh.setPeriodic(1, [3], [1], to_up)
# gmsh.model.mesh.setPeriodic(1, [7], [5], to_up)

# 添加曲线环和表面
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)     # 左侧表面
gmsh.model.geo.addCurveLoop([5, 6, 7, -2], 2)    # 右侧表面（注意-2表示线2的反向）

surface_tag1 = gmsh.model.geo.addPlaneSurface([1])
surface_tag2 = gmsh.model.geo.addPlaneSurface([2])

gmsh.model.geo.synchronize()

# 添加物理组
gmsh.model.addPhysicalGroup(2, [surface_tag1], 1)
gmsh.model.setPhysicalName(2, 1, "surface1")
gmsh.model.addPhysicalGroup(2, [surface_tag2], 2)
gmsh.model.setPhysicalName(2, 2, "surface2")

# # 设置底边的节点数量
gmsh.model.mesh.setTransfiniteCurve(1, 2)  # 左底边：2个节点
gmsh.model.mesh.setTransfiniteCurve(5, 2)  # 右底边：2个节点

# 设置顶边的节点数量  
gmsh.model.mesh.setTransfiniteCurve(3, 2)  # 左顶边：2个节点
gmsh.model.mesh.setTransfiniteCurve(7, 2)  # 右顶边：2个节点

# 设置垂直边和斜边的节点数量
gmsh.model.mesh.setTransfiniteCurve(4, 2)  # 左边：2个节点
gmsh.model.mesh.setTransfiniteCurve(6, 2)  # 右边：2个节点
gmsh.model.mesh.setTransfiniteCurve(2, 2)  # 中间斜边：2个节点

# 🔥 设置表面为结构化（这样每个四边形只生成2个三角形）
gmsh.model.mesh.setTransfiniteSurface(surface_tag1)
gmsh.model.mesh.setTransfiniteSurface(surface_tag2)

# Add BCs
gmsh.model.addPhysicalGroup(1, [4], 3, "LEFT_INFLOW")
gmsh.model.addPhysicalGroup(1, [6], 4, "RIGHT_INFLOW")
gmsh.model.addPhysicalGroup(1, [1], 5, "BOTTOM_INFLOW_1")
gmsh.model.addPhysicalGroup(1, [5], 6, "BOTTOM_INFLOW_2")
gmsh.model.addPhysicalGroup(1, [3], 7, "TOP_INFLOW_1")
gmsh.model.addPhysicalGroup(1, [7], 8, "TOP_INFLOW_2")
gmsh.model.addPhysicalGroup(1, [2], 9, "SHOCK")

# set mesh algorithm
gmsh.option.setNumber("Mesh.MshFileVersion", 4.0)
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.Algorithm", alg_number)

# 🔥 设置网格尺寸限制
# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)

gmsh.model.mesh.generate(2)

# save mesh
output_path = joinpath(@__DIR__, "$name.msh")
gmsh.write(output_path)

# run gmsh gui
if !("-nopopup" in ARGS)
    gmsh.fltk.run()
end

# finalize
gmsh.finalize()