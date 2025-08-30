include("../Euler/src/parameter.jl")
include("SimpleGrid_Euler.jl")


buffer = SpaceTimeBuffer(grid, [S1, S2, S3], analytic_U)
buffer.indices_moving_coords
buffer.indices_moving_coords = [5, 7]


# construct_simpleGrid(
        # grid.cells, buffer.ref, new_xyz_gmsh_coords, 
        # grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
    # )


initial_xyz_gmsh = buffer.initial_guess[1:buffer.index_mesh]
new_xyz_gmsh = deepcopy(initial_xyz_gmsh) # 初始化 new_xyz_gmsh 为初始值
new_xyz_gmsh_coords = [Coord{2, Float64}((new_xyz_gmsh[i], new_xyz_gmsh[8 + i])) for i in 1:8]

test_simple_grid = construct_simpleGrid(
        grid.cells, buffer.ref, new_xyz_gmsh_coords, grid.face_interfaces, 
        grid.face_sets, buffer.indices_moving_coords, 
        buffer.bottom_faceIndex,
        buffer.boundary_faceIndex
    )


model = Model(() -> MadNLP.Optimizer(
    linear_solver=MadNLPHSL.Ma57Solver,
    print_level=MadNLP.INFO,
    max_iter=100,                       # 减少最大迭代次数

))


grid = buffer.grid
num_of_element = length(grid.VOL) # 因为可能要用自己写的 SimpleGrid， 这里先用 VOL 来算 element/node 数量
num_of_nodes = size(grid.VOL[1][1])[1]

num_of_xyz_gmsh = length(grid.xyz_gmsh)

# 计算 参考单元的 失真度
ref = buffer.ref
area_of_ref_elem = sum(ref.H)
G_ref = I(2) # ∂ ξ / ∂ ξ = I_2
δ_ref = norm(G_ref) / det(G_ref)^0.5
δref2 = δ_ref^2
r_msh_ref = area_of_ref_elem * δ_ref^2

n_total = length(buffer.indices_moving_coords) + length(buffer.initial_guess) - buffer.index_mesh

# 储存最开始的 xyz_gmsh 坐标, 在 solver 迭代更新的过程中，可以保证只有 能动的在动，不能动的一直没变
initial_xyz_gmsh = buffer.initial_guess[1:buffer.index_mesh]
new_xyz_gmsh = deepcopy(initial_xyz_gmsh) # 初始化 new_xyz_gmsh 为初始值
new_xyz_gmsh_coords = [Coord{2, Float64}((new_xyz_gmsh[i], new_xyz_gmsh[num_of_xyz_gmsh + i])) for i in 1:num_of_xyz_gmsh]
current_grid = construct_simpleGrid(
        grid.cells, buffer.ref, new_xyz_gmsh_coords, grid.face_interfaces, 
        grid.face_sets, buffer.indices_moving_coords, 
        buffer.interfaces_aligning_shock, 
        buffer.interfaces_aligning_contact_wave,
        buffer.smooth_interior_interfaces,
        buffer.bottom_faceIndex,
        buffer.boundary_faceIndex
    )

# 不是所有的坐标都能动，边界上只有一个坐标，非边界上两个都能动，但corner上两个都不能动
u0_vec = vcat(buffer.initial_guess[buffer.indices_moving_coords], buffer.initial_guess[buffer.index_mesh+1:end])
if length(u0_vec) != n_total
    error("The length of u0_vec must be equal to n_total")
end
@variable(model, u[i=1:n_total], start = u0_vec[i])


mesh_u_vec = deepcopy(u0_vec)
mesh_vec = mesh_u_vec[1: length(buffer.indices_moving_coords)] # 前几个放的是可以动的坐标
u_vec = mesh_u_vec[(length(buffer.indices_moving_coords) + 1):end]  # 后面放的是解变量

new_xyz_gmsh = Vector{Float64}(initial_xyz_gmsh)  # ✅ 类型匹配
new_xyz_gmsh[buffer.indices_moving_coords] = mesh_vec
new_xyz_gmsh_coords = [Coord{2, Float64}((new_xyz_gmsh[i], new_xyz_gmsh[num_of_xyz_gmsh + i])) for i in 1:num_of_xyz_gmsh]

# current_grid = construct_simpleGrid(
#     grid.cells, buffer.ref, 
#     new_xyz_gmsh_coords,
#     grid.face_interfaces, grid.face_sets, buffer.indices_moving_coords
# )


Umatrix = reshape(u_vec, (num_of_element, num_of_nodes, 3))
dU = similar(Umatrix, Float64) 
fill!(dU, zero(Float64))  # 初始化 dU

RHS_for_solution(dU, Umatrix, current_grid)  # 计算残差


W = evaluate_W_forSolver(Umatrix)
One = ones(length(grid.xyz[1]) * 3)
