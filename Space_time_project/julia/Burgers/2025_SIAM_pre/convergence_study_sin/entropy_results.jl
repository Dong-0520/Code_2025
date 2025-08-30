using LinearAlgebra, SparseArrays
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics
using SparseConnectivityTracer, ADTypes
using NonlinearSolve                                                                                                                                                                                                                                                          

include("../../../src/SBPLite.jl")
using .SBPLite
include("../../plotting_helper.jl")

order = 1
mesh = 3
ref = TriangleDiagELGL(order, (2 * order) -1)
ref_elems_data = Dict{String, SBPLite.RefElemData}("Triangle 3" => ref)


# This is random larger mesh from 0 to 0.15
grid = read_mesh(joinpath(@__DIR__, "Burgers_sin_$mesh.msh"), ref_elems_data, Base.identity)
zlimit = [-1.1, 1.1]
nodes_per_cell = length(grid.xyz_q[1])

include(joinpath(@__DIR__, "../helper_functions.jl"))
# ---------------------- slab 1 ---------------------- #
p1 = get_set_up(grid, slab = 1)
U0 = initial_guess(grid)
@time all_U, diffs = space_time_solve(U0, p1, pseudo_dt = 0.0001, num_of_pseudo_time_step = 1000, diss = true, test = true)
u1 = all_U[argmin(diffs)]
U0_for_solver = u1
sol_slab1 = solver_wrapper(u1, p1, maximum_iteration = 100, solver = NewtonRaphson())
u1 = sol_slab1.u
#---------------------- slab 2 ---------------------- #
p2 = get_set_up(grid, u = u1, slab = 2)
@time all_U2, diffs2 = space_time_solve(u1, p2, pseudo_dt = 0.0005, num_of_pseudo_time_step = 2000, diss = true, test = true)
u2 = all_U2[argmin(diffs2)]
sol_slab2 = solver_wrapper(u2, p2, maximum_iteration = 100, solver = NewtonRaphson())
u2 = sol_slab2.u
# observe_numerical_sol(grid, u2, zlimit, show_true_sol = false)
#---------------------- slab 3 ---------------------- #
p3 = get_set_up(grid, u = u2, slab = 3)
@time all_U3, diffs3 = space_time_solve(u2, p3, pseudo_dt = 0.00005, num_of_pseudo_time_step = 4000, diss = true, test = true)
# gif(observe_iteration(grid, all_U3[1:20:end], zlimit, show_true_sol = false), joinpath(@__DIR__, "numerical_test_discts.gif"), fps = 2)
u3 = all_U3[argmin(diffs3)]
sol_slab3 = solver_wrapper(u3, p3, maximum_iteration = 100, solver = TrustRegion())
# sol_slab3 = solver_wrapper(sol_slab3.u, p3, maximum_iteration = 1000, solver = TrustRegion())
# sol_slab3 = solver_wrapper(sol_slab3.u, p3, maximum_iteration = 1000, solver = LevenbergMarquardt())
sol_slab3 = solver_wrapper(sol_slab3.u, p3, maximum_iteration = 50, solver = NewtonRaphson())
u3 = sol_slab3.u
# observe_numerical_sol(grid, u3, zlimit, show_true_sol = false)
#---------------------- slab 4 ---------------------- #
p4 = get_set_up(grid, u = u3, slab = 4)
all_U4, diffs4 = space_time_solve(u3, p4, pseudo_dt = 0.00005, num_of_pseudo_time_step = 4000, diss = true, test = true)
# gif(observe_iteration(grid, all_U4[4000:20:6000], zlimit, show_true_sol = false), "Burgers_sin.gif", fps = 5)
u4 = all_U4[argmin(diffs4)]
# u4 = all_U4[4000]
# observe_numerical_sol(grid, u4, zlimit, show_true_sol = false)
sol_slab4 = solver_wrapper(u4, p4, maximum_iteration = 1000, solver = LevenbergMarquardt())
sol_slab4 = solver_wrapper(sol_slab4.u, p4, maximum_iteration = 1000, solver = TrustRegion())
# sol_slab4 = solver_wrapper(sol_slab4.u, p4, maximum_iteration = 1000, solver = LevenbergMarquardt())
sol_slab4 = solver_wrapper(sol_slab4.u, p4, maximum_iteration = 100, solver = NewtonRaphson())
# observe_numerical_sol(grid, sol_slab4.u, zlimit, show_true_sol = false)
u4 = sol_slab4.u
plot_numerical_sol_at_final_time_perSlab(grid, u4)

#---------------------- slab 5 ---------------------- #
p5 = get_set_up(grid, u = u4, slab = 5)
all_U5, diffs5 = space_time_solve(u4, p5, pseudo_dt = 0.00005, num_of_pseudo_time_step = 500, diss = true, test = true)
# gif(observe_iteration(grid, all_U5[1:5:500], zlimit, show_true_sol = false), "Numerical_test.gif", fps = 3)
u5 = all_U5[argmin(diffs5)]
sol_slab5 = solver_wrapper(u5, p5, maximum_iteration = 100, solver = NewtonRaphson())
sol_slab5 = solver_wrapper(sol_slab5.u, p5, maximum_iteration = 1000, solver = LevenbergMarquardt())
sol_slab5 = solver_wrapper(sol_slab5.u, p5, maximum_iteration = 1000, solver = TrustRegion())
sol_slab5 = solver_wrapper(sol_slab5.u, p5, maximum_iteration = 10, solver = NewtonRaphson())
u5 = sol_slab5.u
observe_numerical_sol(grid, sol_slab5.u, zlimit, show_true_sol = false)

# ---------------------- slab 6 ---------------------- #
p6 = get_set_up(grid, u = u5, slab = 6)
all_U6, diffs6 = space_time_solve(u5, p6, pseudo_dt = 0.000025, num_of_pseudo_time_step = 500, diss = true, test = true)
u6 = all_U6[argmin(diffs6)]
# observe_numerical_sol(grid, u6, zlimit, show_true_sol = false)
sol_slab6 = solver_wrapper(u6, p6, maximum_iteration = 50, solver = NewtonRaphson())
sol_slab6 = solver_wrapper(u6, p6, maximum_iteration = 1000, solver = TrustRegion())
sol_slab6 = solver_wrapper(sol_slab6.u, p6, maximum_iteration = 1000, solver = TrustRegion())
sol_slab6 = solver_wrapper(sol_slab6.u, p6, maximum_iteration = 3000, solver = LevenbergMarquardt())
sol_slab6 = solver_wrapper(sol_slab6.u, p6, maximum_iteration = 100, solver = NewtonRaphson())
u6 = sol_slab6.u
observe_numerical_sol(grid, sol_slab6.u, zlimit, show_true_sol = false)
# ---------------------- slab 7 ---------------------- #
p7 = get_set_up(grid, u = u6, slab = 7)
all_U7, diffs7 = space_time_solve(u6 * 0, p7, pseudo_dt = 0.00005, num_of_pseudo_time_step = 2000, diss = true, test = true)
u7 = all_U7[argmin(diffs7)]
sol_slab7 = solver_wrapper(u7, p7, maximum_iteration = 100, solver = TrustRegion())
sol_slab7 = solver_wrapper(sol_slab7.u, p7, maximum_iteration = 50, solver = NewtonRaphson())
u7 = sol_slab7.u
# observe_numerical_sol(grid, u7, zlimit, show_true_sol = false)
# ---------------------- slab 8 ---------------------- #
p8 = get_set_up(grid, u = u7, slab = 8)
all_U8, diffs8 = space_time_solve(u7, p8, pseudo_dt = 0.000025, num_of_pseudo_time_step = 500, diss = true, test = true)
u8 = all_U8[argmin(diffs8)]
sol_slab8 = solver_wrapper(u8, p8, maximum_iteration = 100, solver = NewtonRaphson())
u8 = sol_slab8.u
# ---------------------- slab 9 ---------------------- #
p9 = get_set_up(grid, u = u8, slab = 9)
all_U9, diffs9 = space_time_solve(u8, p9, pseudo_dt = 0.000025, num_of_pseudo_time_step = 500, diss = true, test = true)
u9 = all_U9[argmin(diffs9)]
sol_slab9 = solver_wrapper(u9, p9, maximum_iteration = 100, solver = NewtonRaphson())
u9 = sol_slab9.u
# observe_numerical_sol(grid, u9, zlimit, show_true_sol = false)
# ---------------------- slab 10 ---------------------- #
p10 = get_set_up(grid, u = u9, slab = 10)
all_U10, diffs10 = space_time_solve(u9 * 0, p10, pseudo_dt = 0.000025, num_of_pseudo_time_step = 4000, diss = true, test = true)
u10 = all_U10[argmin(diffs10)]
# observe_numerical_sol(grid, u10, zlimit, show_true_sol = false)
# gif(observe_iteration(grid, all_U10[2000:10:3000], zlimit, show_true_sol = false), "Numerical_test.gif", fps = 3)
sol_slab10 = solver_wrapper(u10, p10, maximum_iteration = 1000, solver = TrustRegion())
sol_slab10 = solver_wrapper(sol_slab10.u, p10, maximum_iteration = 1000, solver = LevenbergMarquardt())
sol_slab10 = solver_wrapper(sol_slab10.u, p10, maximum_iteration = 30, solver = NewtonRaphson())
sol_slab10 = solver_wrapper(sol_slab10.u, p10, maximum_iteration = 1000, solver = TrustRegion())
u10 = sol_slab10.u
# observe_numerical_sol(grid, sol_slab10.u, zlimit, show_true_sol = false)
# scatter_gif_numerical_sol_for_all_slabs(grid, [u1, u2, u3, u4, u5, u6, u7, u8], 0.025, save_path = joinpath(@__DIR__,"Burgers_sin.gif"))
plot_gif_numerical_sol_for_all_slabs(grid, [u1, u2, u3, u4, u5, u6, u7, u8, u9, u10], 0.025, save_path = joinpath(@__DIR__,"numerical_simulation/Burgers_sin_order$(order)_mesh$(mesh).gif"), color = :blue, fps=2)
@save joinpath(@__DIR__, "numerical_simulation/Burgers_sin_order$(order)_mesh$(mesh).jld2") u1 u2 u3 u4 u5 u6 u7 u8 u9 u10 grid



include(joinpath(@__DIR__, "../helper_functions.jl"))

p, x_sorted_left, u_sorted_left, x_sorted_right, u_sorted_right = plot_numerical_sol_at_final_time_perSlab(grid, u4)
p
plot_gif_numerical_sol_for_all_slabs(grid, [u1, u2, u3, u4], 0.025, save_path = joinpath(@__DIR__,"numerical_simulation/Burgers_sin_order$(order)_mesh$(mesh).gif"), color = :blue, fps=2)



@load joinpath(@__DIR__, "numerical_simulation/Burgers_sin_order4_mesh8.jld2") u1 u2 u3 u4 u5 u6 u7 u8 u9 u10 grid
plot_gif_numerical_sol_for_all_slabs(grid, [u1, u2, u3, u4, u5, u6, u7, u8, u9, u10], 0.025, save_path = joinpath(@__DIR__,"numerical_simulation/Burgers_sin_order4_mesh8_changed.gif"), color = :blue, fps=2)

# ---------------------- compute entropy ---------------------- #
# entropy datas
x_coords = get_xCoords_for_plotting_sol(grid)
num_u1 = get_final_u_for_plotting_sol(grid, u1)
num_u2 = get_final_u_for_plotting_sol(grid, u2)
num_u3 = get_final_u_for_plotting_sol(grid, u3)
num_u4 = get_final_u_for_plotting_sol(grid, u4)
num_u5 = get_final_u_for_plotting_sol(grid, u5)
num_u6 = get_final_u_for_plotting_sol(grid, u6)
num_u7 = get_final_u_for_plotting_sol(grid, u7)
num_u8 = get_final_u_for_plotting_sol(grid, u8)
num_u9 = get_final_u_for_plotting_sol(grid, u9)
num_u10 = get_final_u_for_plotting_sol(grid, u10)
Pnorms = get_Pnorm_for_finalTimeInterface(grid)
entropys = [compute_L2_norm_of_Initialcondition(p1),
            compute_L2_norm_atFinalInterface(num_u1, Pnorms),
            compute_L2_norm_atFinalInterface(num_u2, Pnorms),
            compute_L2_norm_atFinalInterface(num_u3, Pnorms),
            compute_L2_norm_atFinalInterface(num_u4, Pnorms),
            compute_L2_norm_atFinalInterface(num_u5, Pnorms),
            compute_L2_norm_atFinalInterface(num_u6, Pnorms),
            compute_L2_norm_atFinalInterface(num_u7, Pnorms),
            compute_L2_norm_atFinalInterface(num_u8, Pnorms),
            compute_L2_norm_atFinalInterface(num_u9, Pnorms),
            compute_L2_norm_atFinalInterface(num_u10, Pnorms)]

# mesh 3, order 1
#  0.4999999999999999
#  0.4999898391277122
#  0.49994257639902867
#  0.4992655609276441
#  0.4905460604830065
#  0.45677271720131896
#  0.4197269831501198
#  0.38544787312477563
#  0.3548235333024736
#  0.3280271401243519
#  0.3046651904485152

# mesh 4, order 3
# 0.5
# 0.49999999863800926
# 0.4999999461342896
# 0.49995801449107086
# 0.488856388626816
# 0.4564128130265872
# 0.41972470302462006
# 0.38535377157108686
# 0.3547940481084001
# 0.32803236579494455
# 0.3046533575493162



# mesh 12, order 4
# 0.5
# 0.4999998954016056
# 0.49999017963829945
# 0.4997376715368088
# 0.4895231647192377
# 0.456256435463583
# 0.41978889461050645
# 0.38530543385862326
# 0.35472690081699665
# 0.32806636566133895
# 0.3047173653369


# mesh 3, order 2
# 0.4999999999999999
# 0.4999994458452124
# 0.49999758976583564
# 0.49990382798365124
# 0.4889366534661529
# 0.45648521467971964
# 0.41971278754039487
# 0.38535498666630347
# 0.3547675061981436
# 0.3278836234055266
# 0.30440624175432174


using Plots

using Plots, LaTeXStrings


# 示例数据（四组数据，每组数据长度相同）
order_labels = ["order1", "order2", "order3", "order4"]
x_values = 0:0.025:0.25
y_values = 1:4  # 对应 order1, order2, order3, order4


order1_entropy = [0.4999999999999999, 0.4999898391277122, 0.49994257639902867, 0.4992655609276441, 0.4905460604830065, 0.45677271720131896, 0.4197269831501198, 0.38544787312477563, 0.3548235333024736, 0.3280271401243519, 0.3046651904485152]
order2_entropy = [0.4999999999999999, 0.4999994458452124, 0.49999758976583564, 0.49990382798365124, 0.4889366534661529, 0.45648521467971964, 0.41971278754039487, 0.38535498666630347, 0.3547675061981436, 0.3278836234055266, 0.30440624175432174]
order3_entropy = [0.5, 0.49999999863800926, 0.4999999461342896, 0.49995801449107086, 0.488856388626816, 0.4564128130265872, 0.41972470302462006, 0.38535377157108686, 0.3547940481084001, 0.32803236579494455, 0.3046533575493162]
order4_entropy = [0.5, 0.4999998954016056, 0.49999017963829945, 0.4997376715368088, 0.4895231647192377, 0.456256435463583, 0.41978889461050645, 0.38530543385862326, 0.35472690081699665, 0.32806636566133895, 0.3047173653369]

data = [
    order1_entropy, order2_entropy, order3_entropy, order4_entropy
]

# 画瀑布图
p = plot3d(size=(600, 600), legend=false, xlabel=L"t", ylabel="", zlabel=L"\Vert u \Vert_{L_2}",
           xticks=(x_values, string.(x_values)), yticks=(y_values, order_labels))

# 遍历每一组数据并画出曲线
for i in 1:length(data)
    plot3d!(x_values, fill(y_values[i], length(x_values)), data[i], lw=2)
end

p

savefig(p, joinpath(@__DIR__, "numerical_simulation/Burgers_sin_entropy_result.png"))

