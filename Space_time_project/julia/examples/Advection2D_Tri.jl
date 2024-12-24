using OrdinaryDiffEq, LinearAlgebra, Plots, RecursiveArrayTools
using Logging: global_logger
using TerminalLoggers: TerminalLogger
using WriteVTK
global_logger(TerminalLogger())

include("../src/element_type.jl")
using .ElementType

# mesh_file = "./meshes/2d_grid.msh"
mesh_file = "/home/patel/projects/pde_project/element_type/meshes/2d_grid.msh"
ref_elems_data = Dict{DataType, RefElemData}(Triangle => TriangleDiagELGL(1, 1),
                                             Triangle3 => TriangleDiagELGL(2, 3))
grid = read_mesh(mesh_file, ref_elems_data)

a = [1.0 1.0]
bc(x::Coord, t) = sinpi(x[1] + x[2] - 2 * t)
initial_condition(x::Coord) = sinpi(x[1] + x[2])

function cell_deriv!(du, u, p, cell_id)
    grid, a = p
    cell = get_cells(grid, cell_id)
    elem = get_ref_elem_data(grid, cell)
    Λ_q = @views grid.geometric_terms.Λ_q[cell_id]
    Dr, Ds = elem.D
    Dx = 0.5 *
         (Λ_q[:, 1, 1] .* Dr .+ Dr .* Λ_q[:, 1, 1]' .+ Λ_q[:, 2, 1] .* Ds .+ Ds .* Λ_q[:, 2, 1]')
    Dy = 0.5 *
         (Λ_q[:, 1, 2] .* Dr .+ Dr .* Λ_q[:, 1, 2]' .+ Λ_q[:, 2, 2] .* Ds .+ Ds .* Λ_q[:, 2, 2]')

    ux = a[1] * Dx * u[cell_id]
    uy = a[2] * Dy * u[cell_id]
    du[cell_id] = -ux - uy
end

function SAT!(du, u, interface, p)
    grid, a = p
    c1, lf1 = interface.face_1
    c2, lf2 = interface.face_2
    P1, P2 = interface.P1, interface.P2
    elem1, elem2 = get_ref_elem_data(grid, grid.cells[c1]), get_ref_elem_data(grid, grid.cells[c2])
    R_1, R_2 = elem1.R[lf1], elem2.R[lf2]
    mask1, mask2 = @views elem1.f_mask[lf1], elem2.f_mask[lf2]
    n_1, n_2 = @views grid.geometric_terms.N_f[c1][:, mask1], grid.geometric_terms.N_f[c2][:,
                                                                                           mask2]

    lambda_1, lambda_2 = vec(a * n_1), vec(a * n_2)
    u1_face, u2_face = R_1 * u[c1], R_2 * u[c2]
    u1_adj, u2_adj = u2_face[P2], u1_face[P1]
    flux_1 = 0.5 *
             ((lambda_1 .+ abs.(lambda_1)) .* u1_face .+ (lambda_1 .- abs.(lambda_1)) .* u1_adj)
    flux_2 = 0.5 *
             ((lambda_2 .+ abs.(lambda_2)) .* u2_face .+ (lambda_2 .- abs.(lambda_2)) .* u2_adj)
    du[c1] += elem1.H_inv * R_1' * elem1.H_face * (lambda_1 .* u1_face .- flux_1)
    du[c2] += elem2.H_inv * R_2' * elem2.H_face * (lambda_2 .* u2_face .- flux_2)
end

function boundary_condition!(du, u, t, cell_id, face_id, p)
    grid, a = p
    elem = get_ref_elem_data(grid, grid.cells[cell_id])
    R = elem.R[face_id]
    mask = @views elem.f_mask[face_id]

    lambda = a * grid.geometric_terms.N_f[cell_id][:, mask] |> vec
    u_face = R * u[cell_id]
    face_coords = @view grid.xyz_f[cell_id][mask]
    u_adj = bc.(face_coords, t)
    flux = 0.5 * ((lambda .+ abs.(lambda)) .* u_face .+ (lambda .- abs.(lambda)) .* u_adj)
    du[cell_id] += elem.H_inv * R' * elem.H_face * (lambda .* u_face .- flux)
end

function RHS!(du, u, p, t)
    grid, _ = p
    for cell_id in 1:n_cells(grid)
        cell_deriv!(du, u, p, cell_id)
    end

    for interface in grid.face_interfaces
        SAT!(du, u, interface, p)
    end

    boundary_faces = union(get_face_set(grid, "INFLOW_1"), get_face_set(grid, "INFLOW_2")) |>
                     collect
    for (cell_id, face_id) in boundary_faces
        boundary_condition!(du, u, t, cell_id, face_id, p)
    end

    for cell_id in 1:n_cells(grid)
        du[cell_id] = du[cell_id] ./ grid.geometric_terms.J_q[cell_id]
    end
end

u0 = Vector{Vector{Float64}}(undef, n_cells(grid))
du = deepcopy(u0)
for i in 1:n_cells(grid)
    u0[i] = initial_condition.(grid.xyz_q[i])
    du[i] = zeros(length(u0[i]))
end
u0, du = VectorOfArray(u0), VectorOfArray(du)
println("Solving...")
t_span = (0.0, 1.0)
p = (grid, a)
prob = ODEProblem(RHS!, u0, t_span, p)
sol = solve(prob, Tsit5(), progress = true, progress_steps = 10, dtmax = 0.01)
u_final = sol[end]

x, y = Float64[], Float64[]
for i in axes(u0)[2]
    append!(x, get_i_coordinates(grid.xyz[i], 1))
    append!(y, get_i_coordinates(grid.xyz[i], 2))
end

function export_vtk(filename::String, coords::AbstractMatrix{T}, u; variable_name = "u") where {T}
    N_e = size(u, 2)
    nodes_per_cell = size(u, 1)
    cells = [MeshCell(VTKCellTypes.VTK_LAGRANGE_TRIANGLE,
                      collect(((k - 1) * nodes_per_cell + 1):(k * nodes_per_cell))) for k in 1:N_e]
    vtk_grid(filename, coords, cells) do vtk
        vtk[variable_name] = vec(u)
    end
end

using Dates
dir = "./frames/$(Dates.format(now(), "d_m_HH_MM"))"
isdir(dir) || mkdir(dir)
for i in eachindex(sol.u)
    export_vtk("$(dir)/sol_$(lpad(i, 4, '0')).vtu", [x y]', sol.u[i])
end

# anim = @animate for i in 1:length(sol)
#     scatter3d(x, y, vec(sol[i]), xlabel = "x", ylabel = "y", zlabel = "u",
#               title = "t = $(sol.t[i])", xlims = (0, 1), ylims = (0, 1), zlims = (-1.5, 1.5),
#               alpha = 0.8, camera = (60, 30), width = 1600, height = 900)
# end
# gif(anim, "Advection2D_Tri.gif", fps = 60)

# Save frames of surface plots
# using PyPlot
# for i in 1:length(sol)
#     z = vec(sol[i])
#     fig = PyPlot.plt.figure(figsize = (8, 8), dpi = 160)
#     axis = fig.add_subplot(111, projection = "3d")
#     axis.view_init(elev = 30, azim = 135)
#     surf = axis.plot_trisurf(x, y, z, cmap = PyPlot.cm.turbo)
#     fig.colorbar(surf)
#     PyPlot.plt.xlabel("x")
#     PyPlot.plt.ylabel("y")
#     PyPlot.plt.title("t = $(sol.t[i])")
#     PyPlot.plt.savefig("$(dir)/$(i).png")
#     PyPlot.plt.close(fig)
# end
