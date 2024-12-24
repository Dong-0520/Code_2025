using DifferentialEquations, LinearAlgebra
include("../element_type.jl")
using .ElementType

# mesh_file = "./meshes/2d_grid_diagonal_band.msh"
mesh_file = "./meshes/2d_grid.msh"
p = 2
ref_elems = Dict{DataType, ElementType.RefElem}(QuadraticQuadrilateral => QuadRefElemLGL(p),
                                                Quadrilateral => QuadRefElemLGL(1),
                                                Triangle => TriRefDiagEVertices(1, 2))

nel = (10, 10)
LL = (0.0, 0.0);
LR = (1.0, 0.0);
UR = (1.0, 1.0);
UL = (0.0, 1.0);

# grid = read_mesh(mesh_file, ref_elems)
grid = generate_grid(QuadraticQuadrilateral, ref_elems, nel, LL, LR, UR, UL)

hx, hy = 1 / nel[1], 1 / nel[2]
elem = grid.ref_elems[QuadraticQuadrilateral]
Qr = elem.Q
Pr = diagm(elem.w)
Qs = deepcopy(Qr)
Ps = deepcopy(Pr)
Dr = inv(Pr) * Qr
Ds = inv(Ps) * Qs
Dx, Dy = kron(I(3), Dr) * 2 / hx, kron(Ds, I(3)) * 2 / hy
Px = (hx / 2) * Pr
Py = (hy / 2) * Ps
P = Matrix{Float64}(kron(Py, Px))
Pinv = inv(P)

# Okay, now we will iterate over the face interfaces and compute "RHS" evaluation at each interface.
# Each interface is a pair of FaceIndex (which is a pair of cell index and face index). Thus, we
# can compute flux contributions of two faces in a single pass through the faces.

# # Model problem: 2D linear advection equation using summation by parts
# BC
g(x::Coord, t) = sinpi(x[1] + x[2] - 2 * t)

function RHS!(du, u, p, t)
    grid, Dx, Dy, Pinv, Px, Py = p
    for (cell_id, cell) in enumerate(get_cells(grid))
        local_sol = u[cell_id, :]
        ux = Dx * local_sol
        uy = Dy * local_sol
        du[cell_id, :] = -ux - uy
    end

    for interface in grid.topology.face_interfaces
        if interface.face_1[2] == 3 && interface.face_2[2] == 1 ||
           interface.face_1[2] == 2 && interface.face_2[2] == 4
            c1, lf1 = interface.face_1
            c2, lf2 = interface.face_2
        else
            c2, lf2 = interface.face_1
            c1, lf1 = interface.face_2
        end
        cell1, cell2 = get_cells(grid, [c1, c2])

        # Interior state for the elements on either side of this interface
        u1_int = face(lf1, get_ref_elem(grid, cell1), u[c1, :])
        u2_int = face(lf2, get_ref_elem(grid, cell2), u[c2, :])
        u1_ext = interface.flipped ? reverse(u2_int) : u2_int
        u2_ext = interface.flipped ? reverse(u1_int) : u1_int

        # Upwind SAT with dissipation
        # du[c1, :] += -0.5 * Pinv[:, face_dof_indices(cell1, lf1)] * Py * (u1_int - u1_ext)
        du[c2, :] += -Pinv[:, face_dof_indices(cell2, lf2)] * Py * (u2_int - u2_ext)
    end
    for (cell_id, face_id) in get_face_set(grid, "INFLOW")
        cell = get_cells(grid, cell_id)
        u_int = face(face_id, get_ref_elem(grid, cell), u[cell_id, :])
        cell_coords = get_cell_coords(grid, cell_id)[face_dof_indices(cell, face_id)]
        u_ext = g.(cell_coords, t)
        du[cell_id, :] += -Pinv[:, face_dof_indices(cell, face_id)] * Px * (u_int - u_ext)
    end
end

n_cells = length(grid.cells)
nodes_per_cell = n_nodes(grid.cells[1])                 #Uniform number of nodes per cell for now
u = Matrix{Float64}(undef, n_cells, nodes_per_cell)
fill!(u, 0.0)
du = deepcopy(u)
fill!(du, 0.0)


for i in axes(u)[1]
    # Initial condition
    cell_coords = ElementType.get_cell_coords(grid, i)
    for j in axes(u)[2]
        u[i, j] = sinpi(cell_coords[j][1] + cell_coords[j][2])
    end
end

println("Solving...")
t_span = (0.0, 1.0)

p = (grid, Dx, Dy, Pinv, Px, Py)
# RHS!(du, u, p, 0.01)

prob = ODEProblem(RHS!, u, t_span, p)
sol = solve(prob, Tsit5(), dtmax = 0.01)
u_final = sol[end]

function get_cell_coords_as_array(grid)
    x, y = Float64[], Float64[]
    for (i, cell) in enumerate(get_cells(grid))
        cell_coords = get_cell_coords(grid, cell)
        for coord in cell_coords
            push!(x, coord[1])
            push!(y, coord[2])
        end
    end
    return x, y
end
x, y = get_cell_coords_as_array(grid)
using Plots

anim = @animate for i in 1:length(sol)
    z = vec(sol[i]')
    scatter(x, y, z, xlabel = "x", ylabel = "y", zlabel = "u", title = "t = $(sol.t[i])")
end
gif(anim, "advection_2d.gif", fps = 15)
