using LinearAlgebra, BlockArrays, SparseArrays
using JLD2
using Plots
using DoubleFloats
using Logging
using ProgressBars
using LoopVectorization      
using Statistics                                                                                                                                                                                                                                                                          

include("../src/SBPLite.jl")
using .SBPLite
include("plotting_helper.jl")

using Test

ref = QuadRefElemFromDiagETriangle(3,6,true)
tol = 1e-12

@testset "accuracy test of quad from triangle operaotrs" begin
    println("====================")
    for p = 1:4
        ref = QuadRefElemFromDiagETriangle(p, 2 * p, true)
        Dx = ref.D[1]
        Dy = ref.D[2]
        Ex = ref.E[1]
        Ey = ref.E[2]
        Qx = ref.Q[1]
        Qy = ref.Q[2]
        x = ref.rst[1,:]
        y = ref.rst[2,:]
        eEx = maximum(broadcast(abs,Ex-(Qx+transpose(Qx))))
        eEy = maximum(broadcast(abs,Ey-(Qy+transpose(Qy))))
        @test maximum(eEx)<= tol
        @test maximum(eEy)<= tol
        for r = 0:p
            for j = 0:r
                i = r-j
                u = (x.^i).*(y.^j)
                dudx = (i.*x.^max(0,i-1)).*(y.^j)
                dudy = (x.^i).*(j.*y.^max(0,j-1))
                ex = maximum(broadcast(abs,Dx*u-dudx))
                ey = maximum(broadcast(abs,Dy*u-dudy))
                @test maximum(ex)<= tol
                @test maximum(ey)<= tol
                #println("p = ",p," i = ",i," r = ",r," error:",ex," --- ",ey,"\n")
            end
        end
    end
end

@testset "accuracy test of quad from triangle operaotrs" begin
    println("====================")

    xL = 0.0
    xR = 2.0
    yB = 0.0
    yT = 2.0
    Nx = 2
    Ny = 2 

    for p = 1:4
        ref = QuadRefElemFromDiagETriangle(p, 2 * p, true)
        Dx = ref.D[1]
        Dy = ref.D[2]
        Ex = ref.E[1]
        Ey = ref.E[2]
        Qx = ref.Q[1]
        Qy = ref.Q[2]
        x = ref.rst[1,:]
        y = ref.rst[2,:]
        eEx = maximum(broadcast(abs,Ex-(Qx+transpose(Qx))))
        eEy = maximum(broadcast(abs,Ey-(Qy+transpose(Qy))))
        @test maximum(eEx)<= tol
        @test maximum(eEy)<= tol
        for r = 0:p
            for j = 0:r
                i = r-j
                u = (x.^i).*(y.^j)
                dudx = (i.*x.^max(0,i-1)).*(y.^j)
                dudy = (x.^i).*(j.*y.^max(0,j-1))
                ex = maximum(broadcast(abs,Dx*u-dudx))
                ey = maximum(broadcast(abs,Dy*u-dudy))
                @test maximum(ex)<= tol
                @test maximum(ey)<= tol
                #println("p = ",p," i = ",i," r = ",r," error:",ex," --- ",ey,"\n")
            end
        end
    end
end
r = refTri.rst[1,:]
s = refTri.rst[2,:]
y = 1 .+ r .+ s .+ r .* (s .^ (order - 1))

r = ref.rst[1,:]
s = ref.rst[2,:]
y = 1 .+ r .+ s .+ r .* (s .^ (order - 1))
dydr = 1 .+ s .^ (order - 1)
dyds = 1 .+ (order - 1) * r .* (s.^(order - 2))
maximum(abs.(ref.D[1] * y - dydr))
maximum(abs.(ref.D[2] * y - dyds))

r = ref_test.rst[1,:]
s = ref_test.rst[2,:]
y = 1 .+ r .+ s .+ r .* (s .^ (order - 1))
dydr = 1 .+ s .^ (order - 1)
dyds = 1 .+ (order - 1) * r .* (s.^(order - 2))
maximum(abs.(ref_test.D[1] * y - dydr))
maximum(abs.(ref_test.D[2] * y - dyds))




maximum(abs.(ref.D[1] - ref_test.D[1]))
maximum(abs.(ref.D[2] - ref_test.D[2]))
maximum(abs.(ref.rst - ref_test.rst))
maximum(abs.(ref.rst_q - ref_test.rst_q))
maximum(abs.(ref.rst_f - ref_test.rst_f))
maximum(abs.(ref.f_mask[1] - ref_test.f_mask[1]))
maximum(abs.(ref.f_mask[2] - ref_test.f_mask[2]))
maximum(abs.(ref.f_mask[3] - ref_test.f_mask[3]))
maximum(abs.(ref.f_mask[4] - ref_test.f_mask[4]))
maximum(abs.(ref.H - ref_test.H))
maximum(abs.(ref.H_inv - ref_test.H_inv))
maximum(abs.(ref.H_face - ref_test.H_face))
maximum(abs.(ref.Q[1] - ref_test.Q[1]))
maximum(abs.(ref.Q[2] - ref_test.Q[2]))
maximum(abs.(ref.E[1] - ref_test.E[1]))
maximum(abs.(ref.E[2] - ref_test.E[2]))
maximum(abs.(ref.Qt[1] - ref_test.Qt[1]))
maximum(abs.(ref.Qt[2] - ref_test.Qt[2]))
maximum(abs.(ref.Qt_inv - ref_test.Qt_inv))
maximum(abs.(ref.R[1] - ref_test.R[1]))
maximum(abs.(ref.R[2] - ref_test.R[2]))
maximum(abs.(ref.R[3] - ref_test.R[3]))
maximum(abs.(ref.R[4] - ref_test.R[4]))
maximum(abs.(ref.n_rst - ref_test.n_rst))

# deve

include("../src/cg_triangle.jl")

xt, yt, Pt, Qxt, Qyt, Ext, Eyt = cg_triangle.triangle_from_Jesse(degree)
quadξ, quadη, quadH, quadQξ, quadQη, quadEξ, quadEη = cg_triangle.quad_from_triangle_SBP(size(xt)[1], xt, yt, Pt, Qxt, Qyt, Ext, Eyt)
num_of_nodes = size(quadξ)[1]

rs = hcat(quadξ, quadη)'
rs_quad = deepcopy(rs)


Dr = inv(quadH)*quadQξ
Ds = inv(quadH)*quadQη
D = Dr, Ds

quadHinv = inv(quadH)


if degree == 1
    node_id_on_south = [1,4,2]
    node_id_on_east = [2,10,8]
    node_id_on_north = [8,9,3]
    node_id_on_west = [3,6,1]
elseif degree == 2
    node_id_on_south = [1, 7, 8, 2]
    node_id_on_east = [2, 19, 20, 13]
    node_id_on_north = [13, 17, 18, 3]
    node_id_on_west = [3, 11, 12, 1]
elseif degree == 3
    node_id_on_south = [1, 13, 4, 14, 2]
    node_id_on_east = [2, 30, 21, 31, 19]
    node_id_on_north = [19, 28, 20, 29, 3]
    node_id_on_west = [3, 17, 6, 18, 1]
elseif degree == 4
    node_id_on_south = [1, 16, 10, 11, 17, 2]
    node_id_on_east = [2, 41, 37, 38, 42, 28]
    node_id_on_north = [28, 39, 35, 36, 40, 3]
    node_id_on_west = [3, 20, 14, 15, 21, 1]
else
    error("Not implemented yet")
end
H_face_west = Diagonal([abs(quadEξ[i, i]) for i in node_id_on_west])
H_face_east = Diagonal([abs(quadEξ[i, i]) for i in node_id_on_east])
H_face_south = Diagonal([abs(quadEη[i, i]) for i in node_id_on_south])
H_face_north = Diagonal([abs(quadEη[i, i]) for i in node_id_on_north])
@assert H_face_west ≈ H_face_east ≈ H_face_south ≈ H_face_north  "H_faces should be the same, check your ref elem data"
H_face = H_face_west

Qr, Qs = quadQξ, quadQη

E = Qr + Qr', Qs + Qs'

Qt = [Qr' Qs']
Qt_inv = pinv([Qr' Qs'])

num_of_nodes_on_face = length(node_id_on_south)
R1 = zeros(num_of_nodes_on_face, num_of_nodes)
R2 = zeros(num_of_nodes_on_face, num_of_nodes)
R3 = zeros(num_of_nodes_on_face, num_of_nodes)
R4 = zeros(num_of_nodes_on_face, num_of_nodes)
for i in eachindex(node_id_on_east)
    R1[i, node_id_on_south[i]] = 1
    R2[i, node_id_on_east[i]] = 1
    R3[i, node_id_on_north[i]] = 1
    R4[i, node_id_on_west[i]] = 1
end
R = (R1, R2, R3, R4)

rst_f1 = rs[:, node_id_on_south]
rst_f2 = rs[:, node_id_on_east]
rst_f3 = rs[:, node_id_on_north]
rst_f4 = rs[:, node_id_on_west]
rst_f = hcat(rst_f1, rst_f2, rst_f3, rst_f4)

f_mask = [collect(((i-1) * num_of_nodes_on_face + 1):(i * num_of_nodes_on_face)) for i in 1:4]
n_rst = normals_face_quad(RefQuadrilateral, f_mask)
SBPLite.RefElemData(degree, quadH, quadHinv, H_face, D, (Qr, Qs), E, Qt, Qt_inv, R, f_mask, n_rst, rs, rs_quad, rst_f)


p1 = scatter(ref.rst[1, :], ref.rst[2, :], label = "cg_triangle")
for (i, (xi, yi)) in enumerate(zip(ref.rst[1, :],  ref.rst[2, :]))
    annotate!(p1, xi, yi, text(string(i), :left, 10))
end
p1