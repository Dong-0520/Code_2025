include("../src/cg_triangle.jl")
import Main.cg_triangle as cgt
using Test
using LinearAlgebra

#-- dont yet know how to get this to work, i.e., show a plot once I am finished
#ENV["MPLBACKEND"] = "Agg"
#using PyPlot
using Plots

@testset "testing accuracy of triangle omega operators" begin
    println("====================")
    for p = 1:4
        x, y, P, Qx, Qy, Ex, Ey = cgt.traingle_omega(p)
        Dx = inv(P)*Qx
        Dy = inv(P)*Qy
    #plot(x, y,title="nodal distribution", label=["sin(x)" "cos(x)"], linewidth=3, linestyle = :dot)
    #title("nodal distribution for p = ",p)
        for r = 0:p
            for j = 0:r
                i = r-j
                u = (x.^i).*(y.^j)
                dudx = (i.*x.^max(0,i-1)).*(y.^j)
                dudy = (x.^i).*(j.*y.^max(0,j-1))
                ex = maximum(broadcast(abs,Dx*u-dudx))
                ey = maximum(broadcast(abs,Dy*u-dudy))
                @test maximum(ex)<= 10e-14
                @test maximum(ey)<= 10e-14
                #println("p = ",p," i = ",i," r = ",r,"error:",ex," ",ey,"\n")
            end
        end
    end

end


@testset "testing accuracy of diagonal E operators from Jason Hicken as generated by Jesse Chan's code" begin
    println("====================")
    for p = 1:4
        x, y, P, Qx, Qy, Ex, Ey = cgt.triangle_from_Jesse(p)
        #xi = -x
        #eta = -y
        #plot([x,xi], [y,eta],title="nodal distribution", label=["sin(x)" "cos(x)"], linewidth=3, linestyle = :dot)   
        Dx = inv(P)*Qx
        Dy = inv(P)*Qy
        for r = 0:p
            for j = 0:r
                i = r-j
                u = (x.^i).*(y.^j)
                dudx = (i.*x.^max(0,i-1)).*(y.^j)
                dudy = (x.^i).*(j.*y.^max(0,j-1))
                ex = maximum(broadcast(abs,Dx*u-dudx))
                ey = maximum(broadcast(abs,Dy*u-dudy))
                @test maximum(ex)<= 10e-13
                @test maximum(ey)<= 10e-13
                println("p = ",p," i = ",i," r = ",r,"error:",ex," ",ey,"\n")
            end
        end
    end
    #plot(x, y, color="red", linewidth=2.0, linestyle="*")
    #title("nodal distribution for p = ",p)
end

@testset "accuracy test of quad from triangle operaotrs" begin
    println("====================")
    for p = 1:4
        xt, yt, Pt, Qxt, Qyt, Ext, Eyt = cgt.triangle_from_Jesse(p)
        n = size(xt)[1]
        x, y, P, Qx, Qy, Ex, Ey = cgt.quad_from_triangle_SBP(n,xt,yt,Pt,Qxt,Qyt,Ext,Eyt)
        Dx = inv(P)*Qx
        Dy = inv(P)*Qy
        eEx = maximum(broadcast(abs,Ex-(Qx+transpose(Qx))))
        eEy = maximum(broadcast(abs,Ey-(Qy+transpose(Qy))))
        @test maximum(eEx)<= 10e-14
        @test maximum(eEy)<= 10e-14
        for r = 0:p
            for j = 0:r
                i = r-j
                u = (x.^i).*(y.^j)
                dudx = (i.*x.^max(0,i-1)).*(y.^j)
                dudy = (x.^i).*(j.*y.^max(0,j-1))
                ex = maximum(broadcast(abs,Dx*u-dudx))
                ey = maximum(broadcast(abs,Dy*u-dudy))
                @test maximum(ex)<= 10e-14
                @test maximum(ey)<= 10e-14
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
        x, y, P, Qx, Qy, Ex, Ey = cgt.global_first_derivative_operators(xL,xR,yB,yT,Nx,Ny,p)
        Dx = inv(P)*Qx
        Dy = inv(P)*Qy
        eEx = maximum(broadcast(abs,Ex-(Qx+transpose(Qx))))
        eEy = maximum(broadcast(abs,Ey-(Qy+transpose(Qy))))
        @test maximum(eEx)<= 10e-13
        @test maximum(eEy)<= 10e-13
        for r = 0:p
            for j = 0:r
                i = r-j
                u = (x.^i).*(y.^j)
                dudx = (i.*x.^max(0,i-1)).*(y.^j)
                dudy = (x.^i).*(j.*y.^max(0,j-1))
                ex = maximum(broadcast(abs,Dx*u-dudx))
                ey = maximum(broadcast(abs,Dy*u-dudy))
                @test maximum(ex)<= 10e-11
                @test maximum(ey)<= 10e-11
                #println("p = ",p," i = ",i," r = ",r," error:",ex," --- ",ey,"\n")
            end
        end
    end
end