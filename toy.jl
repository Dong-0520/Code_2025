function RHS(ρk, ρv, uk, uv, pk, pv, γ)
    # γ  = 1.4
    Ek = 0.5*ρk*uk^2 + pk/(γ-1)
    Ev = 0.5*ρv*uv^2 + pv/(γ-1)
    @assert Ek > Ev
    @assert ρk > ρv
    @assert pk > pv
    @assert uk > uv
    k_contrinution =  ρk^2 * uk^3 - 2 * ρk * uk * Ek + ρk * uk^2 * ρv * uv - 2 * ρk * uk * ρv * uv^2 - 2 * ρk * uk * pv + 2 * ρk * uv * (Ev + pv)
    v_contrinution = -ρv^2 * uv^3 + 2 * ρv * uv * Ev - ρv * uv^2 * ρk * uk + 2 * ρv * uv * ρk * uk^2 + 2 * ρv * uv * pk - 2 * ρv * uk * (Ek + pk)
    return k_contrinution + v_contrinution
end



function RHS2(ρk, ρv, uk, uv, pk, pv, γ)
    Ek = 0.5*ρk*uk^2 + pk/(γ-1)
    Ev = 0.5*ρv*uv^2 + pv/(γ-1)
    @assert Ek > Ev
    @assert ρk > ρv
    @assert pk > pv
    @assert uk > uv
    # rhok uk^3 - - rhov uv^3 + 2（uv Ev - uk Ek) + (uk^2 rhov uv - uv^2 rhok uk) + 2 ( uv rhok uk^2 - uk rhov uv^2) + 2 (uv pk - uk pv) + 2 (uv Ev + uv pv - ukEk - ukpk)

    return ρk * uk^3 - ρv * uv^3 + 2 * (uv * Ev - uk * Ek) + (uk^2 * ρv * uv - uv^2 * ρk * uk) + 2 * (uv * ρk * uk^2 - uk * ρv * uv^2) + 2 * (uv * pk - uk * pv) + 2 * (uv * Ev + uv * pv - uk * Ek - uk * pk)

end


for i in 1:1000000

    γ = rand(1.1:0.01:2.0)
    γr = (γ - 1)/(γ + 1)

    pk = rand(40.0:0.01:80.0)
    pv = pk * (0.3 + 0.4 * rand())  # pv will be 30-70% of pk

    pr = pk/pv
    ρv = rand(0.0000001:0.01:10.0)
    ρk = (pr + γr)/((pr * γr) + 1) * ρv
    
    uk = rand(-10.0:0.01:40.0)
    if uk > 0
        uv = rand(0.0000001:0.01:10.0)  # uv will be 30-70% of uk
    else
        uv = uk - (0.3 + 0.4 * rand()) * abs(uk)  # uv will be 30-70% of uk
    end
   
    
    if RHS(ρk, ρv, uk, uv, pk, pv, γ) > 0
        println("Found a counter example")
        println("ρk = $ρk")
        println("ρv = $ρv")
        println("uk = $uk")
        println("uv = $uv")
        println("pk = $pk")
        println("pv = $pv")
        break
    end
end


function uv_formula(uk, pk, pv, rhok; γ = 1.4)
    AL = 2 / ((γ + 1) * rhok)
    BL = ((γ - 1) / (γ + 1)) * pk
    fL = (pv - pk) * sqrt(AL / (pv + BL))
    return uk - fL
end

uv_formula(1, 0.01, 46.095, 1)

function s1(uk, pk, pv, rhok; γ = 1.4)
    aL = sqrt(γ*pk/rhok)
    a = pv / pk
    return uk - aL * sqrt( (γ+1)*a/(2*γ) + (γ-1)/(2*γ) )
end

function left_entropy(pk, pv, rhok, rhov, uk, uv; γ = 1.4)

    b = rhov / rhok
    a = pv / pk
    @assert b < a
    @assert b > 1
    sk = log(pk / (rhok^γ))
    sv = log(pv / (rhov^γ))
    aL = sqrt(γ*pk/rhok)
    s1 = uk - aL * sqrt( (γ+1)*a/(2*γ) + (γ-1)/(2*γ) )
    # println(s1)
    if s1 < 0
        println("1 wave is propagating to the left")
    else
        println("1 wave is propagating to the right")
    end
    print("shock speed: ", s1, "\n")
    uv_test = uv_formula(uk, pk, pv, rhok)
    # println(uv_test, " ", uv, " \n")
    @assert isapprox(uv_test, uv, atol=1.0e-3)

    return rhov^2 * uv * sv - rhok^2 * uk * sk + s1 * rhov * sv - s1 * rhok * sk
    
end

function math_entropy(rho, physical_s; γ = 1.4)
    return - rho * physical_s / (γ - 1)
end

function entropy_flux(u, math_s)
    return u * math_s
end

function physical_entropy(p, rho; γ = 1.4)
    return log(p / (rho^γ))
end
#----------------------- Test 4 -------------------------------------
γ = 1.4
uk = 0.0
rhok = 1.0
pk = 0.01
uv = -6.19633
pv = 46.095
rhov = 5.99242

physical_entropy_k = physical_entropy(pk, rhok) + 6
physical_entropy_v = physical_entropy(pv, rhov) + 6

math_entropy_k = math_entropy(rhok, physical_entropy_k)
math_entropy_v = math_entropy(rhov, physical_entropy_v)

entropy_flux_k = entropy_flux(uk, math_entropy_k)
entropy_flux_v = entropy_flux(uv, math_entropy_v)

-(entropy_flux_k - entropy_flux_v) + s1(uk, pk, pv, rhok) * ( math_entropy_k - math_entropy_v )
#----------------------- Test 5 -------------------------------------
# left
uk = 19.5975
pk = 460.894
rhok = 5.99924
uv = 8.68975
pv = 1691.64
rhov = 14.2823

physical_entropy_k = physical_entropy(pk, rhok)
physical_entropy_v = physical_entropy(pv, rhov) 

math_entropy_k = math_entropy(rhok, physical_entropy_k)
math_entropy_v = math_entropy(rhov, physical_entropy_v)

entropy_flux_k = entropy_flux(uk, math_entropy_k)
entropy_flux_v = entropy_flux(uv, math_entropy_v)

-(entropy_flux_k - entropy_flux_v) + s1(uk, pk, pv, rhok) * ( math_entropy_k - math_entropy_v )
# right