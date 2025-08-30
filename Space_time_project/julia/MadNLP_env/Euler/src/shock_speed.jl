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


