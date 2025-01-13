struct CompressibleEulerPlasmaEquations1D{RealT <: Real} <:
    AbstractCompressibleEulerEquations{1, 6}
 gamma::RealT               # ratio of specific heats
 inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

 function CompressibleEulerPlasmaEquations1D(gamma)
     γ, inv_gamma_minus_one = promote(gamma, inv(gamma - 1))
     new{typeof(γ)}(γ, inv_gamma_minus_one)
 end
end

function varnames(::typeof(cons2cons), ::CompressibleEulerPlasmaEquations1D)
    ("rho_ion", "rho_v1_ion", "rho_e_ion", "rho_el", "rho_v1_el", "rho_e_el")
end
varnames(::typeof(cons2prim), ::CompressibleEulerPlasmaEquations1D) = ("rho_ion", "v1_ion", "p_ion", "rho_el", "v1_el", "p_el")

"""
    single_species_flux(u, gamma)

Calculate the flux vector for a single species (ion or electron).
"""
@inline function single_species_flux(rho, rho_v1, rho_e, gamma)
    v1 = rho_v1 / rho
    p = (gamma - 1) * (rho_e - 0.5 * rho_v1 * v1)
    return SVector(rho_v1, rho_v1 * v1 + p, (rho_e + p) * v1)
end

"""
    flux(u, orientation::Integer, equations::CompressibleEulerPlasmaEquations1D)

Calculate the flux for the full plasma system with ions and electrons.
"""
@inline function flux(u, orientation::Integer, equations::CompressibleEulerPlasmaEquations1D)
    rho_ion, rho_v1_ion, rho_e_ion, rho_el, rho_v1_el, rho_e_el = u

    f_ion = single_species_flux(rho_ion, rho_v1_ion, rho_e_ion, equations.gamma)

    f_el = single_species_flux(rho_el, rho_v1_el, rho_e_el, equations.gamma)

    return SVector(f_ion[1], f_ion[2], f_ion[3], f_el[1], f_el[2], f_el[3])
end

"""
    initial_condition_constant(x, t, equations::CompressibleEulerPlasmaEquations1D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equations::CompressibleEulerPlasmaEquations1D)
    RealT = eltype(x)
    rho_ion = 1
    rho_el = 1
    
    # Ion and electron velocities
    rho_v1_ion = convert(RealT, 0.01)
    rho_v1_el = convert(RealT, 0.1)
    
    # Ion and electron internal energies
    rho_e_ion = 10.0
    rho_e_el = 10.0

    return SVector(rho_ion, rho_v1_ion, rho_e_ion, rho_el, rho_v1_el, rho_e_el)
end

"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerPlasmaEquations1D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::CompressibleEulerPlasmaEquations1D)
    RealT = eltype(x)
    c = 2
    A = convert(RealT, 0.1)
    L = 2
    f = 1.0f0 / L
    ω = 2 * convert(RealT, pi) * f
    ini = c + A * sin(ω * (x[1] - t))

    rho = ini
    rho_v1 = ini
    rho_e = ini^2

    return SVector(rho, rho_v1, rho_e, rho, rho_v1, rho_e)
end


"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerPlasmaEquations1D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t,
                                               equations::CompressibleEulerPlasmaEquations1D)
    # Same settings as in `initial_condition`
    RealT = eltype(u)
    c = 2
    A = convert(RealT, 0.1)
    L = 2
    f = 1.0f0 / L
    ω = 2 * convert(RealT, pi) * f
    γ = equations.gamma

    x1, = x

    si, co = sincos(ω * (x1 - t))
    rho = c + A * si
    rho_x = ω * A * co

    # Note that d/dt rho = -d/dx rho.
    # This yields du2 = du3 = d/dx p (derivative of pressure).
    # Other terms vanish because of v = 1.
    du1 = 0
    du2 = rho_x * (2 * rho - 0.5f0) * (γ - 1)
    du3 = du2

    return SVector(du1, du2, du3, du1, du2, du3)
end

#TODO: Maybe there is a better way to do this idk
function single_species_flux_hllc(u_ll, u_rr, orientation::Integer,
                   gamma)
    # Calculate primitive variables and speed of sound
    rho_ll, rho_v1_ll, rho_e_ll = u_ll
    rho_rr, rho_v1_rr, rho_e_rr = u_rr

    v1_ll = rho_v1_ll / rho_ll
    e_ll = rho_e_ll / rho_ll
    p_ll = (gamma - 1) * (rho_e_ll - 0.5f0 * rho_ll * v1_ll^2)
    c_ll = sqrt(gamma * p_ll / rho_ll)

    v1_rr = rho_v1_rr / rho_rr
    e_rr = rho_e_rr / rho_rr
    p_rr = (gamma - 1) * (rho_e_rr - 0.5f0 * rho_rr * v1_rr^2)
    c_rr = sqrt(gamma * p_rr / rho_rr)

    # Obtain left and right fluxes
    f_ll = single_species_flux(rho_ll, rho_v1_ll, rho_e_ll, gamma)
    f_rr = single_species_flux(rho_rr, rho_v1_rr, rho_e_rr, gamma)

    # Compute Roe averages
    sqrt_rho_ll = sqrt(rho_ll)
    sqrt_rho_rr = sqrt(rho_rr)
    sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr
    vel_L = v1_ll
    vel_R = v1_rr
    vel_roe = (sqrt_rho_ll * vel_L + sqrt_rho_rr * vel_R) / sum_sqrt_rho
    ekin_roe = 0.5f0 * vel_roe^2
    H_ll = (rho_e_ll + p_ll) / rho_ll
    H_rr = (rho_e_rr + p_rr) / rho_rr
    H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) / sum_sqrt_rho
    c_roe = sqrt((gamma - 1) * (H_roe - ekin_roe))

    Ssl = min(vel_L - c_ll, vel_roe - c_roe)
    Ssr = max(vel_R + c_rr, vel_roe + c_roe)
    sMu_L = Ssl - vel_L
    sMu_R = Ssr - vel_R
    if Ssl >= 0
        f1 = f_ll[1]
        f2 = f_ll[2]
        f3 = f_ll[3]
    elseif Ssr <= 0
        f1 = f_rr[1]
        f2 = f_rr[2]
        f3 = f_rr[3]
    else
        SStar = (p_rr - p_ll + rho_ll * vel_L * sMu_L - rho_rr * vel_R * sMu_R) /
                (rho_ll * sMu_L - rho_rr * sMu_R)
        if Ssl <= 0 <= SStar
            densStar = rho_ll * sMu_L / (Ssl - SStar)
            enerStar = e_ll + (SStar - vel_L) * (SStar + p_ll / (rho_ll * sMu_L))
            UStar1 = densStar
            UStar2 = densStar * SStar
            UStar3 = densStar * enerStar

            f1 = f_ll[1] + Ssl * (UStar1 - rho_ll)
            f2 = f_ll[2] + Ssl * (UStar2 - rho_v1_ll)
            f3 = f_ll[3] + Ssl * (UStar3 - rho_e_ll)
        else
            densStar = rho_rr * sMu_R / (Ssr - SStar)
            enerStar = e_rr + (SStar - vel_R) * (SStar + p_rr / (rho_rr * sMu_R))
            UStar1 = densStar
            UStar2 = densStar * SStar
            UStar3 = densStar * enerStar

            #end
            f1 = f_rr[1] + Ssr * (UStar1 - rho_rr)
            f2 = f_rr[2] + Ssr * (UStar2 - rho_v1_rr)
            f3 = f_rr[3] + Ssr * (UStar3 - rho_e_rr)
        end
    end
    return SVector(f1, f2, f3)
end

"""
    flux_hllc(u_ll, u_rr, orientation, equations::CompressibleEulerPlasmaEquations1D)

Computes the HLLC flux (HLL with Contact) for compressible Euler equations developed by E.F. Toro
[Lecture slides](http://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf)
Signal speeds: [DOI: 10.1137/S1064827593260140](https://doi.org/10.1137/S1064827593260140)
"""
function flux_hllc(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerPlasmaEquations1D)
    flux_hllc_ion = single_species_flux_hllc(u_ll[1:3], u_rr[1:3], orientation, equations.gamma)
    flux_hllc_el = single_species_flux_hllc(u_ll[4:6], u_rr[4:6], orientation, equations.gamma)
    return vcat(flux_hllc_ion, flux_hllc_el)
end

@inline function max_abs_speeds(u, equations::CompressibleEulerPlasmaEquations1D)
    rho_ion, v1_ion, p_ion, rho_el, v1_el, p_el = cons2prim(u, equations)
    c_ion = sqrt(equations.gamma * p_ion / rho_ion)
    c_el = sqrt(equations.gamma * p_el / rho_el)

    return (abs(v1_ion) + c_ion, abs(v1_el) + c_el) # Is this the right way? I've seen multicomponent where (abs(v1) + c,) is returned
end

@inline function single_species_cons2prim(u, gamma)
    rho, rho_v1, rho_e = u
    v1 = rho_v1 / rho
    p = (gamma - 1) * (rho_e - 0.5f0 * rho * v1^2)
    return SVector(rho, v1, p)
end

@inline function cons2prim(u, equations::CompressibleEulerPlasmaEquations1D)
    prim_ion = single_species_cons2prim(u[1:3], equations.gamma)

    prim_el = single_species_cons2prim(u[4:6], equations.gamma)

    return vcat(prim_ion, prim_el)
end

@inline function single_species_cons2entropy(u, gamma, inv_gamma_minus_one)
    rho, rho_v1, rho_e = u

    v1 = rho_v1 / rho
    v_square = v1^2
    p = (gamma - 1) * (rho_e - 0.5f0 * rho * v_square)
    s = log(p) - gamma * log(rho)
    rho_p = rho / p

    w1 = (gamma - s) * inv_gamma_minus_one -
         0.5f0 * rho_p * v_square
    w2 = rho_p * v1
    w3 = -rho_p

    return SVector(w1, w2, w3)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerPlasmaEquations1D)
    w_ion = single_species_cons2entropy(u[1:3], equations.gamma, equations.inv_gamma_minus_one)
    w_el = single_species_cons2entropy(u[4:6], equations.gamma, equations.inv_gamma_minus_one)
    return vcat(w_ion, w_el)
end

@inline function single_species_entropy2cons(w, gamma, inv_gamma_minus_one)
    # See Hughes, Franca, Mallet (1986) A new finite element formulation for CFD
    # [DOI: 10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)

    # convert to entropy `-rho * s` used by Hughes, France, Mallet (1986)
    # instead of `-rho * s / (gamma - 1)`
    V1, V2, V5 = w .* (gamma - 1)

    # specific entropy, eq. (53)
    s = gamma - V1 + 0.5f0 * (V2^2) / V5

    # eq. (52)
    energy_internal = ((gamma - 1) / (-V5)^gamma)^(inv_gamma_minus_one) *
                      exp(-s * inv_gamma_minus_one)

    # eq. (51)
    rho = -V5 * energy_internal
    rho_v1 = V2 * energy_internal
    rho_e = (1 - 0.5f0 * (V2^2) / V5) * energy_internal
    return SVector(rho, rho_v1, rho_e)
end

@inline function entropy2cons(w, equations::CompressibleEulerPlasmaEquations1D)
    u_ion = single_species_entropy2cons(w[1:3], equations.gamma, equations.inv_gamma_minus_one)
    u_el = single_species_entropy2cons(w[4:6], equations.gamma, equations.inv_gamma_minus_one)
    return vcat(u_ion, u_el)
end

# Convert primitive to conservative variables
@inline function single_species_prim2cons(prim, inv_gamma_minus_one)
    rho, v1, p = prim
    rho_v1 = rho * v1
    rho_e = p * inv_gamma_minus_one + 0.5f0 * (rho_v1 * v1)
    return SVector(rho, rho_v1, rho_e)
end

@inline function prim2cons(prim, equations::CompressibleEulerPlasmaEquations1D)
    u_ion = single_species_prim2cons(prim[1:3], equations.inv_gamma_minus_one)
    u_el = single_species_prim2cons(prim[4:6], equations.inv_gamma_minus_one)
    return vcat(u_ion, u_el)
end

@inline function density(u, equations::CompressibleEulerPlasmaEquations1D)
    rho_ion = u[1]
    rho_el = u[4]
    return rho_ion, rho_el
end

@inline function velocity(u, equations::CompressibleEulerPlasmaEquations1D)
    rho_ion = u[1]
    rho_el = u[4]
    v1_ion = u[2] / rho_ion
    v1_el = u[5] / rho_el
    return v1_ion, v1_el
end

@inline function single_species_pressure(u)
    rho, rho_v1, rho_e = u
    p = (equations.gamma - 1) * (rho_e - 0.5f0 * (rho_v1^2) / rho)
    return p
end

@inline function pressure(u, equations::CompressibleEulerPlasmaEquations1D)
    p_ion = single_species_pressure(u[1:3])
    p_el = single_species_pressure(u[4:6])
    return p_ion, p_el
end