# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

#TODO: Probably needs a name change
mutable struct CompressibleEulerPoissonMultiIonEquations1D{NVARS, NCOMP, RealT <: Real,} <:
               AbstractCompressibleEulerPoissonMultiIonEquations{1, NVARS, NCOMP}

    gammas::SVector{NCOMP, RealT} # Heat capacity ratios
    inv_gammas_minus_one::SVector{NCOMP, RealT} # = inv(gamma - 1)
    epsilon::RealT # m_e / m_i
    scaled_debye_length::RealT

    # Inner Constructor
    function CompressibleEulerPoissonMultiIonEquations1D(gammas::SVector{NCOMP, RealT}, epsilon::RealT,
                                                            scaled_debye_length::RealT) where {NCOMP,
                                                                                                    RealT <:
                                                                                                    Real
                                                                                                    }
        NCOMP >= 1 ||
            throw(DimensionMismatch("`gammas` and `charge_to_mass` must contain at least one value"))

        # Have this in to prevent system with multi-ion that I haven't currently figure out the flux yet
        NCOMP > 2
            throw(ArgumentError("Currently, multi-ion species system is not available yet"))

        # Precompute inverse gamma - 1
        inv_gammas_minus_one = SVector{NCOMP, RealT}(inv.(gammas .- 1))

        NVARS = 3 * NCOMP # We count electron as one component

        return new{NVARS, NCOMP}(gammas,inv_gammas_minus_one, epsilon, scaled_debye_length)
    end
end

# Outer Constructor Delegating to Inner Constructor
function CompressibleEulerPoissonMultiIonEquations1D(; gammas, epsilon, scaled_debye_length)
    # Promote input types
    _gammas = promote(gammas...)
    RealT = eltype(_gammas)
    _epsilon = convert(RealT, epsilon)
    _scaled_debye_length = convert(RealT, scaled_debye_length)

    return CompressibleEulerPoissonMultiIonEquations1D(__gammas,
                                                _epsilon, _scaled_debye_length)
end

@inline function Base.real(::CompressibleEulerPoissonMultiIonEquations1D{NVARS, NCOMP, RealT}) where {
                                                                                               NVARS,
                                                                                               NCOMP,
                                                                                               RealT <:
                                                                                               Real
                                                                                               }
    RealT
end

function varnames(::typeof(cons2cons), equations::CompressibleEulerPoissonMultiIonEquations1D)
    cons = ("rho_electron", "rho_v1_electron", "rho_e_electron")
    for i in eachcomponent(equations)
        i == 1 && continue
            cons = (cons...,
                    tuple("rho_" * string(i), "rho_v1_" * string(i),
                          "rho_e_" * string(i))...) #TODO: maybe display i-1?
    end

    return cons
end

function varnames(::typeof(cons2prim), equations::CompressibleEulerPoissonMultiIonEquations1D)
    prim = ("rho_electron", "v1_electron", "p_electron")
    for i in eachcomponent(equations)
        i == 1 && continue
        prim = (prim...,
                tuple("rho_" * string(i), "v1_" * string(i), "p_" * string(i))...) #TODO: maybe display i-1?
    end
    return prim
end

"""
    initial_condition_constant(x, t, equations::CompressibleEulerPoissonMultiIonEquations1D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t,
                                    equations::CompressibleEulerPoissonMultiIonEquations1D)
    cons = zero(MVector{nvariables(equations), eltype(x)}) # Dummy initial condition for sanity check

    rho = 0.1
    rho_v1 = 1
    rho_e = 10

    for k in eachcomponent(equations)
        set_component!(cons, k, rho, rho_v1, rho_e, equations)
    end

    return SVector(cons)
end

"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerPoissonMultiIonEquations1D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t,
                                            equations::CompressibleEulerPoissonMultiIonEquations1D)
    RealT = eltype(x)
    cons = zero(MVector{nvariables(equations), RealT}) # Dummy convergence test for sanity check

    c = 2
    A = convert(RealT, 0.1)
    L = 2
    f = 1.0f0 / L
    ω = 2 * convert(RealT, pi) * f
    ini = c + A * sin(ω * (x[1] - t))

    rho = ini
    rho_v1 = ini
    rho_e = ini^2

    for k in eachcomponent(equations)
        set_component!(cons, k, rho, rho_v1, rho_e, equations)
    end

    return SVector(cons)
end

"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerPoissonMultiIonEquations1D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t,
                                               equations::CompressibleEulerPoissonMultiIonEquations1D)
    
    # Same settings as in `initial_condition`
    RealT = eltype(u)
    cons = zero(MVector{nvariables(equations), RealT})

    c = 2
    A = convert(RealT, 0.1)
    L = 2
    f = 1.0f0 / L
    ω = 2 * convert(RealT, pi) * f

    x1, = x

    si, co = sincos(ω * (x1 - t))

    for k in eachcomponent(equations)
        gamma = equations.gammas[k]
        rho = c + A * si
        rho_x = ω * A * co

        # Note that d/dt rho = -d/dx rho.
        # This yields du2 = du3 = d/dx p (derivative of pressure).
        # Other terms vanish because of v = 1.
        du1 = 0
        du2 = rho_x * (2 * rho - 0.5f0) * (gamma - 1)
        du3 = du2
        set_component!(cons, k, du1, du2, du3, equations)
    end

    return SVector(cons)
end

"""
    flux(u, orientation::Integer, equations::CompressibleEulerPoissonMultiIonEquations1D)

Calculate the flux for the multiion system. The flux is calculated for each ion species separately.
"""
@inline function flux(u, orientation::Integer,
                      equations::CompressibleEulerPoissonMultiIonEquations1D)
    f = zero(MVector{nvariables(equations), eltype(u)})
    @unpack epsilon = equations

    for k in eachcomponent(equations)
        rho, rho_v1, rho_e = get_component(k, u, equations)
        rho_inv = 1 / rho
        v1 = rho_v1 * rho_inv

        gamma = equations.gammas[k]

        p = pressure(k, rho_e, rho_v1, v1, gamma, equations)
        f1 = rho_v1
        f2 = rho_v1 * v1 + p
        f3 = (rho_e + p) * v1

        set_component!(f, k, f1, f2, f3, equations)
    end

    return SVector(f)
end

"""
    get_component(k, u, equations::CompressibleEulerPoissonMultiIonEquations1D)

Get the variables of component (ion species) `k`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function get_component(k, u, equations::CompressibleEulerPoissonMultiIonEquations1D)
    return SVector(u[(k - 1) * 3 + 1],
                   u[(k - 1) * 3 + 2],
                   u[(k - 1) * 3 + 3])
end

"""
    set_component!(u, k, u1, u2, u3,
                   equations::CompressibleEulerPoissonMultiIonEquations1D)

Set the variables (`u1` to `u3`) of component (ion species) `k`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function set_component!(u, k, u1, u2, u3,
                                equations::CompressibleEulerPoissonMultiIonEquations1D)
    u[(k - 1) * 3 + 1] = u1
    u[(k - 1) * 3 + 2] = u2
    u[(k - 1) * 3 + 3] = u3

    return u
end

# Calculate estimates for maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::CompressibleEulerPoissonMultiIonEquations1D)

    # Calculate velocities
    v_mag_ll = zero(eltype(u_ll))
    v_mag_rr = zero(eltype(u_rr))
    c_ll = zero(eltype(u_ll))
    c_rr = zero(eltype(u_rr))

    for k in eachcomponent(equations)
        rho_ll, rho_v1_ll, rho_e_ll = get_component(k, u_ll, equations)
        rho_rr, rho_v1_rr, rho_e_rr = get_component(k, u_rr, equations)
        gamma = equations.gammas[k]

        # Calculate primitive variables and speed of sound
        v1_ll = rho_v1_ll / rho_ll
        p_ll = (gamma - 1) * (rho_e_ll - 0.5f0 * rho_ll * abs(v1_ll)^2)
        v_mag_ll = max(v_mag_ll, abs(v1_ll))
        c_ll = max(c_ll, sqrt(gamma * p_ll / rho_ll))

        v1_rr = rho_v1_rr / rho_rr
        p_rr = (gamma - 1) * (rho_e_rr - 0.5f0 * rho_rr * abs(v1_rr)^2)
        v_mag_rr = max(v_mag_rr, abs(v1_rr))
        c_rr = max(c_rr, sqrt(gamma * p_rr / rho_rr))
    end

    λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end

@inline function max_abs_speeds(u,
                                equations::CompressibleEulerPoissonMultiIonEquations1D)
    prim = cons2prim(u, equations)
    v1_max = zero(eltype(u))
    for k in eachcomponent(equations)
        rho, v1, p = get_component(k, prim, equations)
        gamma = equations.gammas[k]
        c = sqrt(gamma * p / rho)
        v1_max = max(v1_max, abs(v1) + c)
    end
    return (v1_max,)
end

# Calculate estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::CompressibleEulerPoissonMultiIonEquations1D)
    prim_ll = cons2prim(u_ll, equations)
    prim_rr = cons2prim(u_rr, equations)

    λ_min = oftype(u_ll[1], Inf)
    λ_max = oftype(u_ll[1], -Inf)

    for k in eachcomponent(equations)
        rho_ll, v1_ll, p_ll = get_component(k, prim_ll, equations)
        rho_rr, v1_rr, p_rr = get_component(k, prim_rr, equations)
        gamma = equations.gammas[k]

        λ_min = min(λ_min, v1_ll - sqrt(gamma * p_ll / rho_ll))
        λ_max = max(λ_max, v1_rr + sqrt(gamma * p_rr / rho_rr))
    end

    #Assert that λ_min and λ_max are not Inf
    @assert !isinf(λ_min) "λ_min is Inf"
    @assert !isinf(λ_max) "λ_max is Inf"

    return λ_min, λ_max
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::CompressibleEulerPoissonMultiIonEquations1D)
    prim_ll = cons2prim(u_ll, equations)
    prim_rr = cons2prim(u_rr, equations)

    λ_min = oftype(u_ll[1], Inf)
    λ_max = oftype(u_ll[1], -Inf)

    for k in eachcomponent(equations)
        rho_ll, v1_ll, p_ll = get_component(k, prim_ll, equations)
        rho_rr, v1_rr, p_rr = get_component(k, prim_rr, equations)
        gamma = equations.gammas[k]

        c_ll = sqrt(gamma * p_ll / rho_ll)
        c_rr = sqrt(gamma * p_rr / rho_rr)

        λ_min = min(λ_min, min(v1_ll - c_ll, v1_rr - c_rr))
        λ_max = max(λ_max, max(v1_ll + c_ll, v1_rr + c_rr))
    end

    return λ_min, λ_max
end

@inline function cons2prim(u, equations::CompressibleEulerPoissonMultiIonEquations1D)
    prim = zero(MVector{nvariables(equations), eltype(u)})
    for k in eachcomponent(equations)
        rho, rho_v1, rho_e = get_component(k, u, equations)
        v1 = rho_v1 / rho
        gamma = equations.gammas[k]
        p = pressure(k, rho_e, rho_v1, v1, gamma, equations)
        set_component!(prim, k, rho, v1, p, equations)
    end

    return SVector(prim)
end

#TODO: Does this still work for the electron?
# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerPoissonMultiIonEquations1D)
    w = zero(MVector{nvariables(equations), eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, rho_e = get_component(k, u, equations)
        gamma = equations.gammas[k]
        inv_gamma_minus_one = equations.inv_gammas_minus_one[k]
        v1 = rho_v1 / rho
        v_square = v1^2
        p = pressure(k, rho_e, rho_v1, v1, gamma, equations)
        s = log(p) - gamma * log(rho)
        rho_p = rho / p

        w1 = (gamma - s) * inv_gamma_minus_one -
             0.5 * rho_p * v_square
        w2 = rho_p * v1
        w3 = -rho_p
        set_component!(w, k, w1, w2, w3, equations)
    end

    return SVector(w)
end

@inline function entropy2cons(w, equations::CompressibleEulerPoissonMultiIonEquations1D)
    cons = zero(MVector{nvariables(equations), eltype(w)})
    # See Hughes, Franca, Mallet (1986) A new finite element formulation for CFD
    # [DOI: 10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)

    # convert to entropy `-rho * s` used by Hughes, France, Mallet (1986)
    # instead of `-rho * s / (gamma - 1)`
    for k in eachcomponent(equations)
        w_k = get_component(k, w, equations)
        gamma = equations.gammas[k]
        inv_gamma_minus_one = equations.inv_gammas_minus_one[k]
        V1, V2, V5 = w_k .* (gamma - 1)

        # specific entropy, eq. (53)
        s = gamma - V1 + 0.5f0 * (V2^2) / V5

        # eq. (52)
        energy_internal = ((gamma - 1) / (-V5)^gamma)^(inv_gamma_minus_one) *
                          exp(-s * inv_gamma_minus_one)

        # eq. (51)
        rho = -V5 * energy_internal
        rho_v1 = V2 * energy_internal
        rho_e = (1 - 0.5f0 * (V2^2) / V5) * energy_internal
        set_component!(cons, k, rho, rho_v1, rho_e, equations)
    end
    return SVector(cons)
end

@inline function prim2cons(prim, equations::CompressibleEulerPoissonMultiIonEquations1D)
    @unpack epsilon = equations
    cons = zero(MVector{nvariables(equations), eltype(prim)})

    for k in eachcomponent(equations)
        rho, v1, p = get_component(k, prim, equations)
        inv_gamma_minus_one = equations.inv_gammas_minus_one[k]
        rho_v1 = rho * v1
        if k == 1
            rho_e = p * inv_gamma_minus_one + 0.5f0 * (rho_v1 * v1 * epsilon) # electron
        else
            rho_e = p * inv_gamma_minus_one + 0.5f0 * (rho_v1 * v1) # ion
        end
        set_component!(cons, k, rho, rho_v1, rho_e, equations)
    end

    return SVector(cons)
end

@inline function density(u, equations::CompressibleEulerPoissonMultiIonEquations1D)
    num_components = div(nvariables(equations), 3)
    rhos = zero(MVector{num_components, eltype(u)})

    for k in eachcomponent(equations)
        rho, _, _ = get_component(k, u, equations)
        rhos[k] = rho
    end

    return SVector(rhos)
end

@inline function velocity(u, equations::CompressibleEulerPoissonMultiIonEquations1D)
    num_components = div(nvariables(equations), 3)
    velocities = zero(MVector{num_components, eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, _ = get_component(k, u, equations)
        velocities[k] = rho_v1 / rho
    end

    return SVector(velocities)
end

@inline function pressure(u, equations::CompressibleEulerPoissonMultiIonEquations1D)
    num_components = div(nvariables(equations), 3)
    pressures = zero(MVector{num_components, eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, rho_e = get_component(k, u, equations)
        gamma = equations.gammas[k]
        p = pressure(k, rho_e, rho_v1, v1, gamma, equations)
        pressures[k] = p
    end
    return SVector(pressures)
end

# A function to calculate pressure for either electron or pressure
@inline function pressure(k, rho_e, rho_v1, v1, gamma, equations::CompressibleEulerPoissonMultiIonEquations1D)
    @unpack epsilon = equations
    if k == 1
        p = (gamma - 1) * (rho_e - 0.5 * rho_v1 * v1 * epsilon) # electron's pressure
    else
        p = (gamma - 1) * (rho_e - 0.5 * rho_v1 * v1)
    end
    return p
end
end # @muladd
