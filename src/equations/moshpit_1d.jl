# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    MoshpitEquations1D(gravity, H0)

Mosh pit equations (MPE) in one space dimension. The equations are given by
```math
\begin{aligned}
  \frac{\partial \rho}{\partial t} + \frac{\partial}{\partial x}(\rho v) &= 0 \\
  \frac{\partial}{\partial t}(\rho v) + \frac{\partial}{\partial x}\left(h v^2 + R \rho \right) &= 0
\end{aligned}
```
The unknown quantities of the MPE are the people density ``\rho`` and the velocity ``v``, 
the parameter ``R`` relates pressure ``p`` to density via the ideal gas law at constant temperature ``p = R \rho``

Based on combining 1D Euler Equations at constant temperature & shallow water equations
- Randall J. LeVeque (2002)
  Finite Volume Methods for Hyperbolic Problems
  [DOI: 10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253)
"""
struct MoshpitEquations1D{RealT<:Real} <: AbstractMoshpitEquations{1, 2}
  R::RealT # parameter in pressure equation
end

function MoshpitEquations1D(R=1)
  MoshpitEquations1D(R)
end


have_nonconservative_terms(::MoshpitEquations1D) = False()
varnames(::typeof(cons2cons), ::MoshpitEquations1D) = ("rho", "rho_v")
varnames(::typeof(cons2prim), ::MoshpitEquations1D) = ("rho", "v")


# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::MoshpitEquations1D)
  rho, rho_v = u
  v = velocity(u, equations)

  p = equations.R * rho

  f1 = rho_v
  f2 = rho_v * v + p

  return SVector(f1, f2)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::MoshpitEquations1D)
  # Get the velocity quantities
  v_ll = velocity(u_ll, equations)
  v_rr = velocity(u_rr, equations)

  return max(abs(v_ll), abs(v_rr)) + sqrt(equations.R) # TODO: sqrt of R could be precomputed and stored
end

@inline function max_abs_speeds(u, equations::MoshpitEquations1D)
  # Get the velocity quantities
  v = velocity(u, equations)

  return abs(v) + sqrt(equations.R) # TODO: sqrt of R could be precomputed and stored
end


# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::MoshpitEquations1D)
  v_ll = velocity(u_ll, equations)
  v_rr = velocity(u_rr, equations)

  # Note: Standard HLL, no refined choice by Einfeldt with Roe-Matrix
  λ_min = min(v_ll, v_rr) - sqrt(equations.R)
  λ_max = max(v_ll, v_rr) + sqrt(equations.R)

  return λ_min, λ_max
end


# Helper function to extract the velocity vector from the conservative variables
@inline function velocity(u, equations::MoshpitEquations1D)
  rho, rho_v = u

  v = rho_v / rho

  return v
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::MoshpitEquations1D)
  rho, _ = u

  v = velocity(u, equations)
  return SVector(rho, v)
end


# Convert conservative variables to entropy
# Note, only the first three are the entropy variables, the fourth entry still
# just carries the bottom topography values for convenience
@inline function cons2entropy(u, equations::MoshpitEquations1D)
  rho, rho_v = u

  v = velocity(u, equations)

  w1 = - 0.5 * v^2*rho
  w2 = v

  return SVector(w1, w2)
end


# Convert entropy variables to conservative
@inline function entropy2cons(w, equations::MoshpitEquations1D)
  w1, w2 = w

  rho   = w1 * (-2 /w2^2)
  rho_v = rho * w2
  return SVector(rho, rho_v)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::MoshpitEquations1D)
  rho, v = prim

  return SVector(rho, rho * v)
end


@inline function pressure(u, equations::MoshpitEquations1D)
  rho, _ = u
  p = equations.R * rho
  return p
end


# CARE: Entropy function for the mosh-pit equations is assumed to be the total energy 
@inline entropy(cons, equations::MoshpitEquations1D) = energy_total(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, equations::MoshpitEquations1D)
  rho, rho_v = cons

  # CARE: Use energy of ideal gas with constant temperature, where we normalize such that c_v T = 0
  e = (h_v^2) / (2 * h)
  return e
end


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::MoshpitEquations1D)
  rho, rho_v = u
  return (rho_v^2) / (2 * rho)
end


# Calculate potential energy for a conservative state `cons`
@inline function energy_internal(cons, equations::MoshpitEquations1D)
  return zero(cons) # TODO: Not really consistent with pressure equation that assumes a nonzero T (=> c_v = 0)
end

end # @muladd
