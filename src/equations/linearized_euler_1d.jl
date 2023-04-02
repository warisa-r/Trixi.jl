using LinearAlgebra

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    LinearizedEulerEquations1D(rho_0, v_0, c_0)

The one-dimensional linearized Euler equations where `rho_0`, `v_0` and `c_0` are
the uniform mean density, mean velocity in x-direction, and
mean speed of sound, respectively.
"""
mutable struct LinearizedEulerEquations1D{RealT<:Real} <: LinearizedEulerEquations{1, 3}
  rho_0::RealT
  v_0::RealT
  c_0::RealT

  EigVals::Vector{RealT}
  EigVecMat::Matrix{RealT}
  EigVecMatInv::Matrix{RealT}

  function LinearizedEulerEquations1D(rho_0, v_0, c_0)
    if rho_0 < 0
      throw(ArgumentError("rho_0 must be non-negative"))
    end
    if c_0 < 0
      throw(ArgumentError("c_0 must be non-negative"))
    end

    newLinEuler1D = new{typeof(rho_0)}(rho_0, v_0, c_0)

    # Compute analytical solution (numerically)
    EigDecomp = eigen([v_0 rho_0 0; 0 v_0 1.0/rho_0; 0 c_0^2*rho_0 v_0])
    newLinEuler1D.EigVals   = EigDecomp.values
    newLinEuler1D.EigVecMat = EigDecomp.vectors

    # Known solution
    #newLinEuler1D.EigVals   = [v_0 - c_0; v_0; v_0 + c_0]
    #newLinEuler1D.EigVecMat = [-rho_0/c_0 1 rho_0/c_0; 1 0 1; -rho_0*c_0 0 rho_0*c_0]

    newLinEuler1D.EigVecMatInv = inv(newLinEuler1D.EigVecMat)

    return newLinEuler1D
  end
end

function LinearizedEulerEquations1D(; rho_0, v_0, c_0)
  return LinearizedEulerEquations1D(rho_0, v_0, c_0)
end


varnames(::typeof(cons2cons), ::LinearizedEulerEquations1D) = ("rho_prime", "v_prime", "p_prime")
varnames(::typeof(cons2prim), ::LinearizedEulerEquations1D) = ("rho_prime", "v_prime", "p_prime")


@inline function flux(u, orientation::Integer, equations::LinearizedEulerEquations1D)
  (; rho_0, v_0, c_0) = equations
  rho_prime, v_prime, p_prime = u # "_prime" = ()' means fluctuation

  f1 = v_0 * rho_prime + rho_0 * v_prime
  f2 = v_0 * v_prime + p_prime / rho_0
  f3 = c_0^2 * rho_0 * v_prime + v_0 * p_prime # For an ideal gas it holds: c^2 * rho(_0) = gamma * p(_0)

  return SVector(f1, f2, f3)
end


#@inline have_constant_speed(::LinearizedEulerEquations1D) = Val(false) # From Lars Christmann
@inline have_constant_speed(::LinearizedEulerEquations1D) = Val(true) # I see no reason why they should not have constant speed

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::LinearizedEulerEquations1D)
  (; v_0, c_0) = equations
  return abs(v_0) + c_0 # Maximum eigenvalue in absolute sense
end

@inline function max_abs_speeds(equations::LinearizedEulerEquations1D)
  (; v_0, c_0) = equations
  return abs(v_0) + c_0 # Maximum eigenvalue in absolute sense
end

# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer, equations::LinearizedEulerEquations1D)
  (; v_0, c_0) = equations

  # CARE: No guarantee, basically copied from compressible case
  λ_min = v_0 - c_0
  λ_max = v_0 + c_0

  return λ_min, λ_max
end


# Convert conservative variables to primitive
@inline cons2prim(u, equations::LinearizedEulerEquations1D) = cons2cons(u, equations)
@inline cons2entropy(u, ::LinearizedEulerEquations1D) = u

function compute_char_initial_pos(x, t, equations::LinearizedEulerEquations1D)
  return [x; x; x] .- t * equations.EigVals 
end

function compute_primal_sol(CharVars, equations::LinearizedEulerEquations1D)
  return equations.EigVecMat * CharVars
end


# CAVEAT: Copied from 2D case, not sure if physically correct
function initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations1D)
  rho_prime = -cospi(2*t) * sinpi(2*x[1])
  v_prime   = sinpi(2*t) * cospi(2*x[1])
  p_prime   = rho_prime

  return SVector(rho_prime, v_prime, p_prime)
end

function initial_condition_char_vars_convergence_test(x, p::Int, equations::LinearizedEulerEquations1D)
  return transpose(equations.EigVecMatInv[p,:]) * [-1.0 * sinpi.(2*x);
                                                    0.0 * cospi.(2*x);
                                                   -1.0 * sinpi.(2*x)]
end

function initial_condition_entropy_wave(x, t, equations::LinearizedEulerEquations1D)
  # Parameters
  alpha = 1.0
  beta  = 250.0
  center = 0.5

  rho_prime = alpha * exp(-beta * (x[1] - center)^2)
  v_prime   = 0.0
  p_prime   = 0.0

  return SVector(rho_prime, v_prime, p_prime)
end

function initial_condition_char_vars_entropy_wave(x, p::Int, equations::LinearizedEulerEquations1D)
  # Parameters
  alpha = 1.0
  beta  = 250.0
  center = 0.5

  # Equivalent to 
  #W0 = equations.EigVecMatInv * [alpha * exp.(-beta * (x .- center).^2); zeros(1, length(x)); zeros(1, length(x))]
  #return W0[p]

  #return dot(equations.EigVecMatInv[p,:], [alpha * exp.(-beta * (x .- center).^2); zeros(1, length(x)); zeros(1, length(x))])

  return dot(equations.EigVecMatInv[p,:], initial_condition_entropy_wave(x, 0, equations))
end

function initial_condition_acoustic_wave(x, t, equations::LinearizedEulerEquations1D)
  # Parameters
  alpha     = 0.1
  beta      = 50.0
  center    = 0.5
  Direction = 1 # Intended to be either -1 or +1

  Gaussian = alpha * exp(-beta * (x[1] - center)^2)

  rho_prime = Direction * equations.rho_0 * Gaussian / (equations.c_0 * equations.v_0)
  v_prime   = Gaussian
  p_prime   = Direction * equations.rho_0 * Gaussian * (equations.c_0 * equations.v_0)

  return SVector(rho_prime, v_prime, p_prime)
end

function initial_condition_char_vars_acoustic_wave(x, p::Int, equations::LinearizedEulerEquations1D)
  # Parameters
  alpha = 1.0
  beta  = 250.0
  center = 0.5
  Direction = 1 # Intended to be either -1 or +1

  Gaussian = alpha * exp.(-beta * (x .- center).^2)

  return dot(equations.EigVecMatInv[p,:],  [Direction * equations.rho_0 * Gaussian / equations.c_0; 
                                            Gaussian; 
                                            Direction * equations.rho_0 * Gaussian * equations.c_0])
end

function initial_condition_rest(x, t, equations::LinearizedEulerEquations1D)
  return SVector(0.0, 0.0, 0.0)
end

"""
    Boundary (Inlet) Condition taken from "acoustic_perturbation_2d.jl".
Boundary conditions for a solid wall.
"""
function boundary_condition_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                                 equations::LinearizedEulerEquations1D)
  # Boundary state is equal to the inner state except for the velocity. For boundaries
  # in the -x/+x direction, we multiply the velocity (in the x direction by) -1.
  u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3])

  # Calculate boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


end # muladd
  
