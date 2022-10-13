using LinearAlgebra

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    LinearizedEulerEquations1D_PosDepMach(rho_0, v_0, c_0)

The one-dimensional linearized Euler equations where `rho_0`, `v_0` and `c_0` are
the uniform mean density, mean velocity in x-direction, and
mean speed of sound, respectively.

This is a testcase for PERK to study the interaction of a few stage scheme at low mach numbers
and a many stage scheme for high mach numbers.
"""

# Now 4 variables because v_0 changes now with position x.
# See https://github.com/trixi-framework/Trixi.jl/issues/358 and
# https://trixi-framework.github.io/Trixi.jl/stable/tutorials/adding_nonconservative_equation/, for instance.
mutable struct LinearizedEulerEquations1D_PosDepMach{RealT<:Real} <: LinearizedEulerEquations{1, 4}
  rho_0::RealT
  # Now we have two background speeds
  v_0Low::RealT
  v_0High::RealT
  c_0::RealT

  #=
  EigVals::Vector{RealT}
  EigVecMat::Matrix{RealT}
  EigVecMatInv::Matrix{RealT}
  =#

  function LinearizedEulerEquations1D_PosDepMach(rho_0, v_0Low, v_0High, c_0)
    if rho_0 < 0
      throw(ArgumentError("rho_0 must be non-negative"))
    end
    if c_0 < 0
      throw(ArgumentError("c_0 must be non-negative"))
    end

    newLinEuler1D = new{typeof(rho_0)}(rho_0, v_0Low, v_0High, c_0)

    #=
    # Compute analytical solution (numerically)
    EigDecomp = eigen([v_0 rho_0 0; 0 v_0 1.0/rho_0; 0 c_0^2*rho_0 v_0])
    newLinEuler1D.EigVals   = EigDecomp.values
    newLinEuler1D.EigVecMat = EigDecomp.vectors

    # Known solution
    #newLinEuler1D.EigVals   = [v_0 - c_0; v_0; v_0 + c_0]
    #newLinEuler1D.EigVecMat = [-rho_0/c_0 1 rho_0/c_0; 1 0 1; -rho_0*c_0 0 rho_0*c_0]

    newLinEuler1D.EigVecMatInv = inv(newLinEuler1D.EigVecMat)
    =#

    return newLinEuler1D
  end
end

function LinearizedEulerEquations1D_PosDepMach(; rho_0, v_0Low, v_0High, c_0)
  return LinearizedEulerEquations1D_PosDepMach(rho_0, v_0Low, v_0High, c_0)
end


varnames(::typeof(cons2cons), ::LinearizedEulerEquations1D_PosDepMach) = ("rho_prime", "v_prime", "p_prime", "v0_of_x")
varnames(::typeof(cons2prim), ::LinearizedEulerEquations1D_PosDepMach) = ("rho_prime", "v_prime", "p_prime", "v0_of_x")


@inline function flux(u, orientation::Integer, equations::LinearizedEulerEquations1D_PosDepMach)
  (; rho_0, v_0Low, v_0High, c_0) = equations
  rho_prime, v_prime, p_prime, v0_of_x = u # "_prime" = ()' means fluctuation

  f1 = v0_of_x * rho_prime + rho_0 * v_prime
  f2 = v0_of_x * v_prime + p_prime / rho_0
  f3 = c_0^2 * rho_0 * v_prime + v0_of_x * p_prime # For an ideal gas it holds: c^2 * rho(_0) = gamma * p(_0)

  return SVector(f1, f2, f3, 0)
end


# Location dependent v_0
@inline have_constant_speed(::LinearizedEulerEquations1D_PosDepMach) = Val(false) 

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::LinearizedEulerEquations1D_PosDepMach)
  (; c_0) = equations
  return max(abs(u_ll[4]), abs(u_rr[4])) + c_0 # Maximum eigenvalue in absolute sense
end

@inline function max_abs_speeds(u, equations::LinearizedEulerEquations1D_PosDepMach)
  (; c_0) = equations

  return abs(u[4]) + c_0
end

# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer, equations::LinearizedEulerEquations1D_PosDepMach)
  (; c_0) = equations

  # CARE: No guarantee, basically copied from compressible case
  λ_min = min(u_ll[4], u_rr[4]) - c_0
  λ_max = max(u_ll[4], u_rr[4]) + c_0

  return λ_min, λ_max
end


# Convert conservative variables to primitive
@inline cons2prim(u, equations::LinearizedEulerEquations1D_PosDepMach) = cons2cons(u, equations)
@inline cons2entropy(u, ::LinearizedEulerEquations1D_PosDepMach) = u

function initial_condition_entropy_wave(x, t, equations::LinearizedEulerEquations1D_PosDepMach)
  # Parameters
  alpha = 1.0
  beta  = 250.0
  center = 0.5

  rho_prime = alpha * exp(-beta * (x[1] - center)^2)
  v_prime   = 0.0
  p_prime   = 0.0

  v0_of_x   = x[1] < center/2 || x[1] > 1.5*center ? equations.v_0Low : equations.v_0Low + 
                (equations.v_0High - equations.v_0Low) * exp(-(x[1] - center)^2/0.01)
  return SVector(rho_prime, v_prime, p_prime, v0_of_x)
end

function initial_condition_acoustic_wave(x, t, equations::LinearizedEulerEquations1D_PosDepMach)
  # Parameters
  alpha     = 1.0
  beta      = 250.0
  center    = 0.5
  Direction = 1 # Intended to be either -1 or +1

  Gaussian = alpha * exp(-beta * (x[1] - center)^2)

  rho_prime = Direction * equations.rho_0 * Gaussian / equations.c_0
  v_prime   = Gaussian
  p_prime   = Direction * equations.rho_0 * Gaussian * equations.c_0
  v0_of_x   = x[1] < center/2 || x[1] > 1.5*center ? equations.v_0Low : equations.v_0Low + 
                (equations.v_0High - equations.v_0Low) * exp(-(x[1] - center)^2/0.01)

  return SVector(rho_prime, v_prime, p_prime, v0_of_x)
end

function initial_condition_rest(x, t, equations::LinearizedEulerEquations1D_PosDepMach)
  center = 0.5

  return SVector(0.0, 0.0, 0.0, 
                 x[1] < center/2 || x[1] > 1.5*center ? equations.v_0Low : equations.v_0Low + 
                  (equations.v_0High - equations.v_0Low) * exp(-(x[1] - center)^2/0.01))
end

end # muladd
  
