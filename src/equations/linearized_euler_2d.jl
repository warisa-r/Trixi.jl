# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    LinearizedEulerEquations2D(rho_0, v1_0, v2_0, c_0)

The two-dimensional linearized Euler equations where `rho_0`, `v1_0`, `v2_0` and `c_0` are
the uniform mean density, mean velocity in x-direction, mean velocity in y-direction and
mean speed of sound, respectively.
"""
struct LinearizedEulerEquations2D{RealT<:Real} <: LinearizedEulerEquations{2, 4}
  rho_0::RealT
  v1_0::RealT
  v2_0::RealT
  c_0::RealT

  function LinearizedEulerEquations2D(rho_0, v1_0, v2_0, c_0)
    if rho_0 < 0
      throw(ArgumentError("rho_0 must be non-negative"))
    elseif c_0 < 0 # FIXME: In my opinion, these are no exclusive and may occur simultaneously
      throw(ArgumentError("c_0 must be non-negative"))
    end

    return new{typeof(rho_0)}(rho_0, v1_0, v2_0, c_0)
  end
end

function LinearizedEulerEquations2D(; rho_0, v1_0, v2_0, c_0)
  return LinearizedEulerEquations2D(rho_0, v1_0, v2_0, c_0)
end


varnames(::typeof(cons2cons), ::LinearizedEulerEquations2D) = ("rho_prime", "v1_prime", "v2_prime", "p_prime")
varnames(::typeof(cons2prim), ::LinearizedEulerEquations2D) = ("rho_prime", "v1_prime", "v2_prime", "p_prime")


@inline function flux(u, orientation::Integer, equations::LinearizedEulerEquations2D)
  (; rho_0, v1_0, v2_0, c_0) = equations
  rho_prime, v1_prime, v2_prime, p_prime = u
  if orientation == 1
    f1 = v1_0 * rho_prime + rho_0 * v1_prime
    f2 = v1_0 * v1_prime + p_prime / rho_0
    f3 = v1_0 * v2_prime
    f4 = v1_0 * p_prime + c_0^2 * rho_0 * v1_prime
  else
    f1 = v2_0 * rho_prime + rho_0 * v2_prime
    f2 = v2_0 * v1_prime
    f3 = v2_0 * v2_prime + p_prime / rho_0
    f4 = v2_0 * p_prime + c_0^2 * rho_0 * v2_prime
  end

  return SVector(f1, f2, f3, f4)
end

# Analogous to flux in "compressible_euler_2d"
@inline function flux(u, normal_direction::AbstractVector, equations::LinearizedEulerEquations2D)
  (; rho_0, v1_0, v2_0, c_0) = equations
  rho_prime, v1_prime, v2_prime, p_prime = u

  v_normal = v1_0 * normal_direction[1] + v2_0 * normal_direction[2]
  f1 = v_normal * rho_prime + rho_0 * v1_prime
  f2 = v_normal * v1_prime + p_prime / rho_0
  f3 = v_normal * v2_prime
  f4 = v_normal * p_prime + c_0^2 * rho_0 * v1_prime

  return SVector(f1, f2, f3, f4)
end


@inline have_constant_speed(::LinearizedEulerEquations2D) = Val(true)

@inline function max_abs_speeds(equations::LinearizedEulerEquations2D)
  (; v1_0, v2_0, c_0) = equations
  return abs(v1_0) + c_0, abs(v2_0) + c_0
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::LinearizedEulerEquations2D)
  (; v1_0, v2_0, c_0) = equations
  if orientation == 1
    return abs(v1_0) + c_0
  else # orientation == 2
    return abs(v2_0) + c_0
  end
end


# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquations2D)

  if orientation == 1 # x-direction
    λ_min = equations.v1_0 - equations.c_0
    λ_max = equations.v1_0 + equations.c_0
  else # y-direction
    λ_min = equations.v2_0 - equations.c_0
    λ_max = equations.v2_0 + equations.c_0
  end

  return λ_min, λ_max
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::LinearizedEulerEquations2D) = cons2cons(u, equations)
@inline cons2entropy(u, ::LinearizedEulerEquations2D) = u


function initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations2D)
  rho_prime = -cospi(2*t) * (sinpi(2*x[1]) + sinpi(2*x[2]))
  v1_prime = sinpi(2*t) * cospi(2*x[1])
  v2_prime = sinpi(2*t) * cospi(2*x[2])
  p_prime = rho_prime

  return SVector(rho_prime, v1_prime, v2_prime, p_prime)
end

function source_terms_convergence_test(u, x, t, equations::LinearizedEulerEquations2D)
  rho_prime = -cospi(2*t) * (cospi(2*x[1]) + cospi(2*x[2])) * 2 * pi
  v1_prime = -sinpi(2*t) * sinpi(2*x[1]) * 2 * pi
  v2_prime = -sinpi(2*t) * sinpi(2*x[2]) * 2 * pi
  p_prime = rho_prime

  return SVector(rho_prime, v1_prime, v2_prime, p_prime)
end

function initial_condition_entropy_wave(x, t, equations::LinearizedEulerEquations2D)
  # Parameters
  alpha = 1.0
  beta  = 250.0
  center = 0.5

  rho_prime = alpha * exp(-beta * ((x[1] - center)^2 + (x[2] - center)^2))

  return SVector(rho_prime, 0.0, 0.0, 0.0)
end

"""
    Boundary Condition taken from "acoustic_perturbation_2d.jl"
Boundary conditions for a solid wall.
"""
function boundary_condition_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                                 equations::LinearizedEulerEquations2D)
  # Boundary state is equal to the inner state except for the velocity. For boundaries
  # in the -x/+x direction, we multiply the velocity in the x direction by -1.
  # Similarly, for boundaries in the -y/+y direction, we multiply the velocity in the
  # y direction by -1
  if direction in (1, 2) # x direction
    u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4])
  else # y direction
    u_boundary = SVector(u_inner[1], u_inner[2], -u_inner[3], u_inner[4])
  end

  # Calculate boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end

end # muladd
