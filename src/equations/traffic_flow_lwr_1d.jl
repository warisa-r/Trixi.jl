# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


  @doc raw"""
      TrafficFlowLWR1D
  
  The classic Lighthill-Witham Richards model for 1D traffic flow.
  ```math
  \partial_t u + u_{\text{max}} \partial_1 u(1-u) = 0
  ```
  See e.g. Section 11.1 of 
  - Randall LeVeque (2002)
  Finite Volume Methods for Hyperbolic Problems
  [DOI: 10.1017/CBO9780511791253]https://doi.org/10.1017/CBO9780511791253
  """
  struct TrafficFlowLWR1D{RealT<:Real} <: AbstractTrafficFlowLWR{1, 1} 
    u_max::RealT
  
    function TrafficFlowLWR1D(u_max=1.0)
      new{typeof(u_max)}(u_max)
    end
  end
  
  
  varnames(::typeof(cons2cons), ::TrafficFlowLWR1D) = ("scalar", )
  varnames(::typeof(cons2prim), ::TrafficFlowLWR1D) = ("scalar", )
  
  
  # Set initial conditions at physical location `x` for time `t`
  """
      initial_condition_constant(x, t, equations::TrafficFlowLWR1D)
  
  A constant initial condition to test free-stream preservation.
  """
  function initial_condition_constant(x, t, equation::TrafficFlowLWR1D)
    return SVector(2.0)
  end
  
  
  """
      initial_condition_convergence_test(x, t, equations::TrafficFlowLWR1D)
  
  A smooth initial condition used for convergence tests.
  """
  function initial_condition_convergence_test(x, t, equation::TrafficFlowLWR1D)
    c = 2.0
    A = 1.0
    L = 1
    f = 1/L
    omega = 2 * pi * f
    scalar = c + A * sin(omega * (x[1] - t))
  
    return SVector(scalar)
  end
  
  
  """
      source_terms_convergence_test(u, x, t, equations::TrafficFlowLWR1D)
  
  Source terms used for convergence tests in combination with
  [`initial_condition_convergence_test`](@ref).
  """
  @inline function source_terms_convergence_test(u, x, t, equations::TrafficFlowLWR1D)
    # Same settings as in `initial_condition`
    c = 2.0
    A = 1.0
    L = 1
    f = 1/L
    omega = 2 * pi * f
    du = omega * cos(omega * (x[1] - t)) * (1 + sin(omega * (x[1] - t)))
  
    return SVector(du)
  end
  
  
  # Pre-defined source terms should be implemented as
  # function source_terms_WHATEVER(u, x, t, equations::TrafficFlowLWR1D)
  
  
  # Calculate 1D flux in for a single point
  @inline function flux(u, orientation::Integer, equation::TrafficFlowLWR1D)
    return SVector(equation.u_max * u[orientation] * (1.0 - u[orientation]))
  end
  
  
  # Calculate 1D jacobian of flux in for a single point
  @inline function flux_jacobian(u, orientation::Integer,equation::TrafficFlowLWR1D)
    return SVector(equation.u_max * (1.0 - 2 * u[orientation]))
  end
  
  
  # Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
  @inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equation::TrafficFlowLWR1D)
    u_L = u_ll[1]
    u_R = u_rr[1]
  
    λ_max = max(abs(flux_jacobian(u_L, orientation, equation)[orientation]), 
                abs(flux_jacobian(u_R, orientation, equation)[orientation]))
  end
  
  
  # Calculate minimum and maximum wave speeds for HLL-type fluxes
  @inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer, equation::TrafficFlowLWR1D)
    u_L = u_ll[1]
    u_R = u_rr[1]
  
    jac_L = flux_jacobian(u_L, orientation, equation)[orientation]
    jac_R = flux_jacobian(u_R, orientation, equation)[orientation]
    
    λ_min = min(jac_L, jac_R)
    λ_max = max(jac_L, jac_R)
  
    return λ_min, λ_max
  end
  
  
  @inline function max_abs_speeds(u, equation::TrafficFlowLWR1D)
    return (abs(flux_jacobian(u, 1, equation)[1]),)
  end
  
  # TODO: Work flux engquist-osher out! (split integral)
  
  
  # Convert conservative variables to primitive
  @inline cons2prim(u, equation::TrafficFlowLWR1D) = u
  
  # Convert conservative variables to entropy variables
  @inline cons2entropy(u, equation::TrafficFlowLWR1D) = u
  
  
  # Calculate entropy for a conservative state `cons`
  @inline entropy(u::Real, ::TrafficFlowLWR1D) = 0.5 * u^2
  @inline entropy(u, equation::TrafficFlowLWR1D) = entropy(u[1], equation)
  
  
  # Calculate total energy for a conservative state `cons`
  @inline energy_total(u::Real, ::TrafficFlowLWR1D) = 0.5 * u^2
  @inline energy_total(u, equation::TrafficFlowLWR1D) = energy_total(u[1], equation)
  
  
  end # @muladd
  