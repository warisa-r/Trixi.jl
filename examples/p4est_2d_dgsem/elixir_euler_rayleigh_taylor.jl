
using Trixi, OrdinaryDiffEq, Plots

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_rayleigh_taylor_instability(coordinates, t, equations::CompressibleEulerEquations2D)

Setup used for the Rayleigh-Taylor instability. Initial condition adapted from
- Shi, Jing, Yong-Tao Zhang, and Chi-Wang Shu (2003).
  Resolution of high order WENO schemes for complicated flow structures.
  [DOI](https://doi.org/10.1016/S0021-9991(03)00094-9).

- Remacle, Jean-François, Joseph E. Flaherty, and Mark S. Shephard (2003).
  An adaptive discontinuous Galerkin technique with an orthogonal basis applied to compressible
  flow problems.
  [DOI](https://doi.org/10.1137/S00361445023830)

The domain is [0, 0.25] x [0, 1]. Boundary conditions can be reflective wall boundary conditions on
all boundaries or
- periodic boundary conditions on the left/right boundaries
- Dirichlet boundary conditions on the top/bottom boundaries

This should be used together with `source_terms_rayleigh_taylor_instability`, which is
defined below.
"""
@inline function initial_condition_rayleigh_taylor_instability(x, t,
                                                               equations::CompressibleEulerEquations2D,
                                                               slope=1000)
  tol = 1e2*eps()

  if x[2] < 0.5
    p = 2*x[2] + 1
  else
    p = x[2] + 3/2
  end

  # smooth the discontinuity to avoid ambiguity at element interfaces
  smoothed_heaviside(x, left, right) = left + 0.5*(1 + tanh(slope * x)) * (right-left)
  rho = smoothed_heaviside(x[2] - 0.5, 2.0, 1.0)

  c = sqrt(equations.gamma * p / rho)
  # the velocity is multiplied by sin(pi*y)^6 as in Remacle et al. 2003 to ensure that the
  # initial condition satisfies reflective boundary conditions at the top/bottom boundaries.
  v = -0.025 * c * cos(8*pi*x[1]) * sin(pi*x[2])^6
  u = 0.0

  return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function source_terms_rayleigh_taylor_instability(u, x, t,
                                                          equations::CompressibleEulerEquations2D)
  g = 1.0
  rho, rho_v1, rho_v2, rho_e = u

  return SVector(0.0, 0.0, g*rho, g*rho_v2)
end

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_hll,
               volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

num_elements = 32
trees_per_dimension = (num_elements, 4 * num_elements)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=1, initial_refinement_level=0,
                 coordinates_min=(0.0, 0.0), coordinates_max=(0.25, 1.0),
                 periodicity=false)

initial_condition = initial_condition_rayleigh_taylor_instability

boundary_conditions = Dict( :x_neg => boundary_condition_slip_wall,
                            #:x_neg => boundary_condition_periodic,
                            :y_neg => boundary_condition_slip_wall,
                            :y_pos => boundary_condition_slip_wall,
                            :x_pos => boundary_condition_slip_wall
                            #:x_pos => boundary_condition_periodic
                            )

#=                            
boundary_conditions_parabolic = Dict( :x_neg => boundary_condition_slip_wall,
                                      :y_neg => boundary_condition_slip_wall,
                                      :y_pos => boundary_condition_slip_wall,
                                      :x_pos => boundary_condition_slip_wall)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions=(boundary_conditions,
                                                                  boundary_conditions_parabolic))
=#

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition, solver;
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1.5)

callbacks = CallbackSet(summary_callback,
                        #analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

#=
sol = solve(ode, RDPK3SpFSAL49(); abstol=1.0e-6, reltol=1.0e-6,
            ode_default_options()..., callback=callbacks);
=#


sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
plot(sol)

pd = PlotData2D(sol)

plot(pd["rho"])