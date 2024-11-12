# The same setup as tree_1d_dgsem/elixir_euler_source_terms.jl
# to verify the StructuredMesh implementation against TreeMesh
using Convex, ECOS, Clarabel
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test

# Note that the expected EOC of 5 is not reached with this flux.
# Using flux_hll instead yields the expected EOC.
solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

coordinates_min = (0.0,)
coordinates_max = (2.0,)
cells_per_dimension = (16,)

mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

# Construct embedded order paired explicit Runge-Kutta method with 10 stages and 6 evaluation stages for given simulation setup.
# Pass `tspan` to calculate maximum time step allowed for the bisection algorithm used 
# in calculating the polynomial coefficients in the ODE algorithm.
ode_algorithm = Trixi.EmbeddedPairedRK3(10, 10, tspan, semi)
#cfl_number = Trixi.calculate_cfl(ode_algorithm, ode)

stepsize_callback = StepsizeCallback(cfl = 1.8)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        save_solution,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
