using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
equations_euler = Trixi.CompressibleEulerTwoFluidsEquations1D(gammas = (5/3, 5/3),
                                                       epsilon = 1e-4)
delta = 1e-2 # perturbation amplitude

initial_condition = initial_condition_convergence_test #TODO: change this
polydeg = 3

solver = DGSEM(polydeg = polydeg, surface_flux = flux_hll,
               volume_integral = VolumeIntegralPureLGLFiniteVolume(flux_hll))

coordinates_min = 0.0
coordinates_max = 2.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000)

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# semidiscretization of the hyperbolic diffusion equations
#=
equations_plasma = HyperbolicDiffusionEquations1D()
solver_plasma = DGSEM(polydeg, flux_lax_friedrichs)

semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_condition,
                                            solver_gravity,
                                            source_terms = source_terms_harmonic) #TODO: Check if this is valid for our equations

###############################################################################
# combining both semidiscretizations for Euler + Poisson equation for electric potential
parameters = ParametersEulerplasma(scaled_debye_length = 2.0,
                                    cfl = 1.5,
                                    resid_tol = 1.0e-10,
                                    n_iterations_max = 1000,
                                    timestep_plasma = timestep_plasma_erk52_3Sstar!)

semi = SemidiscretizationEulerplasma(semi_euler, semi_plasma, parameters)
=#
###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 2.0)
ode = semidiscretize(semi_euler, tspan) # Temporarily using semi_euler to check for coding mistakes

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi_euler, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = false,
                                     save_final_solution = false,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary