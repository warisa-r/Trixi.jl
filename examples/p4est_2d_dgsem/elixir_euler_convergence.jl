
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

# Get the DG approximation space
surface_flux = flux_hllc
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux)

###############################################################################

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)

trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=3, initial_refinement_level=0,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 300
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=300,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
