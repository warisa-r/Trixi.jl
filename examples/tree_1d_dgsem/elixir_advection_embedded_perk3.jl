
# Convex and ECOS are imported because they are used for finding the optimal time step and optimal 
# monomial coefficients in the stability polynomial of P-ERK time integrators.
using Convex, ECOS

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0 # maximum coordinate

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000) # set maximum capacity of tree data structure

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 20.0
tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step

alive_callback = AliveCallback(alive_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 0.1,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver

# Construct embedded order paired explicit Runge-Kutta method with 10 stages and 7 evaluation stages for given simulation setup.
# Pass `tspan` to calculate maximum time step allowed for the bisection algorithm used 
# in calculating the polynomial coefficients in the ODE algorithm.
ode_algorithm = Trixi.EmbeddedPairedExplicitRK2(10, tspan, semi)
#ode_algorithm = Trixi.PairedExplicitRK2(10, tspan, semi)
# Calculate the CFL number for the given ODE algorithm and ODE problem (cfl_number calculate from dt_opt of the optimization of
# b values in the Butcher tableau of the ODE algorithm).
#cfl_number = Trixi.calculate_cfl(ode_algorithm, ode)
#stepsize_callback = StepsizeCallback(cfl = cfl_number)

controller = Trixi.PIDController(0.60, -0.33, 0) # Intiialize the controller 

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        save_solution,
                        analysis_callback)

###############################################################################
# run the simulation
#sol = Trixi.solve(ode, ode_algorithm,
#                  dt = 1.0, # Manual time step value, will be overwritten by the stepsize_callback when it is specified.
#                  save_everystep = false, callback = callbacks);

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 1.0, # Manual time step value, will be overwritten by the stepsize_callback when it is specified.
                  save_everystep = false, callback = callbacks, controller = controller, abstol = 1e-4, reltol = 1e-4);

# Print the timer summary
summary_callback()

#=
ode_algorithm_cfl = Trixi.PairedExplicitRK2(10, tspan, semi)
ode_algorithm_embedded = Trixi.EmbeddedPairedExplicitRK2(10, tspan, semi)
cfl_number = Trixi.calculate_cfl(ode_algorithm, ode)
stepsize_callback = StepsizeCallback(cfl = cfl_number)

# Tolerances to test
tolerances = 10.0 .^ (-7:1:-1)

# Arrays to store errors for both cases
errors_embedded_callback = Float64[]
errors_cfl_callback = Float64[]

# Arrays to store num_rhs for both case
nums_rhs_embedded = []
nums_rhs_cfl = fill(200, length(tolerances))

# Define callbacks (reuse from earlier setup)
callbacks_embedded = CallbackSet(summary_callback, alive_callback, save_solution,
analysis_callback)
callbacks_cfl = CallbackSet(summary_callback, alive_callback, save_solution,
   analysis_callback, stepsize_callback)

# Loop over each tolerance value
for tol in tolerances

    # Solve with the StepsizeCallback
    num_rhs_embedded, sol_embedded = Trixi.solve(ode, ode_algorithm_embedded, dt = 1.0, # Manual time step value, will be overwritten by the stepsize_callback when it is specified.
                                save_everystep = false, callback = callbacks_embedded, controller = controller, abstol = tol, reltol = tol);
    results_embedded = analysis_callback(sol_embedded)
    push!(errors_embedded_callback, results_embedded.l2[1]) # Assuming l2[1] stores the error
    push!(nums_rhs_embedded, num_rhs_embedded)

    # Solve without the StepsizeCallback
    sol_cfl = Trixi.solve(ode, ode_algorithm_cfl, dt = 1.0, # Manual time step value, will be overwritten by the stepsize_callback when it is specified.
                          save_everystep = false, callback = callbacks_cfl);
    results_cfl = analysis_callback(sol_cfl)
    push!(errors_cfl_callback, results_cfl.l2[1]) # Assuming l2[1] stores the error
end

round_cfl = round(cfl_number, digits=3)

# Plot the error against tolerances
plot(tolerances, errors_embedded_callback, xscale = :log10, yscale = :log10,
     marker = :circle, label = "PERK21", color = :blue,
     xlabel = "Tolerances (abstol = reltol)", ylabel = "L2 Error",
     title = "Error vs. Tolerance from 1D advection with tree mesh")
plot!(tolerances, errors_cfl_callback, marker = :square, color = :red,
         label = "CFL = $round_cfl")

plot!(size = (1000, 800))

savefig("plot_l2_error_advection_PERK21.png")

# Plot the error against tolerances
plot(tolerances, nums_rhs_embedded, xscale = :log10,
     marker = :circle, label = "PERK21", color = :blue,
     xlabel = "Tolerances (abstol = reltol)", ylabel = "Number of RHS evaluation",
     title = "Number of RHS evaluation vs. Tolerance from 1D advection with tree mesh")
plot!(tolerances, nums_rhs_cfl, marker = :square, color = :red,
         label = "CFL = $round_cfl")

plot!(size = (1000, 800))

savefig("plot_num_rhs_advection_PERK21.png")
=#