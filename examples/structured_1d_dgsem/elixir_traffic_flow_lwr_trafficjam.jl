# The same setup as tree_1d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh

using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

max_car_density = 1.0
equations = TrafficFlowLWR1D(max_car_density)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
flux = flux_lax_friedrichs
flux = flux_hll
#flux = Trixi.flux_engquist_osher

solver = DGSEM(polydeg=3, surface_flux=flux)

#=
coordinates_min = (-1.0,) # minimum coordinate
coordinates_max = (1.0,) # maximum coordinate
cells_per_dimension = (32,)

# Create curved mesh with 16 cells
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)
=#

coordinates_min = -1.0
coordinates_max =  1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 10.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

pd = PlotData1D(sol)
plot(sol)