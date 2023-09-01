# The same setup as tree_2d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh

using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (1, 1)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

trees_per_dimension = (16, 16)

# Create P4estMesh with 8 x 8 trees and 16 x 16 elements
mesh = P4estMesh(trees_per_dimension, polydeg=3,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 initial_refinement_level=0)

mesh_file = joinpath(@__DIR__, "square_unstructured_1.inp")
isfile(mesh_file) || download("https://gist.githubusercontent.com/efaulhaber/a075f8ec39a67fa9fad8f6f84342cbca/raw/a7206a02ed3a5d3cadacd8d9694ac154f9151db7/square_unstructured_1.inp",
                              mesh_file)

mesh = P4estMesh{2}(mesh_file, polydeg=3, initial_refinement_level=2)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver,
                                    boundary_conditions=Dict(
                                      :all => BoundaryConditionDirichlet(initial_condition_convergence_test)
                                    ))

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

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback)


###############################################################################
# run the simulation
#=
# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
=#
b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS

#callbacks_Stage = (PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6,), variables=(Trixi.scalar,)), )

ode_algorithm = PERK_Multi(4, 2, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_Adv/P4est/",
                            bS, cEnd, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#=
ode_algorithm = PERK(4, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_Adv/P4est/",
                            bS, cEnd)
=#

# S=4 only
dt = 0.0272965311996813413 * 1.1

# S=8 only
dt = dt = 0.0272965311996813413 * 2.3

# S=16 only
#dt = dt = 0.0272965311996813413 * 4.4

dt = 0.0272965311996813413 * 0.6

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks);

Plots.plot(sol)
Plots.plot!(getmesh(PlotData2D(sol)))

# Print the timer summary
summary_callback()