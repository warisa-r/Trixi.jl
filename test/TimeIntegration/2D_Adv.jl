# The same setup as tree_2d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh

using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (1, -1)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=2, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

NumCells = 25
cells_per_dimension = (NumCells, NumCells)

# Create curved mesh with 16 x 16 elements
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

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


# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback) #, stepsize_callback)


###############################################################################
# run the simulation

NumCellsRef = 25
CFLCells = NumCellsRef / NumCells

dtRef = 0.127168984670788632
NumStagesRef = 16

NumStages = 16
CFL = 1.0


NumStages = 28 
CFL = 1.0


NumStages = 56
CFL = 1.0

#=
NumStages = 112
CFL = 0.7
=#

dtOptMin = dtRef * (NumStages / NumStagesRef) * CFL * CFLCells

#ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_Adv/")


ode_algorithm = FE2S(NumStages, 
                     "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_Adv/" * string(NumStages) * "/")


sol = Trixi.solve(ode, ode_algorithm,
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()
plot(sol)
