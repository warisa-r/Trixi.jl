# The same setup as tree_2d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh

using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the linear advection equation
a = 2.5
advection_velocity = (a, -a)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-5.0, -5.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 5.0,  5.0) # maximum coordinates (max(x), max(y))

NumCells = 100
cells_per_dimension = (NumCells, NumCells)

# Create curved mesh with 16 x 16 elements
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_gauss, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 100.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=1000, 
                                     extra_analysis_errors=(:l1_error,))


# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback) #, stepsize_callback)

stepsize_callback = StepsizeCallback(cfl=1)
callbacksDE = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation


NumCellsRef = 24
CFLCells = NumCellsRef / NumCells

aRef = 1
CFL_a = aRef / a

dtRef = 0.384728193515911698
NumStagesRef = 16

NumStages = 16
CFL = 1.0


NumStages = 30
CFL = 1.0


NumStages = 60
CFL = 1.0


NumStages = 120
CFL = 0.98


CFL_Convergence = 1/1

dtOptMin = dtRef * (NumStages / NumStagesRef) * CFL * CFLCells * CFL_a * CFL_Convergence

#ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_Adv/")


ode_algorithm = FE2S(NumStages, 
                     "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_Adv/" * string(NumStages) * "/NegBeta/")

NumEigVals, EigVals = Trixi.read_file("/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_Adv/EigenvalueList_24.txt", ComplexF64)
M = Trixi.InternalAmpFactor(NumStages, ode_algorithm.alpha, ode_algorithm.beta, EigVals * NumStages / NumStagesRef * dtRef * CFL * CFL_a)
display(M * 10^(-15))
display(dtOptMin^3)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks);

#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacksDE);
=#

# Print the timer summary
summary_callback()
plot(sol)
