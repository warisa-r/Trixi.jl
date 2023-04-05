using OrdinaryDiffEq, Plots, LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the linearized Euler equations

# Convergence test
equations = LinearizedEulerEquations2D(1.0, 1.0, 1.0, 1.0)

#flux=flux_lax_friedrichs
flux = flux_hll
solver = DGSEM(polydeg=4, surface_flux=flux)

#coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0,  1.0) # maximum coordinates (max(x), max(y))

RefinementLevel = 6
# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=RefinementLevel,
                n_cells_max=30_000)

initial_condition = initial_condition_convergence_test

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms=source_terms_convergence_test)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 0.2
tspan = (0.0, 10.5)
#tspan = (0.0, 0.0) # test discretization accuracy

ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

analysis_interval = 1000

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, 
                                     extra_analysis_errors=(:l1_error,))

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval=analysis_interval)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

stepsize_callback = StepsizeCallback(cfl=1)
callbacksDE = CallbackSet(summary_callback, analysis_callback, alive_callback, stepsize_callback)

###############################################################################
# run the simulation


dtRef = 0.0457468077940575338
NumStagesRef = 16

RefinementOpt = 3
CFL_Refinement = 1.0 / 2^(RefinementLevel - RefinementOpt)


NumStages = 16
CFL = 0.99


NumStages = 32
CFL = 0.99


#=
NumStages = 60
CFL = 0.99
=#

#=
NumStages = 96
CFL = 0.73
=#

#=
# Seems not to converge
NumStages = 128
CFL = 0.73
=#

CFL_Convergence = 1/32

dtOptMin = NumStages / NumStagesRef * dtRef * CFL * CFL_Refinement * CFL_Convergence

#ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_LinEuler_ConvTest/")


ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_LinEuler_ConvTest/" * 
                                string(NumStages) * "/NegBeta/")


#=               
NumEigVals, EigVals = Trixi.read_file("/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_LinEuler_ConvTest/EigenvalueList_Refined3.txt", ComplexF64)
M = Trixi.InternalAmpFactor(NumStages, ode_algorithm.alpha, ode_algorithm.beta, EigVals * NumStages / NumStagesRef * dtRef * CFL)
display(M * 10^(-15))
display(dtOptMin^3)
=#


sol = Trixi.solve(ode, ode_algorithm,
                  #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
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