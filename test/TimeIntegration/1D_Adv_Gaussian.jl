
using OrdinaryDiffEq, Plots, LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
PolyDegree = 3

solver = DGSEM(polydeg=PolyDegree, surface_flux=flux_lax_friedrichs)
#solver = DGSEM(polydeg=PolyDegree, surface_flux=flux_godunov) # Equivalent

coordinates_min = -5.0 # minimum coordinate
coordinates_max =  5.0 # maximum coordinate

InitialRefinement = 9
# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                # Start from one cell => Results in 1 + 2 + 4 + 8 + 16 = 2^5 - 1 = 31 cells
                initial_refinement_level=InitialRefinement,
                n_cells_max=30_000) # set maximum capacity of tree data structure

initial_condition = initial_condition_gauss

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

StartTime = 0.0
EndTime = 100
#EndTime = 0

# Create ODEProblem
ode = semidiscretize(semi, (StartTime, EndTime));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

Interval = 5000
# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=Interval, 
                                     extra_analysis_errors=(:conservation_error, :l1_error))

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, 
                        analysis_callback)

stepsize_callback = StepsizeCallback(cfl=0.125)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks_DE = CallbackSet(summary_callback, analysis_callback, stepsize_callback)
###############################################################################
# run the simulation

dtRef = 0.0360458314265997635
NumStagesRef = 16

CFL = 1.0
NumStages = 104

CFL_Convergence = 1/1

dtOptMin = NumStages / NumStagesRef * dtRef * CFL * CFL_Convergence


ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Adv/" * 
                                string(NumStages) * "/PosBeta/")


#=
ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Adv/" * 
                                string(NumStages) * "/NonLebedev/")
=#

NumEigVals, EigVals = Trixi.read_file("/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Adv/EigenvalueList_Refined9.txt", ComplexF64)
M = Trixi.InternalAmpFactor(NumStages, ode_algorithm.alpha, ode_algorithm.beta, EigVals * dtOptMin)
display(M * 10^(-15))
display(dtOptMin^3)


sol = Trixi.solve(ode, ode_algorithm,
                  #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks);

#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks_DE);                       
=#

# Print the timer summary
summary_callback()

plot(sol)