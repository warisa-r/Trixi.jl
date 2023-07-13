
using OrdinaryDiffEq, Plots, LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

PolyDegree = 3

solver = DGSEM(polydeg=PolyDegree, surface_flux=flux_lax_friedrichs)

coordinates_min = -5.0 # minimum coordinate
coordinates_max =  5.0 # maximum coordinate

InitialRefinement = 6
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
EndTime = 10


# Create ODEProblem
ode = semidiscretize(semi, (StartTime, EndTime));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

Interval = 500
# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=Interval, extra_analysis_errors=(:conservation_error,))

# 4 Levels

amr_controller = ControllerThreeLevel(semi, 
                                      IndicatorMax(semi, variable=first),
                                      base_level=InitialRefinement,
                                      med_level=InitialRefinement+1, med_threshold=0.1, #0.1
                                      max_level=InitialRefinement+3, max_threshold=0.6) #0.6


# 3 Levels

amr_controller = ControllerThreeLevel(semi, 
                                      IndicatorMax(semi, variable=first),
                                      base_level=InitialRefinement,
                                      med_level=InitialRefinement+1, med_threshold=0.1, #0.1
                                      max_level=InitialRefinement+2, max_threshold=0.6) #0.6


amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=false) # Adaption of initial condition not yet supported

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, 
                        analysis_callback,
                        amr_callback)

#callbacks = CallbackSet(summary_callback, analysis_callback)

stepsize_callback = StepsizeCallback(cfl=0.5)                    
callbacksSSPRK22 = CallbackSet(summary_callback,
                               amr_callback,
                               analysis_callback,
                               stepsize_callback)         

###############################################################################
# run the simulation

# S=4, D=3
CFL = 0.7
dt = 0.0545930727967061102 / (2.0^(InitialRefinement - 4)) * CFL * coordinates_max

b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS
ode_algorithm = PERK_Multi(4, 2, "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/", 
                           bS, cEnd, stage_callbacks = ())

sol = Trixi.solve(ode, ode_algorithm, dt = dt, save_everystep=false, callback=callbacks);

#=
sol = solve(ode, SSPRK22(),
            dt = 42.0,
            save_everystep=false, callback=callbacksSSPRK22);
=#

# Print the timer summary
summary_callback()

plot(sol, plot_title = EndTime)

pd = PlotData1D(sol)
plot!(getmesh(pd))