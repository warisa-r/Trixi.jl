
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the 1 linearized Euler equations

equations = LinearizedEulerEquations1D(1.0, 1.0, 0.1)

initial_condition = initial_condition_acoustic_wave

solver = DGSEM(polydeg=2, surface_flux=flux_lax_friedrichs)

coordinates_min = 0
coordinates_max = 1
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=9,
                n_cells_max=30_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10/1.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, extra_analysis_errors=(:l1_error, ))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation


dtRef = 0.022
NumStagesRef = 16

#CFL = 0.7
CFL = 0.99
NumStages = 30

CFLGrid = 0.25

dtOptMin = NumStages / NumStagesRef * dtRef * CFL * CFLGrid

#ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Lin_Euler/Acoustic_Wave/")


ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Lin_Euler/Acoustic_Wave/" * 
                                string(NumStages) * "/")


sol = Trixi.solve(ode, ode_algorithm,
                  #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary

plot(sol)