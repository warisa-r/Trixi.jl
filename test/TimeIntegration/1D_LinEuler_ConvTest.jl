
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the 1 linearized Euler equations

equations = LinearizedEulerEquations1D(1.0, 1.0, 1.0)

initial_condition = initial_condition_convergence_test

solver = DGSEM(polydeg=2, surface_flux=flux_lax_friedrichs)

coordinates_min = 0
coordinates_max = 1

RefinementLevel = 7
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=RefinementLevel,
                n_cells_max=30_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, extra_analysis_errors=(:l1_error, ))

alive_callback = AliveCallback(analysis_interval=analysis_interval)


callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation


dtRef = 0.0124186746499617591
NumStagesRef = 16

NumStages = 16
CFL = 1.0

#=
NumStages = 30
CFL = 1.0


NumStages = 60
CFL = 1.0


NumStages = 120
CFL = 0.7
=#

RefinementOpt = 7
CFL_Refinement = 1.0 / 2^(RefinementLevel - RefinementOpt)

dtOptMin = NumStages / NumStagesRef * dtRef * CFL * CFL_Refinement

ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Lin_Euler/ConvergenceTest/")

#=
ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Lin_Euler/ConvergenceTest/" * 
                                string(NumStages) * "/")
=#

sol = Trixi.solve(ode, ode_algorithm,
                  #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks);


summary_callback() # print the timer summary

plot(sol)