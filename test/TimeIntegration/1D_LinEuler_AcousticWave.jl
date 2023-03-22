
using OrdinaryDiffEq, Plots, LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the 1 linearized Euler equations

equations = LinearizedEulerEquations1D(1.0, 1.0, 0.1)

initial_condition = initial_condition_acoustic_wave

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

tspan = (0.0, 10/1.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, extra_analysis_errors=(:l1_error, ))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

stepsize_callback = StepsizeCallback(cfl=0.25)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks_DE = CallbackSet(summary_callback, analysis_callback, stepsize_callback)                        

###############################################################################
# run the simulation


dtRef = 0.022
NumStagesRef = 16

#CFL = 0.7
CFL = 0.99
NumStages = 120

RefinementOpt = 7
CFL_Refinement = 1.0 / 2^(RefinementLevel - RefinementOpt)

CFL_Conv = 1/1

dtOptMin = NumStages / NumStagesRef * dtRef * CFL * CFL_Refinement * CFL_Conv

#ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Lin_Euler/Acoustic_Wave/")


ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Lin_Euler/Acoustic_Wave/" * 
                                string(NumStages) * "/")

NumEigVals, EigVals = Trixi.read_file("/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Lin_Euler/Acoustic_Wave/EigenvalueList_Refined7.txt", ComplexF64)  
M = Trixi.MaxInternalAmpFactor(NumStages, ode_algorithm.alpha, ode_algorithm.beta, EigVals * dtOptMin)

sol = Trixi.solve(ode, ode_algorithm,
                  #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks);

#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                  dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep=false, callback=callbacks_DE);          
=#

plot(sol)

A = jacobian_ad_forward(semi)
Eigenvalues = eigvals(A)

EigValsReal = real(Eigenvalues)

println("Maximum real part of all EV of initial condiguration: ", maximum(EigValsReal))


A = jacobian_ad_forward(semi, tspan[end], sol.u[end])
Eigenvalues = eigvals(A)

EigValsReal = real(Eigenvalues)

println("Maximum real part of all EV of final condiguration: ", maximum(EigValsReal))

summary_callback() # print the timer summary