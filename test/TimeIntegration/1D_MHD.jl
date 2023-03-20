
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
gamma = 5/3
equations = IdealGlmMhdEquations1D(gamma)

initial_condition = initial_condition_convergence_test

surf_flux = flux_lax_friedrichs
surf_flux = flux_hll

#=
volume_flux = flux_hindenlang_gassner

solver = DGSEM(polydeg=4, surface_flux=surf_flux,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))
=#

solver = DGSEM(polydeg=4, surface_flux=surf_flux)     

coordinates_min = 0.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors=(:l2_error_primitive,
                                                            :linf_error_primitive),
                                     extra_analysis_integrals=(entropy, energy_total,
                                                               energy_kinetic, energy_internal,
                                                               energy_magnetic, cross_helicity))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)


###############################################################################
# run the simulation

#=
dtRef = 0.0188048589589016057
NumStagesRef = 16
CFL = 1.0
=#

dtRef = 0.018435236577440211
NumStagesRef = 16
CFL = 0.55

NumStages = 16

dtOptMin = NumStages / NumStagesRef * dtRef * CFL

#ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_MHD/")


ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_MHD/" * 
                                string(NumStages) * "/")

#=                                
NumEigVals, EigVals = Trixi.read_file("/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_MHD/EigenvalueList_Refined6.txt", ComplexF64)
dtRefStages = Trixi.MaxTimeStep(dtRef, EigVals, ode_algorithm)
=#

sol = Trixi.solve(ode, ode_algorithm,
                  #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks);

#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
=#

summary_callback() # print the timer summary
plot(sol)