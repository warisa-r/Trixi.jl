
using OrdinaryDiffEq, Plots, LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = InviscidBurgersEquation1D()

initial_condition = initial_condition_convergence_test

flux = flux_godunov
#flux = flux_lax_friedrichs
solver = DGSEM(polydeg=3, surface_flux=flux)

coordinates_min = 0.0
coordinates_max = 1.0

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=8,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_convergence_test)                            


###############################################################################
# ODE solvers, callbacks etc.

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors=(:l2_error_primitive,
                                                            :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution)
                        #stepsize_callback)


###############################################################################
# run the simulation

tspan = (0.0, 2)
ode = semidiscretize(semi, tspan)

NumStagesRef = 16
dtRef = 0.00291312662768177694

NumStages = 16
CFL = 0.79

NumStages = 32
CFL = 0.4798

dt = dtRef * NumStages/NumStagesRef * CFL
        
#ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/BurgersSourceTerm/")

ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/BurgersSourceTerm/" * 
                                string(NumStages) * "/")

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
plot(sol)


A = jacobian_ad_forward(semi)
Eigenvalues = eigvals(A)

# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

# Sometimes due to numerical issues some eigenvalues have positive real part, which is erronous (for hyperbolic eqs)
Eigenvalues = Eigenvalues[real(Eigenvalues) .< 0]

EigValsReal = real(Eigenvalues)
EigValsImag = imag(Eigenvalues)

plotdata = scatter(EigValsReal, EigValsImag, label = "Start")
display(plotdata)    

A = jacobian_ad_forward(semi, tspan[end], sol.u[end])
Eigenvalues = eigvals(A)

# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

EigValsReal = real(Eigenvalues)
EigValsImag = imag(Eigenvalues)

print(maximum(EigValsReal))

plotdata = scatter!(EigValsReal, EigValsImag, label = "End")
display(plotdata)
