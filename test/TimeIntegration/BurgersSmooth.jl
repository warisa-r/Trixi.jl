
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

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors=(:l1_error, ))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)
                        #stepsize_callback)


###############################################################################
# run the simulation

tspan = (0.0, 5.0)
tspan = (0.0, 0.5) # To test discretization accuracy

ode = semidiscretize(semi, tspan)


NumStagesRef = 16
dtRef = 0.00291312662768177694

NumStages = 16
CFL = 0.79


NumStages = 28
# Negative beta
CFL = 0.72

#=
NumStages = 56
# Negative beta
CFL = 0.69
=#

#=
NumStages = 112
# Negative beta
CFL = 0.28
=#

CFL_Convergence = 1/1

dt = dtRef * NumStages/NumStagesRef * CFL * CFL_Convergence
        
#ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/BurgersSourceTerm/")


ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/BurgersSourceTerm/" * 
                                string(NumStages) * "/NegBeta/")


#=
NumEigVals, EigVals = Trixi.read_file("/home/daniel/git/MA/EigenspectraGeneration/Spectra/BurgersSourceTerm/EigenvalueList_Refined8.txt", ComplexF64)                                
M = Trixi.InternalAmpFactor(NumStages, ode_algorithm.alpha, ode_algorithm.beta, EigVals * NumStages / NumStagesRef * dtRef * CFL)
display(M * 10^(-15))
display(dt^3)

n_SSPRKS3 = 15
ode_algorithm = Trixi.SSPRKS3(n_SSPRKS3)
dt = Trixi.MaxTimeStep(n_SSPRKS3, tspan[2], EigVals, ode_algorithm)

M_SSPRKS3 = Trixi.InternalAmpFactor(EigVals * dt, ode_algorithm)
display(M_SSPRKS3 * 10^(-15))
display(dt^4) # Third order method => p + 1 = 4
=#

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks);

plot(sol)

TV0 = 0
for i in 1:length(sol.u[1])-1
  TV0 += abs(sol.u[1][i+1] - sol.u[1][i])
end

println("Initial Total Variation:\t", TV0)

TV = 0
for i in 1:length(sol.u[end])-1
  TV += abs(sol.u[end][i+1] - sol.u[end][i])
end

println("Final Total Variation:\t\t", TV)
println("TV Difference:\t\t\t", TV - TV0)

summary_callback() # print the timer summary

A = jacobian_ad_forward(semi)
Eigenvalues = eigvals(A)

EigValsReal = real(Eigenvalues)

println("Maximum real part of all EV of initial condiguration: ", maximum(EigValsReal))


A = jacobian_ad_forward(semi, tspan[end], sol.u[end])
Eigenvalues = eigvals(A)

EigValsReal = real(Eigenvalues)

println("Maximum real part of all EV of final condiguration: ", maximum(EigValsReal))
