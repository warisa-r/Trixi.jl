
using OrdinaryDiffEq, LinearAlgebra, Plots
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = InviscidBurgersEquation1D()

# Discontinuous initial condition (Riemann Problem) leading to a shock to test e.g. correct shock speed.
function initial_condition_shock(x, t, equation::InviscidBurgersEquation1D)
  scalar = x[1] < 0.5 ? 1.5 : 0.5

  return SVector(scalar)
end

initial_condition = initial_condition_shock

num_flux = flux_lax_friedrichs
num_flux = flux_godunov
solver = DGSEM(polydeg=0, surface_flux=num_flux)

coordinates_min = 0.0
coordinates_max = 1.0

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

A = jacobian_ad_forward(semi)

println("Number Eigenvalues: ", size(A)[1])

Eigenvalues = eigvals(A)

# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

# Sometimes due to numerical issues some eigenvalues have positive real part, which is erronous (for hyperbolic eqs)
Eigenvalues = Eigenvalues[real(Eigenvalues) .< 0]

NumEigVals = length(Eigenvalues)
println("Number Eigenvalues in 2nd quadrant: ", NumEigVals)
display(Eigenvalues); println();

EigValsReal = real(Eigenvalues)
EigValsImag = imag(Eigenvalues)

plotdata = scatter(EigValsReal, EigValsImag, label = "Spectrum")
display(plotdata)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors=(:l2_error_primitive,
                                                            :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
