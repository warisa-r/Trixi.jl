
using OrdinaryDiffEq, Plots, LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the 1 linearized Euler equations

equations = LinearizedEulerEquations1D(1.0, 1.0, 0.5)

solver = DGSEM(polydeg=4, surface_flux=flux_hll)

coordinates_min = 0
coordinates_max = 1
RefinementLevel = 7
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=RefinementLevel,
                n_cells_max=30_000)

function initial_condition_char_vars(x, t, equations::LinearizedEulerEquations1D)
  # Trace back characteristics
  x_char = Trixi.compute_char_initial_pos(x, t, equations)

  # Employ periodicity
  for p = 1:3
    while x_char[p] < coordinates_min
      x_char[p] += coordinates_max - coordinates_min
    end
    while x_char[p] > coordinates_max
      x_char[p] -= coordinates_max - coordinates_min
    end
  end

  # Set up characteristic variables
  w = zeros(3)
  for p = 1:3
    w[p] = dot(equations.eigenvectors_inv[p,:], initial_condition_convergence_test(x_char[p], 0, equations)) # Assumes t_0 = 0
  end

  return Trixi.compute_primal_sol(w, equations)
end                

initial_condition = initial_condition_char_vars
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10)
#tspan = (0.0, 0) # Test spatial discretization accuracy

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, extra_analysis_errors=(:l1_error, ))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

stepsize_callback = StepsizeCallback(cfl=1.0)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks_DE = CallbackSet(summary_callback, analysis_callback, stepsize_callback)                        

###############################################################################
# run the simulation

#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                  dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep=false, callback=callbacks_DE);          
=#

dtRef = 0.0065130328005761837
NumStagesRef = 16


NumStages = 16
CFL = 1.0


NumStages = 28
CFL = 1.0


NumStages = 56
CFL = 1.0


NumStages = 112
CFL = 1.0


CFL_Convergence = 1/32

dtOptMin = NumStages / NumStagesRef * dtRef * CFL * CFL_Convergence

#ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_LinEuler/")


ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_LinEuler/" * 
                                string(NumStages) * "/NegBeta/")

#ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/Optim_Roots/RKM_Construction/InternalStability/5Params/MinimizeBeta/LebedevsIdea/")

#=
NumEigVals, EigVals = Trixi.read_file("/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_LinEuler/Hull_7.txt", ComplexF64)

M = Trixi.InternalAmpFactor(NumStages, ode_algorithm.alpha, ode_algorithm.beta, EigVals * NumStages / NumStagesRef * dtRef * CFL)
display(M * 10^(-15))
display(dtOptMin^3)
=#

sol = Trixi.solve(ode, ode_algorithm,
                  #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks);

#plot(sol)