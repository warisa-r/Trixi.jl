
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the 1 linearized Euler equations

equations = LinearizedEulerEquations1D(1.0, 1.0, 1.0)

initial_condition = initial_condition_entropy_wave

solver = DGSEM(polydeg=2, surface_flux=flux_hll)

coordinates_min = 0
coordinates_max = 1
RefinementLevel = 7
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=RefinementLevel,
                n_cells_max=30_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
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

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                  dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep=false, callback=callbacks_DE);          

plot(sol)
pd = PlotData1D(sol)
plot(pd["rho_prime"])

xeval = 0.9
x_char = Trixi.compute_char_initial_pos(xeval, tspan[2], equations)
w = zeros(3, 1)
for p = 1:3
  w[p] = Trixi.initial_condition_char_vars_entropy_wave(x_char[p], p, equations)
end
display(Trixi.compute_primal_sol(w, equations))