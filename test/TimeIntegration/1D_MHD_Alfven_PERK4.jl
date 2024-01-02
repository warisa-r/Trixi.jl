
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
gamma = 5/3
equations = IdealGlmMhdEquations1D(gamma)

initial_condition = initial_condition_convergence_test

volume_flux = flux_hindenlang_gassner
solver = DGSEM(polydeg=4, surface_flux=flux_hll,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = 0.0
coordinates_max = 1.0

refinement_patches = ()

#=
refinement_patches = (
  (type="box", coordinates_min=(0.0), coordinates_max=(0.5)),
)
=#

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                refinement_patches=refinement_patches,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors=(:l2_error_primitive,
                                                            :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)


###############################################################################
# run the simulation

Stages = [10, 6]
#Stages = [10]
Stages = [6]
#Stages = [5]
#Stages = [6, 5]

#ode_algorithm = PERK4_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/1D_MHD_AlfvenWave/p4/")

CFL = 0.7 # [6, 5] With refinement

CFL = 1.0

# S = 5
dt = 0.002268333469433 * CFL

# S = 6
dt = 0.004367387960068 * CFL

# S = 10
#dt = 0.009300599096775 * CFL

# Compare case with refinement to PERK 3:

CFL = 1.0

# p = 3, S = 4
dt = 0.00378903581702616074 * CFL

# p = 3, S = 7
#dt = 0.00760528563318075584

Stages = [7, 4]
#Stages = [4]
#Stages = [7]

ode_algorithm = PERK3_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/1D_MHD_AlfvenWave/p3/")


sol = Trixi.solve(ode, ode_algorithm, dt = dt,
                  save_everystep=false, callback=callbacks);

using Plots
plot(sol)

pd = PlotData1D(sol)

plot(getmesh(pd))


stepsize_callback = StepsizeCallback(cfl=1.0)
callbacks = CallbackSet(summary_callback, stepsize_callback,
                        analysis_callback, alive_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

plot(sol)

summary_callback() # print the timer summary