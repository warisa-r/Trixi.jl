
using OrdinaryDiffEq
using Trixi
using Plots

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)

A medium blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)
  # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)
  phi = atan(y_norm, x_norm)
  sin_phi, cos_phi = sincos(phi)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1  = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2  = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  p   = r > 0.5 ? 1.0E-3 : 1.245

  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_blast_wave

#surface_flux = flux_lax_friedrichs
surface_flux = flux_hlle
volume_flux  = flux_ranocha
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0, -2.0)
coordinates_max = ( 2.0,  2.0)
InitialRefinement = 4
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=InitialRefinement,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=true,
                                          variable=density_pressure)

MaxRefinement = InitialRefinement + 2
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=InitialRefinement,
                                      max_level =MaxRefinement, max_threshold=0.01)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=0.9)

callbacks_Step = CallbackSet(summary_callback,
                        alive_callback,
                        amr_callback, stepsize_callback)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        amr_callback)

###############################################################################
# run the simulation
#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks_Step);
=#

#=
CFL = 0.33
dt = 0.115520691731217091 / (2.0^(InitialRefinement - 3)) * CFL

b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS

ode_algorithm = PERK_Multi(4, 2, "/home/daniel/git/MA/EigenspectraGeneration/2D_CEE_BlastWave/",
                           bS, cEnd, stage_callbacks = ())
=#

CFL = 0.94
dt = 0.115520691731217091 / (2.0^(MaxRefinement - 3)) * CFL                           
ode_algorithm = PERK(4, "/home/daniel/git/MA/EigenspectraGeneration/2D_CEE_BlastWave/")

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
plot(sol)

pd = PlotData2D(sol)
plot(pd["p"])
plot!(getmesh(pd))