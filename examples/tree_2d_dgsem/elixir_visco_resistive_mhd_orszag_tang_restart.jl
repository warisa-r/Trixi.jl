
using OrdinaryDiffEq
using Trixi

###############################################################################
# create a restart file

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
prandtl_number() = 1.0

# Make less diffusive to still have convection-dominated spectra
#mu() = 1e-2
mu() = 1e-3
#eta() = 1e-2
eta() = 1e-3

gamma = 5/3
equations = IdealGlmMhdEquations2D(gamma)
equations_parabolic = ViscoResistiveMhdDiffusion2D(equations, mu = mu(),
                                          Prandtl = prandtl_number(),
                                          eta = eta(),
                                          gradient_variables = GradientVariablesPrimitive())

"""
    initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations2D)

The classical Orszag-Tang vortex test case. Here, the setup is taken from
- https://onlinelibrary.wiley.com/doi/pdf/10.1002/fld.4681
"""
function initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations2D)
  rho = 1.0
  v1 = -2 * sqrt(pi) * sin(x[2])
  v2 =  2 * sqrt(pi) * sin(x[1])
  v3 = 0.0
  p = 15/4 + 0.25 * cos(4*x[1]) + 0.8 * cos(2*x[1])*cos(x[2]) - cos(x[1])*cos(x[2]) + 0.25 * cos(2*x[2])
  B1 = -sin(x[2])
  B2 =  sin(2.0*x[1])
  B3 = 0.0
  psi = 0.0
  return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end
initial_condition = initial_condition_orszag_tang

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
volume_flux  = (flux_hindenlang_gassner, flux_nonconservative_powell)
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

coordinates_min = (0.0, 0.0)
coordinates_max = (2*pi, 2*pi)


###############################################################################
# adapt the parameters that have changed compared to "elixir_advection_extended.jl"

# Note: If you get a restart file from somewhere else, you need to provide
# appropriate setups in the elixir loading a restart file

restart_file = "restart_000007.h5"

restart_filename = joinpath("out", restart_file)
mesh = load_mesh(restart_filename, n_cells_max=100000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver)

tspan = (load_time(restart_filename), 2.0)
dt = load_dt(restart_filename)
ode = semidiscretize(semi, tspan, restart_filename; split_form = false);

summary_callback = SummaryCallback()

analysis_interval = 200000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=false,
                                          variable=density_pressure)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=3,
                                      med_level =7, med_threshold=0.04,
                                      max_level =9, max_threshold=0.4)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=10, # PERK, DGLDDRK73_C
                           #interval=31, # SSPRK33
                           #interval = 15, # ParsaniKetchesonDeconinck3S53
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)


cfl = 1.9 # p = 2, S = 12
cfl = 1.9 # p = 3, S = 10
cfl = 1.9 # p = 3, S = 6
cfl = 1.9 # p = 3, S = 8

stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        amr_callback,
                        stepsize_callback,
                        save_restart,
                        glm_speed_callback)

#=
integrator = init(ode, CarpenterKennedy2N54(williamson_condition=false),
                  dt=dt, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep=false, callback=callbacks);

# Get the last time index and work with that.
integrator.iter = load_timestep(restart_filename)
integrator.stats.naccept = integrator.iter


###############################################################################
# run the simulation

sol = solve!(integrator)
=#

Stages = [6, 4, 3]
#Stages = [8, 5, 4]

cS2 = 1.0
ode_algorithm = PERK3_Multi(Stages, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p3/", cS2)

sol = Trixi.solve(ode, ode_algorithm, dt = dt,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
