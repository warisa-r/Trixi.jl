
using OrdinaryDiffEq, Plots
using Trixi

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
equations_parabolic = ViscoResistiveMhd2D(equations, mu = mu(),
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

#volume_flux  = (flux_central, flux_nonconservative_powell)
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
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=100000)


semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan; split_form = false)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=10000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

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
                           #interval=10,
                           interval=31, # SSPRK33
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)


cfl = 1.9 # p = 2, S = 12
cfl = 1.9 # p = 3, S = 11

stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        #save_solution,
                        amr_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

# S = 3, p = 2 Ref = 4
dt = 0.0161709425439767083

# S = 4, p = 2 Ref = 4
#dt = 0.0274786205467535198

LevelCFL = Dict([(42, 42.0)])
Integrator_Mesh_Level_Dict = Dict([(42, 42)])
b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS

# TODO: Do also p=3

# S = 4, p = 3, Ref = 4
dt = 0.0155320306832436481
# Series: 4, 6, 11

#=
ode_algorithm = PERK_Multi(3, 2, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p2/",
                           bS, cEnd,
                           LevelCFL, Integrator_Mesh_Level_Dict)


ode_algorithm = PERK(12, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p2/",
                     bS, cEnd)
=#

#=
Stages = [11, 6, 4]

cS2 = 1.0
ode_algorithm = PERK3_Multi(Stages, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p3/", cS2,
                            LevelCFL, Integrator_Mesh_Level_Dict)

ode_algorithm = PERK3(11, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p3/")                            

sol = Trixi.solve(ode, ode_algorithm, dt = dt,
                  save_everystep=false, callback=callbacks);
=#

#cfl = 1.9 # DGLDDRK73_C Max Level 9, base lvl = 3
cfl = 0.6 # SSPRK33 Max Level 9, base lvl = 3

stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        amr_callback,
                        stepsize_callback,
                        glm_speed_callback)

sol = solve(ode, SSPRK33();
            dt = 1.0,
            ode_default_options()..., callback=callbacks,
            thread = OrdinaryDiffEq.True());

for i = 1:10
  solver = DGSEM(basis, surface_flux, volume_integral)

  coordinates_min = (0.0, 0.0)
  coordinates_max = (2*pi, 2*pi)
  mesh = TreeMesh(coordinates_min, coordinates_max,
                  initial_refinement_level=4,
                  n_cells_max=100000)
  
  
  semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver) 
  
  ode = semidiscretize(semi, tspan; split_form = false)

  sol = solve(ode, DGLDDRK73_C();
              dt = 1.0,
              ode_default_options()..., callback=callbacks,
              thread = OrdinaryDiffEq.True());
end


summary_callback() # print the timer summary
plot(sol)

pd = PlotData2D(sol)
plot(pd["rho"])
plot(pd["p"])
plot(pd["B1"])
plot!(getmesh(pd))