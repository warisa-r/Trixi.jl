
using OrdinaryDiffEq, Plots
using Trixi


###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
equations = IdealGlmMhdEquations2D(1.4)

"""
    initial_condition_rotor(x, t, equations::IdealGlmMhdEquations2D)

The classical MHD rotor test case. Here, the setup is taken from
- Dominik Derigs, Gregor J. Gassner, Stefanie Walch & Andrew R. Winters (2018)
  Entropy Stable Finite Volume Approximations for Ideal Magnetohydrodynamics
  [doi: 10.1365/s13291-018-0178-9](https://doi.org/10.1365/s13291-018-0178-9)
"""
function initial_condition_rotor(x, t, equations::IdealGlmMhdEquations2D)
  # setup taken from Derigs et al. DMV article (2018)
  # domain must be [0, 1] x [0, 1], γ = 1.4
  dx = x[1] - 0.5
  dy = x[2] - 0.5
  r = sqrt(dx^2 + dy^2)
  f = (0.115 - r)/0.015
  if r <= 0.1
    rho = 10.0
    v1 = -20.0*dy
    v2 = 20.0*dx
  elseif r >= 0.115
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
  else
    rho = 1.0 + 9.0*f
    v1 = -20.0*f*dy
    v2 = 20.0*f*dx
  end
  v3 = 0.0
  p = 1.0
  B1 = 5.0/sqrt(4.0*pi)
  B2 = 0.0
  B3 = 0.0
  psi = 0.0
  return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end
initial_condition = initial_condition_rotor

# Paper mentioned above uses outflow BCs
function boundary_condition_outflow(u_inner, orientation, direction, x, t,
                                    surface_flux_function,
                                    equations::IdealGlmMhdEquations2D)

  return surface_flux_function(u_inner, u_inner, orientation, equations)
end

boundary_conditions = (x_neg=boundary_condition_outflow,
                       x_pos=boundary_condition_outflow,
                       y_neg=boundary_condition_outflow,
                       y_pos=boundary_condition_outflow)

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
volume_flux  = (flux_hindenlang_gassner, flux_nonconservative_powell)
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
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
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000,
                periodicity=false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.15)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=false,
                                          variable=density_pressure)
# For density_pressure                                          
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=3,
                                      med_level =7, med_threshold=0.0041,
                                      max_level =9, max_threshold=0.25)                                   
amr_callback = AMRCallback(semi, amr_controller,
                           #interval=40, # PERK, DGLDDRK73_C
                           #interval=40*26, # SSPRK33
                            interval = 10,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

cfl = 0.03 # SSPRK33
#cfl = 0.8 # DGLDDRK73_C
#cfl = 0.82 # S = 10, AMR, PERK
#cfl = 0.8 # S = 10, AMR, PERK Single
cfl = 0.7 # 3,4,6 PERK
cfl = 0.2

stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.8, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        amr_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

# p = 3, BaseRef = 3

# S = 3
dt = 0.0080186414951458575

# S = 4
dt = 0.019130091787519633

# S = 6
dt = 0.0509158711698546568

# S = 8
dt = 0.082571630552411084

# S = 10
dt = 0.116699115931260173

# S = 12
#dt = 0.14293596705974779

# For InitialRefinement = 4
dt *= 0.5 

#Stages = [10, 6, 4, 3]
Stages = [6, 4, 3]


LevelCFL = Dict([(42, 42.0)])
Integrator_Mesh_Level_Dict = Dict([(42, 42)])

cS2 = 1.0
ode_algorithm = PERK3_Multi(Stages, "/home/daniel/git/Paper_AMR_PERK/Data/MHD_Rotor/", cS2,
                            LevelCFL, Integrator_Mesh_Level_Dict)

#ode_algorithm = PERK3(10, "/home/daniel/git/Paper_AMR_PERK/Data/MHD_Rotor/")


for i = 1:1
    mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000,
                periodicity=false)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions=boundary_conditions)

    ode = semidiscretize(semi, tspan)

    #=
    sol = Trixi.solve(ode, ode_algorithm,
                    dt = dt,
                    save_everystep=false, callback=callbacks);
    =#
    
    #=
    sol = solve(ode, SSPRK33(;thread = OrdinaryDiffEq.True());
                dt=dt,
                save_everystep=false, callback=callbacks,
                ode_default_options()...);
    =#
    #=
    sol = solve(ode, ParsaniKetchesonDeconinck3S53(;thread = OrdinaryDiffEq.True());
                dt = 1.0,
                ode_default_options()..., callback=callbacks)
    =#
    
    sol = solve(ode, DGLDDRK73_C(;thread = OrdinaryDiffEq.True());
                dt = 1.0,
                ode_default_options()..., callback=callbacks)
    
end

summary_callback() # print the timer summary


plot(sol)

pd = PlotData2D(sol)
plot(pd["rho"])
plot(pd["p"], title = "\$ p, t_f = 0.15 \$")
plot(getmesh(pd), xlabel = "\$x\$", ylabel="\$y\$", title = "Mesh at \$t_f = 0.15\$")
