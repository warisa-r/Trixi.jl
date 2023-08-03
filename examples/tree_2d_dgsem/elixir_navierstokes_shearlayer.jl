
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

# TODO: parabolic; unify names of these accessor functions
prandtl_number() = 0.72
mu() = 1.0/3.0 * 10^(-3) # equivalent to Re = 3000

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu=mu(),
                                                          Prandtl=prandtl_number())

function initial_condition_shear_layer(x, t, equations::CompressibleEulerEquations2D)
  k = 80
  delta = 0.05
  u0 = 1.0
  Ms = 0.1 # maximum Mach number

  rho = 1.0
  v1  = x[2] <= 0.5 ? u0 * tanh(k*(x[2]*0.5 - 0.25)) : u0 * tanh(k*(0.75 -x[2]*0.5))
  v2  = u0 * delta * sin(2*pi*(x[1]*0.5 + 0.25))
  p   = (u0 / Ms)^2 * rho / equations.gamma # scaling to get Ms

  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_shear_layer

volume_flux = flux_ranocha
solver = DGSEM(polydeg=3, surface_flux=flux_hllc,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
InitialRefinement = 4
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=100_000)


semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan; split_form = true)

summary_callback = SummaryCallback()

analysis_interval = 50
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     extra_analysis_integrals=(energy_kinetic,
                                                               energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval,)

amr_indicator = IndicatorLöhner(semi, variable=v_x)                                          
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=4,
                                      med_level = 5, med_threshold=0.05,
                                      max_level =6, max_threshold=0.1)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        amr_callback)

###############################################################################
# run the simulation


CFL = 0.5 # b1 = 0
CFL = 0.57 # b1 = 0.5
dt = 0.00320512892678380019  / (2.0^(InitialRefinement - 3)) * CFL

#=
b1   = 0.5
bS   = 1.0 - b1
cEnd = 0.5/bS
ode_algorithm = PERK_Multi(4, 2, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_NavierStokes_ShearLayer/", 
                           bS, cEnd, stage_callbacks = ())

sol = Trixi.solve(ode, ode_algorithm, dt = dt, save_everystep=false, callback=callbacks);
=#

time_int_tol = 1e-6
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacks)


summary_callback() # print the timer summary

plot(sol)
pd = PlotData2D(sol)
plot!(getmesh(pd))