
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

# TODO: parabolic; unify names of these accessor functions
prandtl_number() = 0.72
mu() = 6.25e-4 # equivalent to Re = 1600

equations = CompressibleEulerEquations3D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu=mu(),
                                                          Prandtl=prandtl_number())

"""
    initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations3D)

The classical viscous Taylor-Green vortex, as found for instance in

- Jonathan R. Bull and Antony Jameson
  Simulation of the Compressible Taylor Green Vortex using High-Order Flux Reconstruction Schemes
  [DOI: 10.2514/6.2014-3210](https://doi.org/10.2514/6.2014-3210)
"""
# See also https://www.sciencedirect.com/science/article/pii/S187775031630299X?via%3Dihub#sec0070
function initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations3D)
  A  = 1.0 # magnitude of speed
  Ms = 0.1 # maximum Mach number

  rho = 1.0
  v1  =  A * sin(x[1]) * cos(x[2]) * cos(x[3])
  v2  = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
  v3  = 0.0
  p   = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
  p   = p + 1.0/16.0 * A^2 * rho * (cos(2*x[1])*cos(2*x[3]) + 2*cos(2*x[2]) + 2*cos(2*x[1]) + cos(2*x[2])*cos(2*x[3]))

  return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_taylor_green_vortex

volume_flux = flux_ranocha
solver = DGSEM(polydeg=2, surface_flux=flux_hlle,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0, -1.0) .* pi
coordinates_max = ( 1.0,  1.0,  1.0) .* pi
InitialRefinement = 4 # For real run
#InitialRefinement = 2 # For tests
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=InitialRefinement,
                n_cells_max=2500000)

#=
trees_per_dimension = (1, 1, 1)

mesh = P4estMesh(trees_per_dimension, polydeg=2,
                  coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                  periodicity=(true, true, true), initial_refinement_level=4)
=#

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver)
        
#=
semi = SemidiscretizationHyperbolic(mesh, equations,
                                             initial_condition, solver)
=#
###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan; split_form = false) # PERK
#ode = semidiscretize(semi, tspan) # For ODE integrators

#=
restart_file = "restart_000600.h5"
restart_filename = joinpath("out", restart_file)
mesh = load_mesh(restart_filename)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver)

tspan = (load_time(restart_filename), 2.0)
dt = load_dt(restart_filename)
ode = semidiscretize(semi, tspan, restart_filename; split_form = false);
=#

summary_callback = SummaryCallback()

analysis_interval = 20

analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals=(energy_kinetic,
                                     #energy_internal,
                                     enstrophy))
#=
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     analysis_errors = Symbol[])
=#

save_restart = SaveRestartCallback(interval=600,
                                   save_final_restart=true)

amr_indicator = IndicatorLöhner(semi,
                                variable=Trixi.enstrophy)
amr_indicator = IndicatorMax(semi, variable = Trixi.enstrophy)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=InitialRefinement,
                                      med_level =InitialRefinement+1, med_threshold=5.0, # 10 for this gives good results
                                      max_level =InitialRefinement+2, max_threshold=30.0)

amr_callback = AMRCallback(semi, amr_controller,
                           #interval=20, # PERK
                           interval = 18, #DGLDDRK73_C, PERK single
                           #interval=34, # RDPK3SpFSAL35
                           #interval = 30, # ParsaniKetchesonDeconinck3S53
                           #interval = 53, # SSPRK33
                           adapt_initial_condition=false,
                           adapt_initial_condition_only_refine=true)

#stepsize_callback = StepsizeCallback(cfl=4.1) # PERK 3, 4, 6
stepsize_callback = StepsizeCallback(cfl=4.4) # PERK 6
#stepsize_callback = StepsizeCallback(cfl=4.4) # DGLDDRK73_C
#stepsize_callback = StepsizeCallback(cfl=2.6) # ParsaniKetchesonDeconinck3S53
#stepsize_callback = StepsizeCallback(cfl=1.5) # SSPRK33

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        #save_restart,
                        amr_callback)

###############################################################################
# run the simulation

# S = 3, p = 3, Ref refinment: 2
dt = 0.018749952315556585

Stages = [11, 6, 4, 3]
Stages = [6, 4, 3] # Have three levels anyway

cS2 = 1.0
ode_algorithm = PERK3_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/3D_TGV/")
ode_algorithm = PERK3(6, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/3D_TGV/")


sol = Trixi.solve(ode, ode_algorithm, dt = dt,
                  save_everystep=false, callback=callbacks);

#=
sol = solve(ode, DGLDDRK73_C(;thread = OrdinaryDiffEq.True());
            dt = 1.0,
            ode_default_options()..., callback=callbacks)
=#

#=
sol = solve(ode, ParsaniKetchesonDeconinck3S53(;thread = OrdinaryDiffEq.True());
            dt = 1.0,
            ode_default_options()..., callback=callbacks)
=#

sol = solve(ode, SSPRK33(;thread = OrdinaryDiffEq.True());
            dt = 1.0,
            ode_default_options()..., callback=callbacks);

callbacksDE = CallbackSet(summary_callback,
                  analysis_callback,
                  amr_callback)                  
time_int_tol = 1e-4
sol = solve(ode, RDPK3SpFSAL35(;thread = OrdinaryDiffEq.True()); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacksDE)


summary_callback() # print the timer summary
plot(sol)

pd = PlotData2D(sol)
plot(pd["p"])
plot!(getmesh(pd))