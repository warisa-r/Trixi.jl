
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
ode = semidiscretize(semi, tspan; split_form = false)
#ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     analysis_errors = Symbol[],
                                     extra_analysis_integrals=(energy_kinetic,
                                     energy_internal,
                                     enstrophy))

analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     analysis_errors = Symbol[])

amr_indicator = IndicatorLöhner(semi,
                                variable=Trixi.v3)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=InitialRefinement,
                                      med_level =InitialRefinement+1, med_threshold=0.7,
                                      max_level =InitialRefinement+2, max_threshold=0.8)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=20,
                           adapt_initial_condition=false,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=4.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        amr_callback)

###############################################################################
# run the simulation

# S = 3, p = 3, Ref refinment: 2
dt = 0.018749952315556585

Stages = [11, 6, 4, 3]
Stages = [6, 4, 3] # Have three levels anyway

cS2 = 1.0
ode_algorithm = PERK3_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/3D_TGV/")

#ode_algorithm = PERK3(11, "/home/daniel/git/Paper_AMR_PERK/Data/RayleighTaylorInstability/p3/")
#ode_algorithm = PERK3(3, "/home/daniel/git/Paper_AMR_PERK/Data/RayleighTaylorInstability/p3/")

sol = Trixi.solve(ode, ode_algorithm, dt = dt,
                  save_everystep=false, callback=callbacks)

callbacksDE = CallbackSet(summary_callback,
                  analysis_callback,
                  amr_callback)                  
time_int_tol = 1e-5
sol = solve(ode, RDPK3SpFSAL35(;thread = OrdinaryDiffEq.True()); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacksDE)


summary_callback() # print the timer summary
plot(sol)

pd = PlotData2D(sol)
plot(pd["p"])
plot!(getmesh(pd))