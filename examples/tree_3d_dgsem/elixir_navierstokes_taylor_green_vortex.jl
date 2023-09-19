
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
InitialRefinement = 3
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=InitialRefinement,
                n_cells_max=100_000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan; split_form = false)

summary_callback = SummaryCallback()

analysis_interval = 200

analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     extra_analysis_integrals=(enstrophy, ))

amr_indicator = IndicatorLöhner(semi,
                                variable=Trixi.v2)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=InitialRefinement,
                                      med_level =InitialRefinement+1, med_threshold=0.7, # med_level = current level
                                      max_level =InitialRefinement+3, max_threshold=0.83)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=10,
                           adapt_initial_condition=false,
                           adapt_initial_condition_only_refine=true)

alive_callback = AliveCallback(analysis_interval=analysis_interval,)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        amr_callback)

###############################################################################
# run the simulation

# S = 3, Ref. Level = 1
S = 3
CFL = 1.0
dt = 0.0326516980654560048 / (2.0^(InitialRefinement - 2)) * CFL

#=
S = 6
CFL = 1.0
dt = 0.0776305753039196105 / (2.0^(InitialRefinement - 2)) * CFL


S = 12
CFL = 0.25
dt = 0.165335757314096554 / (2.0^(InitialRefinement - 2)) * CFL

S = 16
CFL = 1.0
dt = 0.221319526090155705 / (2.0^(InitialRefinement - 2)) * CFL


S = 24
CFL = 0.85
dt = 0.324692290225081 / (2.0^(InitialRefinement - 2)) * CFL
=#

bS = 1.0
cEnd = 0.5

ode_algorithm = PERK(S, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/3D_TGV/", bS, cEnd)

Integrator_Mesh_Level_Dict = Dict([(3, 4), (4, 3), (5, 2), (6, 1)])

#LevelCFL = [1.0, 1.0, 0.7, 0.4]
LevelCFL = [0.4, 0.7, 1.0, 1.0]
ode_algorithm = PERK_Multi(3, 3, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/3D_TGV/", 
                           bS, cEnd, LevelCFL, Integrator_Mesh_Level_Dict, stage_callbacks = ())


sol = Trixi.solve(ode, ode_algorithm, dt = dt, save_everystep=false, callback=callbacks);

Plots.plot(sol)

#=
time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacks)
=#

summary_callback() # print the timer summary
