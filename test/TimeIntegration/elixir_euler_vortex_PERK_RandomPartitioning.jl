
using OrdinaryDiffEq, Plots
using Trixi

# Ratio of specific heats
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

EdgeLength = 10

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case as presented in 
https://spectrum.library.concordia.ca/id/eprint/985444/1/Paired-explicit-Runge-Kutta-schemes-for-stiff-sy_2019_Journal-of-Computation.pdf
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
  # Evaluate error after full domain traversion
  if t == 2*EdgeLength
    t = 0
  end

  # initial center of the vortex
  inicenter = SVector(0.0, 0.0)
  # strength of the vortex
  S = 13.5
  # Radius of vortex
  R = 1.5
  # Free-stream Mach 
  M = 0.4
  # base flow
  v1 = 1.0
  v2 = 1.0
  vel = SVector(v1, v2)

  cent = inicenter + vel*t      # advection of center
  cent = x - cent               # distance to centerpoint
  cent = SVector(cent[2], -cent[1])
  r2 = cent[1]^2 + cent[2]^2

  f = (1 - r2) / (2 * R^2)

  rho = (1 - (S*M/pi)^2 * (gamma - 1)*exp(2*f) / 8)^(1/(gamma - 1))

  du = S/(2*π*R)*exp(f) # vel. perturbation
  vel = vel + du*cent
  v1, v2 = vel

  p = rho^gamma / (gamma * M^2)
  prim = SVector(rho, v1, v2, p)
  return prim2cons(prim, equations)
end
initial_condition = initial_condition_isentropic_vortex

surf_flux = flux_hllc # Better flux, allows much larger timesteps
PolyDeg = 6
solver = DGSEM(polydeg=PolyDeg, surface_flux=surf_flux)


coordinates_min = (-EdgeLength, -EdgeLength)
coordinates_max = ( EdgeLength,  EdgeLength)

Refinement = 6
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=Refinement,
                n_cells_max=100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2*EdgeLength)
#tspan = (0.0, 0) # Find out minimal error (based on initial distribution)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors=(:l1_error,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacksPERK = CallbackSet(summary_callback,
                            analysis_callback, alive_callback)                

###############################################################################
# run the simulation

NumDoublings = 2
Integrator_Mesh_Level_Dict = Dict([(1, 3), (2, 2), (3, 1)])
#Integrator_Mesh_Level_Dict = Dict([(1, 1), (2, 2), (3, 3)])

NumBaseStages = 3
# S = 3, p = 2, d = 2
dtRefBase = 0.259612106506210694
# S = 3, p = 2, d = 6
dtRefBase = 0.0446146026341011772

#=
NumBaseStages = 4
# S = 4, p = 3
dtRefBase = 0.170426237621541077

# S = 4, p = 3, d = 6
dtRefBase = 0.0616218607581686263
=#

CFL_Conv = 0.5

BaseRefinement = 3
dtOptMin = dtRefBase * 2.0^(BaseRefinement - Refinement) * CFL_Conv


LevelCFL = [0.7, 0.7, 0.7]
b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS

ode_algorithm = PERK_Multi(NumBaseStages, NumDoublings, 
                           #"/home/daniel/git/MA/EigenspectraGeneration/2D_CEE_IsentropicVortex/PolyDeg2/",
                           "/home/daniel/git/MA/EigenspectraGeneration/2D_CEE_IsentropicVortex/PolyDeg6/",
                           bS, cEnd,
                           LevelCFL, Integrator_Mesh_Level_Dict,
                           stage_callbacks = ())


#=
cS2 = 1.0 # = c_{S-2}
ode_algorithm = PERK3_Multi(NumBaseStages, NumDoublings, 
                           #"/home/daniel/git/MA/EigenspectraGeneration/2D_CEE_IsentropicVortex/PolyDeg3/",
                           "/home/daniel/git/MA/EigenspectraGeneration/2D_CEE_IsentropicVortex/PolyDeg6/",
                           cS2,
                           LevelCFL, Integrator_Mesh_Level_Dict,
                           stage_callbacks = ())
=#

#=
ode_algorithm = PERK(3, "/home/daniel/git/MA/EigenspectraGeneration/2D_CEE_IsentropicVortex/PolyDeg2/", bS, cEnd)
CFL = 0.8 * 0.25 # S = 12, p = 2
=#

#=
ode_algorithm = PERK3(4, "/home/daniel/git/MA/EigenspectraGeneration/2D_CEE_IsentropicVortex/PolyDeg3/")
CFL = 0.5 * 0.25 # S = 16, p = 3
=#

#dtOptMin = dtRefBase * 2.0^(BaseRefinement - Refinement) * CFL


sol = Trixi.solve(ode, ode_algorithm,
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacksPERK);

summary_callback() # print the timer summary
plot(sol)

pd = PlotData2D(sol)
plot(pd["rho"])
plot!(getmesh(pd))