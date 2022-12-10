
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

# Ratio of specific heats
gamma = 1.4

equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case as presented in 
https://spectrum.library.concordia.ca/id/eprint/985444/1/Paired-explicit-Runge-Kutta-schemes-for-stiff-sy_2019_Journal-of-Computation.pdf
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
  # Evaluate error after full domain traversion
  if t == 40
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

surf_flux = flux_lax_friedrichs # = Rusanov, originally used
surf_flux = flux_hll # Better flux, allows much larger timesteps
PolyDeg = 6
solver = DGSEM(polydeg=PolyDeg, surface_flux=surf_flux)

coordinates_min = (-20.0, -20.0)
coordinates_max = ( 20.0,  20.0)

Refinement = 6
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=Refinement,
                n_cells_max=30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 40.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     extra_analysis_errors=(:conservation_error,),
                                     extra_analysis_integrals=(entropy, energy_total,
                                                               energy_kinetic, energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

NumCells = 2^Refinement
NumCellsRef = 4


NumBaseStages = 4
NumDoublings = 3

# CARE: For linea increasing PERK
#NumDoublings = 12 # = Num Increases


#=
NumBaseStages = 32
NumDoublings = 0
=#

dtRefBase = 0.255777720361038519 # 4

CFL_Stability = 0.99
CFL_Convergence = 1/7

dtOptMin = dtRefBase / (NumCells/NumCellsRef) * CFL_Stability * CFL_Convergence

ode_algorithm = PERK_Multi(NumBaseStages, NumDoublings, 
                "/home/daniel/Desktop/git/MA/EigenspectraGeneration/Spectra/2D_ComprEuler_Vortex/4Cells/S32/")

#=
ode_algorithm = PERK(NumBaseStages, 
                "/home/daniel/Desktop/git/MA/EigenspectraGeneration/Spectra/2D_ComprEuler_Vortex/4Cells/")
=#

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks);

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = 2.27e-3,
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
plot(sol)

pd = PlotData2D(sol)
plot(pd["rho"])