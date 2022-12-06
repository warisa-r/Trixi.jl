
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
solver = DGSEM(polydeg=6, surface_flux=surf_flux)

coordinates_min = (-20.0, -20.0)
coordinates_max = ( 20.0,  20.0)
cells_per_dimension = (80, 80)

# Create curved mesh with 16 x 16 elements
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

#tspan = (0.0, 20.0)
tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     extra_analysis_errors=(:conservation_error,),
                                     extra_analysis_integrals=(entropy, energy_total,
                                                               energy_kinetic, energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

# NOTE: The timesteps in the PERK publication where {10, 5, 2.5, 1.25} * 10^{-4}
if surf_flux == flux_lax_friedrichs
  stepsize_callback = StepsizeCallback(cfl=0.36) # this gives roughly 27 * 10^{-4}
elseif surf_flux == flux_hll
  stepsize_callback = StepsizeCallback(cfl=0.71) # this gives roughly 53 * 10^{-4}
end

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation


sol = solve(ode, SSPRK22(),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

#=
CFL = 0.99 / 1

NumStages = 18

NumStageRef = 16
dtRef = 1.08834650475182571

#=
NumStageRef = 8
dtRef = 0.539668011389949243
=#


RefinementOptimizedFor = 3

dtOptMin = dtRef * (NumStages / NumStageRef) / (2.0^(Refinement - RefinementOptimizedFor)) * CFL

ode_algorithm = FE2S(NumStages, "/home/daniel/Desktop/git/MA/EigenspectraGeneration/Spectra/2D_ComprEuler_Vortex/")

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks);
=#


summary_callback() # print the timer summary
plot(sol)