
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
solver = DGSEM(polydeg=6, surface_flux=surf_flux)

coordinates_min = (-20.0, -20.0)
coordinates_max = ( 20.0,  20.0)
NumCells = 80
cells_per_dimension = (NumCells, NumCells)

mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

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

# NOTE: The timesteps in the PERK publication where {10, 5, 2.5, 1.25} * 10^{-4}
if surf_flux == flux_lax_friedrichs
  stepsize_callback = StepsizeCallback(cfl=0.36) # this gives roughly 27 * 10^{-4}
elseif surf_flux == flux_hll
  stepsize_callback = StepsizeCallback(cfl=0.71) # this gives roughly 53 * 10^{-4}
end

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)
                        #stepsize_callback)


###############################################################################
# run the simulation

#=
# PERK papers use (SSP[?]) RK2 for comparison
sol = solve(ode, SSPRK22(),
            #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            dt = 4.8e-03,
            save_everystep=false, callback=callbacks);
=#

# Testruns: Check if spatial accuracy limits now
#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            dt = 0.75e-2,
            save_everystep=false, callback=callbacks);
=#

# Control case: See if 2x2 is sufficient to estimate 80x80
NumStages = 4
CFL_Opt             = 1.0 # By definition
CFL_Internal        = 1.0
CFL_ConvergenceStudy = 1.4 * 0.83 # 2x2

#=
CFL_ConvergenceStudy = 0.99 # 3x3
CFL_ConvergenceStudy = 0.99 # 4x4
CFL_ConvergenceStudy = 0.99 # 5x5
CFL_ConvergenceStudy = 1.0 # 6x6
CFL_ConvergenceStudy = 1.0 # 7x7
CFL_ConvergenceStudy = 0.99 # 8x8
=#


# 26 
#=
NumStages = 26
CFL_Opt = 0.95535463700870915

CFL_Internal = 0.79 # Lebedev
CFL_Internal = 0.04 # Successive Intermediate classic RK
CFL_Internal = 0.8 # Classic RK

CFL_ConvergenceStudy = 1


# 52 
NumStages = 52
CFL_Opt = 0.93135613167399789

CFL_Internal = 0.7 # Lebedev
CFL_Internal = 0.01 # Successive Intermediate classic RK
CFL_Internal = 0.48 # Classic RK

CFL_ConvergenceStudy = 1
=#


# 104
#=
NumStages = 104
CFL_Opt = 0.91021259292341095

CFL_Internal = 0.28 # Lebedev
CFL_Internal = 0.004 # Successive Intermediate classic RK
CFL_Internal = 0.23 # Classic RK

CFL_ConvergenceStudy = 1
=#

NumStageRef = 16
dtRef = 2.29160435257381323

NumStageRef = 4

NumCellsRef = 2
dtRef = 0.419013938898388005 # 2x2

#=
NumCellsRef = 3
dtRef = 0.338754492229782045 # 3x3

NumCellsRef = 4
dtRef = 0.255777720361038519 # 4x4

NumCellsRef = 5
dtRef = 0.204524988307639433 # 5x5

NumCellsRef = 6
dtRef = 0.168881459738258854 # 6x6

NumCellsRef = 7
dtRef = 0.145751630958147865 # 7x7

NumCellsRef = 8
dtRef = 0.127459716796329298 # 8x8
=#

dtOptMin = dtRef * (NumStages / NumStageRef) / (NumCells/NumCellsRef) * CFL_Opt * CFL_Internal * CFL_ConvergenceStudy

#ode_algorithm = FE2S(NumStages, "/home/daniel/Desktop/git/MA/EigenspectraGeneration/Spectra/2D_ComprEuler_Vortex/")

ode_algorithm = PERK(NumStages, 
                "/home/daniel/Desktop/git/MA/EigenspectraGeneration/Spectra/2D_ComprEuler_Vortex/")

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
plot(sol)

pd = PlotData2D(sol)
plot(pd["rho"])