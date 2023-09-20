using OrdinaryDiffEq, LinearAlgebra, Plots
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

equations = CompressibleEulerEquations2D(1.4)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=2, surface_flux=flux_hllc,
               volume_integral=VolumeIntegralWeakForm())

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
InitialRefinement = 4
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=InitialRefinement,
                periodicity=(true, false),
                n_cells_max=30_000) # set maximum capacity of tree data structure
#=
LLID = Trixi.local_leaf_cells(mesh.tree)
Trixi.refine!(mesh.tree, LLID[1:128])
LLID = Trixi.local_leaf_cells(mesh.tree)
Trixi.refine!(mesh.tree, LLID[1:128])
LLID = Trixi.local_leaf_cells(mesh.tree)
Trixi.refine!(mesh.tree, LLID[1:128])
=#

# Note: the initial condition cannot be specialized to `CompressibleNavierStokesDiffusion2D`
#       since it is called by both the parabolic solver (which passes in `CompressibleNavierStokesDiffusion2D`)
#       and by the initial condition (which passes in `CompressibleEulerEquations2D`).
# This convergence test setup was originally derived by Andrew Winters (@andrewwinters5000)
function initial_condition_navier_stokes_convergence_test(x, t, equations)
  # Amplitude and shift
  A = 0.5
  c = 2.0

  # convenience values for trig. functions
  pi_x = pi * x[1]
  pi_y = pi * x[2]
  pi_t = pi * t

  rho = c + A * sin(pi_x) * cos(pi_y) * cos(pi_t)
  v1  = sin(pi_x) * log(x[2] + 2.0) * (1.0 - exp(-A * (x[2] - 1.0)) ) * cos(pi_t)
  v2  = v1
  p   = rho^2

  return prim2cons(SVector(rho, v1, v2, p), equations)
end

@inline function source_terms_navier_stokes_convergence_test(u, x, t, equations)
  y = x[2]

  # TODO: parabolic
  # we currently need to hardcode these parameters until we fix the "combined equation" issue
  # see also https://github.com/trixi-framework/Trixi.jl/pull/1160
  inv_gamma_minus_one = inv(equations.gamma - 1)

  # Same settings as in `initial_condition`
  # Amplitude and shift
  A = 0.5
  c = 2.0

  # convenience values for trig. functions
  pi_x = pi * x[1]
  pi_y = pi * x[2]
  pi_t = pi * t

  # compute the manufactured solution and all necessary derivatives
  rho    =  c  + A * sin(pi_x) * cos(pi_y) * cos(pi_t)
  rho_t  = -pi * A * sin(pi_x) * cos(pi_y) * sin(pi_t)
  rho_x  =  pi * A * cos(pi_x) * cos(pi_y) * cos(pi_t)
  rho_y  = -pi * A * sin(pi_x) * sin(pi_y) * cos(pi_t)


  v1    =       sin(pi_x) * log(y + 2.0) * (1.0 - exp(-A * (y - 1.0))) * cos(pi_t)
  v1_t  = -pi * sin(pi_x) * log(y + 2.0) * (1.0 - exp(-A * (y - 1.0))) * sin(pi_t)
  v1_x  =  pi * cos(pi_x) * log(y + 2.0) * (1.0 - exp(-A * (y - 1.0))) * cos(pi_t)
  v1_y  =       sin(pi_x) * (A * log(y + 2.0) * exp(-A * (y - 1.0)) + (1.0 - exp(-A * (y - 1.0))) / (y + 2.0)) * cos(pi_t)

  v2    = v1
  v2_t  = v1_t
  v2_x  = v1_x
  v2_y  = v1_y

  p    = rho * rho
  p_t  = 2.0 * rho * rho_t
  p_x  = 2.0 * rho * rho_x
  p_y  = 2.0 * rho * rho_y

  # Note this simplifies slightly because the ansatz assumes that v1 = v2
  E   = p * inv_gamma_minus_one + 0.5 * rho * (v1^2 + v2^2)
  E_t = p_t * inv_gamma_minus_one + rho_t * v1^2 + 2.0 * rho * v1 * v1_t
  E_x = p_x * inv_gamma_minus_one + rho_x * v1^2 + 2.0 * rho * v1 * v1_x
  E_y = p_y * inv_gamma_minus_one + rho_y * v1^2 + 2.0 * rho * v1 * v1_y

  # compute the source terms
  # density equation
  du1 = rho_t + rho_x * v1 + rho * v1_x + rho_y * v2 + rho * v2_y

  # x-momentum equation
  du2 = ( rho_t * v1 + rho * v1_t + p_x + rho_x * v1^2
                                        + 2.0   * rho  * v1 * v1_x
                                        + rho_y * v1   * v2
                                        + rho   * v1_y * v2
                                        + rho   * v1   * v2_y)
  # y-momentum equation
  du3 = ( rho_t * v2 + rho * v2_t + p_y + rho_x * v1    * v2
                                        + rho   * v1_x  * v2
                                        + rho   * v1    * v2_x
                                        +         rho_y * v2^2
                                        + 2.0   * rho   * v2 * v2_y)
  # total energy equation
  du4 = ( E_t + v1_x * (E + p) + v1 * (E_x + p_x)
              + v2_y * (E + p) + v2 * (E_y + p_y))

  return SVector(du1, du2, du3, du4)
end

initial_condition = initial_condition_navier_stokes_convergence_test

# define inviscid boundary conditions
boundary_conditions = (; x_neg = boundary_condition_periodic,
                         x_pos = boundary_condition_periodic,
                         y_neg = boundary_condition_slip_wall,
                         y_pos = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions=boundary_conditions, 
                                    source_terms=source_terms_navier_stokes_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 5)
ode = semidiscretize(semi, tspan)
#ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=100)
analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
#callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

amr_indicator = IndicatorMax(semi, variable=Trixi.density)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = InitialRefinement,
                                      med_level  = InitialRefinement+1, med_threshold=2.0,
                                      max_level  = InitialRefinement+2, max_threshold=2.3)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=10,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = CallbackSet(summary_callback, analysis_callback, amr_callback)

###############################################################################
# run the simulation

#=
stepsize_callback = StepsizeCallback(cfl=1.0)
callbacksCK = CallbackSet(summary_callback, analysis_callback, stepsize_callback, amr_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacksCK);
=#

#=
time_int_tol = 1e-6
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol, dt = 1e-5,
            ode_default_options()..., callback=callbacks)
=#


# mu = 1e-5, HLLC flux, non-adapted, finer mesh
CFL = 0.6 # Two refinements
CFL = 0.43 # Three
dt = 0.0319591159226547479 / (2.0^(InitialRefinement - 4)) * CFL


#=
# mu = 1e-5, HLLC flux, adapted
CFL = 1.2 # Three levels
CFL = 0.75 # Four levels
dt = 0.0364957930534728823 / (2.0^(InitialRefinement - 3)) * CFL

# Base Level: 3
dt = 0.0472580105059023507 / (2.0^(InitialRefinement - 3)) * CFL
=#

b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS
ode_algorithm = PERK_Multi(4, 3, #"/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_NavierStokes_Convergence/Adapted/", 
                           "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_NavierStokes_Convergence/NonAdapted/", 
                           bS, cEnd, stage_callbacks = ())


# S = 4             
CFL_Single = 0.25 * CFL
# dt for adapted spectrum
dt = 0.0319591159226547479 / (2.0^(InitialRefinement - 4)) * CFL_Single
S = 4


#=
# S = 8                   
CFL_Single = 0.25 * CFL
# dt for adapted spectrum
dt = 0.0705015182429633608 / (2.0^(InitialRefinement - 4)) * CFL_Single
S = 8
=#

#=
# S = 16                
CFL_Single = 0.25 * CFL
# dt for adapted spectrum
dt = 0.143722531198727673 / (2.0^(InitialRefinement - 4)) * CFL_Single
S = 16
=#

#=
# S = 32             
CFL_Single = 0.125 * 0.63
# dt for adapted spectrum
dt = 0.28912970427227858 / (2.0^(InitialRefinement - 4)) * CFL_Single
S = 32
=#

ode_algorithm = PERK(S, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_NavierStokes_Convergence/NonAdapted/", bS, cEnd)


sol = Trixi.solve(ode, ode_algorithm, dt = dt, save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

plot(sol)
pd = PlotData2D(sol)
plot(pd["rho"])
plot!(getmesh(pd))