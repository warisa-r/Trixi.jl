using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

prandtl_number() = 0.72
mu() = 1e-4

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

@inline function initial_condition_mach2_flow(x, t, equations)
  # set the freestream flow parameters
  rho_freestream = 1.4
  v1 = 0.2
  v2 = 0.0
  p_freestream = 1.0

  prim = SVector(rho_freestream, v1, v2, p_freestream)
  return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach2_flow

# Boundary conditions for free-stream testing
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

polydeg = 3
volume_flux = flux_ranocha

solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/PERK_mesh/SD7003Laminar/"
mesh_file = path * "sd7003_laminar_straight_sided_Trixi.inp"

boundary_symbols = [:Airfoil, :FarField]
mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)


boundary_conditions = Dict(:FarField => boundary_condition_free_stream,
                           #:Airfoil => boundary_condition_free_stream)
                           :Airfoil => boundary_condition_slip_wall)

boundary_conditions_parabolic = Dict(:FarField => boundary_condition_free_stream,
                                     #:Airfoil => boundary_condition_free_stream)
                                     :Airfoil => boundary_condition_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                     initial_condition, solver;
                                     boundary_conditions = (boundary_conditions,
                                                            boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 3.0)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

ode_algorithm = PERK4(5, "/home/daniel/git/MA/EigenspectraGeneration/2D_CEE_IsentropicVortex/PolyDeg6/")

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacksPERK);

sol = solve(ode, SSPRK104(; thread = OrdinaryDiffEq.True()),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary


using Plots

pd = PlotData2D(sol)
plot(sol)

plot(pd["v1"], xlim = [-1, 2], ylim = [-1, 1])
plot!(getmesh(pd))

plot(getmesh(pd), xlim = [-1, 2], ylim = [-1, 1])