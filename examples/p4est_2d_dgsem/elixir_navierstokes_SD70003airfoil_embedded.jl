using OrdinaryDiffEq
using Trixi
using Convex, ECOS
using NLsolve
###############################################################################
# semidiscretization of the compressible Euler equations
U_inf = 0.2
c_inf = 1.0
rho_inf = 1.4 # with gamma = 1.4 => p_inf = 1.0
Re = 10000.0
airfoil_cord_length = 1.0
t_c = airfoil_cord_length / U_inf
aoa = 4 * pi / 180
u_x = U_inf * cos(aoa)
u_y = U_inf * sin(aoa)
gamma = 1.4
prandtl_number() = 0.72
mu() = rho_inf * U_inf * airfoil_cord_length / Re
equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())
@inline function initial_condition_mach02_flow(x, t, equations)
    # set the freestream flow parameters
    rho_freestream = 1.4
    v1 = 0.19951281005196486 # 0.2 * cos(aoa)
    v2 = 0.01395129474882506 # 0.2 * sin(aoa)
    p_freestream = 1.0
    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_mach02_flow
# Boundary conditions for free-stream testing
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)
velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)
polydeg = 3
surf_flux = flux_hllc
vol_flux = flux_chandrashekar
solver = DGSEM(polydeg = polydeg, surface_flux = surf_flux,
               volume_integral = VolumeIntegralFluxDifferencing(vol_flux))
###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)
path = 
mesh_file = "examples/p4est_2d_dgsem/sd7003_laminar_straight_sided_Trixi.inp"
boundary_symbols = [:Airfoil, :FarField]
mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols,
                    initial_refinement_level = 0)
boundary_conditions = Dict(:FarField => boundary_condition_free_stream,
                           :Airfoil => boundary_condition_slip_wall)
boundary_conditions_parabolic = Dict(:FarField => boundary_condition_free_stream,
                                     :Airfoil => boundary_condition_airfoil)
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))
###############################################################################
# ODE solvers, callbacks etc.
# CARE: This might even be too long to run on a Laptop in reasonable time
tspan = (0.0, 0.1 * t_c) # Try to get into a state where initial pressure wave is gone
ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
# You can expect CFL around this value for p = 3, S = 12 (not tested)
#CFL = 6.0
#stepsize_callback = StepsizeCallback(cfl = CFL)
# For plots etc
save_solution = SaveSolutionCallback(interval = 1_000_000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out")
alive_callback = AliveCallback(alive_interval = 20)
callbacks = CallbackSet(analysis_callback,
                        alive_callback,
                        save_solution,
                        summary_callback);
###############################################################################
# run the simulation
path = "examples/p4est_2d_dgsem/Eigenvalues_SD7003_SampleCase.txt"

# Read the eigenvalues from the .txt file as a Vector{ComplexF64}
eig_vals_from_file = readdlm(path, ComplexF64)

# Ensure the data is in the correct format
eig_vals_vector = vec(eig_vals_from_file)

ode_algorithm = Trixi.EmbeddedPairedExplicitRK2(16, eig_vals_vector)
controller = Trixi.PIDController(0.60, -0.33, 0) # Intiialize the controller

ol = Trixi.solve(ode, ode_algorithm,
                  dt = 1.0, # Manual time step value, will be overwritten by the stepsize_callback when it is specified.
                  save_everystep = false, callback = callbacks, controller = controller, abstol = 1e-3, reltol = 1e-3);

summary_callback() # print the timer summary
