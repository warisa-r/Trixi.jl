
using OrdinaryDiffEq
using Trixi
using Plots

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_weak_blast_wave

# BCs must be passed as Dict
boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:all => boundary_condition)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-1.0, s - 1.0)
f2(s) = SVector(1.0, s + 1.0)
f3(s) = SVector(s, -1.0 + sin(0.5 * pi * s))
f4(s) = SVector(s, 1.0 + sin(0.5 * pi * s))
faces = (f1, f2, f3, f4)

Trixi.validate_faces(faces)
mapping_flag = Trixi.transfinite_mapping(faces)

# Get the uncurved mesh from a file (downloads the file if not available locally)
# Unstructured mesh with 24 cells of the square domain [-1, 1]^n
mesh_file = Trixi.download("https://gist.githubusercontent.com/efaulhaber/63ff2ea224409e55ee8423b3a33e316a/raw/7db58af7446d1479753ae718930741c47a3b79b7/square_unstructured_2.inp",
                           joinpath(@__DIR__, "square_unstructured_2.inp"))

mesh = P4estMesh{2}(mesh_file, polydeg = 3,
                    mapping = mapping_flag,
                    initial_refinement_level = 1)

# Refine bottom left quadrant of each tree to level 2
function refine_fn(p4est, which_tree, quadrant)
    quadrant_obj = unsafe_load(quadrant)
    if quadrant_obj.x == 0 && quadrant_obj.y == 0 && quadrant_obj.level < 2
        # return true (refine)
        return Cint(1)
    else
        # return false (don't refine)
        return Cint(0)
    end
end

# Refine recursively until each bottom left quadrant of a tree has level 2
# The mesh will be rebalanced before the simulation starts
refine_fn_c = @cfunction(refine_fn, Cint,
                         (Ptr{Trixi.p4est_t}, Ptr{Trixi.p4est_topidx_t},
                          Ptr{Trixi.p4est_quadrant_t}))
Trixi.refine_p4est!(mesh.p4est, true, refine_fn_c, C_NULL)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

ode_algorithm = Trixi.PairedExplicitRK2(10, tspan, semi)
cfl_number = Trixi.calculate_cfl(ode_algorithm, ode)
stepsize_callback = StepsizeCallback(cfl = 0.7 * cfl_number)

ode_algorithm = Trixi.PairedExplicitRK2(10, tspan, semi)
controller = Trixi.PIDController(0.60, -0.33, 0) # Intiialize the controller 

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution, stepsize_callback)
###############################################################################
# run the simulation

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks, controller = controller, abstol = 1e-1, reltol = 1e-1);
summary_callback() # print the timer summary


ode_algorithm_cfl = Trixi.PairedExplicitRK2(10, tspan, semi)
ode_algorithm_embedded = Trixi.EmbeddedPairedExplicitRK2(10, tspan, semi)
cfl_number = Trixi.calculate_cfl(ode_algorithm, ode)
stepsize_callback = StepsizeCallback(cfl = 0.7 * cfl_number)

# Tolerances to test
tolerances = 10.0 .^ (-7:1:-1)

# Arrays to store errors for both cases
errors_embedded_callback = Float64[]
errors_cfl_callback = Float64[]

# Arrays to store num_rhs for both case
nums_rhs_embedded = []
nums_rhs_cfl = fill(41, length(tolerances))

# Define callbacks (reuse from earlier setup)
callbacks_embedded = CallbackSet(summary_callback, alive_callback, save_solution,
analysis_callback)
callbacks_cfl = CallbackSet(summary_callback, alive_callback, save_solution,
   analysis_callback, stepsize_callback)

# Loop over each tolerance value
for tol in tolerances

    # Solve with the StepsizeCallback
    num_rhs_embedded, sol_embedded = Trixi.solve(ode, ode_algorithm_embedded, dt = 1.0, # Manual time step value, will be overwritten by the stepsize_callback when it is specified.
                                save_everystep = false, callback = callbacks_embedded, controller = controller, abstol = tol, reltol = tol);
    results_embedded = analysis_callback(sol_embedded)
    push!(errors_embedded_callback, results_embedded.l2[1]) # Assuming l2[1] stores the error
    push!(nums_rhs_embedded, num_rhs_embedded)

    # Solve without the StepsizeCallback
    sol_cfl = Trixi.solve(ode, ode_algorithm_cfl, dt = 1.0, # Manual time step value, will be overwritten by the stepsize_callback when it is specified.
                          save_everystep = false, callback = callbacks_cfl);
    results_cfl = analysis_callback(sol_cfl)
    push!(errors_cfl_callback, results_cfl.l2[1]) # Assuming l2[1] stores the error
end

round_cfl = round( 0.7 * cfl_number, digits=3)

# Plot the error against tolerances
plot(tolerances, errors_embedded_callback, xscale = :log10,
     marker = :circle, label = "PERK21", color = :blue,
     xlabel = "Tolerances (abstol = reltol)", ylabel = "L2 Error",
     title = "Error (rho) vs. Tolerance from weak blast test case (non-uniform grid)")
plot!(tolerances, errors_cfl_callback, marker = :square, color = :red,
         label = "CFL = $round_cfl")
plot!(size = (1000, 800))

savefig("plot_l2_error_weakblast_PERK21.png")

# Plot the error against tolerances
plot(tolerances, nums_rhs_embedded, xscale = :log10,
     marker = :circle, label = "PERK21", color = :blue,
     xlabel = "Tolerances (abstol = reltol)", ylabel = "Number of RHS evaluation",
     title = "Number of RHS evaluation vs. Tolerance from weak blast test case (non-uniform grid)")
plot!(tolerances, nums_rhs_cfl, marker = :square, color = :red,
         label = "CFL = $round_cfl")

plot!(size = (1000, 800))

savefig("plot_num_rhs_weakblast_PERK21.png")
