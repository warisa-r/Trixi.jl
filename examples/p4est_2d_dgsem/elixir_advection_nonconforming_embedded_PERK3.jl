
using OrdinaryDiffEq
using Trixi
using Convex, ECOS
using NLsolve

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 4 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-1.0, s - 1.0)
f2(s) = SVector(1.0, s + 1.0)
f3(s) = SVector(s, -1.0 + sin(0.5 * pi * s))
f4(s) = SVector(s, 1.0 + sin(0.5 * pi * s))

# Create P4estMesh with 3 x 2 trees and 6 x 4 elements,
# approximate the geometry with a smaller polydeg for testing.
trees_per_dimension = (3, 2)
mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 faces = (f1, f2, f3, f4),
                 initial_refinement_level = 1)

# Refine bottom left quadrant of each tree to level 4
function refine_fn(p4est, which_tree, quadrant)
    quadrant_obj = unsafe_load(quadrant)
    if quadrant_obj.x == 0 && quadrant_obj.y == 0 && quadrant_obj.level < 4
        # return true (refine)
        return Cint(1)
    else
        # return false (don't refine)
        return Cint(0)
    end
end

# Refine recursively until each bottom left quadrant of a tree has level 4
# The mesh will be rebalanced before the simulation starts
refine_fn_c = @cfunction(refine_fn, Cint,
                         (Ptr{Trixi.p4est_t}, Ptr{Trixi.p4est_topidx_t},
                          Ptr{Trixi.p4est_quadrant_t}))
Trixi.refine_p4est!(mesh.p4est, true, refine_fn_c, C_NULL)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 0.2
tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

ode_algorithm = Trixi.EmbeddedPairedExplicitRK3(10, 10, tspan, semi)

controller = Trixi.PIDController(0.60, -0.33, 0) # Intiialize the controller 

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = Trixi.solve(ode, ode_algorithm,
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks, controller = controller); # Stats: (naccept = 518, nreject = 1)

#ode_algorithm = Trixi.PairedExplicitRK3(10, tspan, semi)

# For Paired Explicit Runge-Kutta methods, we receive an optimized timestep for a given reference semidiscretization.
# To allow for e.g. adaptivity, we reverse-engineer the corresponding CFL number to make it available during the simulation.
#cfl_number = Trixi.calculate_cfl(ode_algorithm, ode)
#stepsize_callback = StepsizeCallback(cfl = cfl_number)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
#callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

#sol = Trixi.solve(ode, ode_algorithm,
#            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#            save_everystep = false, callback = callbacks);