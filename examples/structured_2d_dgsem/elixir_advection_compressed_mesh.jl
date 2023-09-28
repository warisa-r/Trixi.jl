
using OrdinaryDiffEq, Plots, LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (-1.0, 1.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-1.0, 1.0 * (s + 1.0))
f2(s) = SVector( 1.0, 1.0 * (s + 1.0))

k = 1.7
alpha = 0.4
beta = 1 - alpha*2^(k-1)
gamma = beta - 1

f3(s) = SVector(alpha*(s+1)^k + beta*s + gamma, 0)
f4(s) = SVector(alpha*(s+1)^k + beta*s + gamma, 2.0)

cells_per_dimension = (64, 8)

# Create curved mesh with 16 x 16 elements
mesh = StructuredMesh(cells_per_dimension, (f1, f2, f3, f4))

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 5));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.7)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)


###############################################################################
# run the simulation
#=
# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
=#

b1 = 0.0
bS = 1 - b1
cEnd = 0.5/bS

Integrator_Mesh_Level_Dict = Dict([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)])
LevelCFL = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

CFL = 0.04527533522559635 / 0.125
# S = 4, dx = 0.125
dt = 0.0272965401672991008 * CFL

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback)

ode_algorithm = PERK_Multi(4, 6, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_Adv/Structured/",
                           bS, cEnd,
                           LevelCFL, Integrator_Mesh_Level_Dict)

sol = Trixi.solve(ode, ode_algorithm, dt = dt, save_everystep=false, callback=callbacks);


# Print the timer summary
summary_callback()
plot(sol)
pd = PlotData2D(sol)
plot!(getmesh(pd))

#plot(getmesh(pd))