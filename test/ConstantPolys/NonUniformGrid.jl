using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

PolyDegree = 0
numerical_flux = flux_lax_friedrichs
solver = DGSEM(polydeg=PolyDegree, surface_flux=numerical_flux)
               #volume_integral=VolumeIntegralPureLGLFiniteVolume(numerical_flux))

coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0 # maximum coordinate

RefinementLevel = 4
# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                # Start from one cell => Results in 1 + 2 + 4 + 8 + 16 = 2^5 - 1 = 31 cells
                initial_refinement_level=RefinementLevel,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# Manual refinement
LLID = Trixi.local_leaf_cells(mesh.tree)
num_leafs = length(LLID)

# Refine middle of mesh
@assert num_leafs % 4 == 0
Trixi.refine!(mesh.tree, LLID[Int(num_leafs/4)+1 : Int(3*num_leafs/4)])

initial_condition = initial_condition_convergence_test

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

StartTime = 0.0
EndTime = 1.5

# Create ODEProblem
ode = semidiscretize(semi, (StartTime, EndTime));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

Interval = 100
# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=Interval, extra_analysis_errors=(:conservation_error,))


stepsize_callback = StepsizeCallback(cfl=1.0)     
callbacks = CallbackSet(summary_callback, analysis_callback,
                        stepsize_callback)


###############################################################################
# run the simulation


sol = solve(ode, 
            CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

plot(sol)

pd = PlotData1D(sol)
plot!(getmesh(pd))