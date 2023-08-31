
using OrdinaryDiffEq, Plots, LinearAlgebra, DelimitedFiles
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 2e-5
#diffusivity() = 0
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

PolyDeg = 3
surface_flux = flux_lax_friedrichs

solver = DGSEM(polydeg=PolyDeg, surface_flux=surface_flux)

coordinates_min = -5.0 # minimum coordinate
coordinates_max =  5.0 # maximum coordinate

InitialRefinement = 4
# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                # Start from one cell => Results in 1 + 2 + 4 + 8 + 16 = 2^5 - 1 = 31 cells
                initial_refinement_level=InitialRefinement,
                n_cells_max=30_000) # set maximum capacity of tree data structure


# First refinement
# Refine mesh locally 
LLID = Trixi.local_leaf_cells(mesh.tree)
num_leafs = length(LLID)

# Refine right 3 quarters of mesh
@assert num_leafs % 4 == 0
Trixi.refine!(mesh.tree, LLID[Int(num_leafs/4)+1 : num_leafs])


# Second refinement
# Refine mesh locally
LLID = Trixi.local_leaf_cells(mesh.tree)
num_leafs = length(LLID)

@assert num_leafs % 4 == 0
# Refine third quarter to ensure we have only transitions from coarse->medium->fine
Trixi.refine!(mesh.tree, LLID[Int(2*num_leafs/4)+1 : Int(3*num_leafs/4)])

initial_condition = initial_condition_gauss

# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), 
  initial_condition, solver;
  boundary_conditions=(boundary_conditions, boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

CFL = 0.99

# S=4, D=3
dt = 0.0545930727967061102 / (2.0^(InitialRefinement - 4)) * CFL * coordinates_max

StartTime = 0.0
EndTime = 50
EndTime = 0.27


# Create ODEProblem
ode = semidiscretize(semi, (StartTime, EndTime); split_form = false);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=1000, extra_analysis_errors=(:conservation_error,))

callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# run the simulation

b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS

#callbacks_Stage = (PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6,), variables=(Trixi.scalar,)), )

LevelCFL = [1.0, 1.0, 1.0]

ode_algorithm = PERK_Multi(4, 2, "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/",
                           bS, cEnd,
                           LevelCFL,
                           stage_callbacks = ())

           
#ode_algorithm = PERK(16, "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/", bS, cEnd)


sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

#PlotData = plot(sol)
#savefig(PlotData, string(i) * ".png")

plot!(sol)
#scatter(sol)
pd = PlotData1D(sol)
plot!(getmesh(pd))

plot(sol.u[end])

display(norm(sol.u[end] - A_OneStep * sol.u[1], Inf))

