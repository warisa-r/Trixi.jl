
using OrdinaryDiffEq, Plots, LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
PolyDegree = 3
solver = DGSEM(polydeg=PolyDegree, surface_flux=flux_lax_friedrichs)

coordinates_min = -1.0 # minimum coordinate
coordinates_max =  1.0 # maximum coordinate

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
Trixi.refine!(mesh.tree, LLID[Int(num_leafs/4)+1 : end])


# Second refinement
# Refine mesh locally
LLID = Trixi.local_leaf_cells(mesh.tree)
num_leafs = length(LLID)

# Refine third quarter to ensure we have only transitions from coarse->medium->fine
@assert num_leafs % 4 == 0
Trixi.refine!(mesh.tree, LLID[Int(2*num_leafs/4) : Int(3*num_leafs/4)])

#=
# Third refinement
# Refine mesh locally
LLID = Trixi.local_leaf_cells(mesh.tree)
num_leafs = length(LLID)

@assert num_leafs % 4 == 0
Trixi.refine!(mesh.tree, LLID[Int(2*num_leafs/4) : Int(3*num_leafs/4)])
=#

initial_condition = initial_condition_convergence_test
#initial_condition = initial_condition_gauss

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

StartTime = 0.0
#EndTime = 0.0
EndTime = 1

# Create ODEProblem
ode = semidiscretize(semi, (StartTime, EndTime));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100, extra_analysis_errors=(:conservation_error,))

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
#=
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)
=#
# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
#stepsize_callback = StepsizeCallback(cfl=1.0)

#=
amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first),
                                      base_level=4,
                                      med_level=5, med_threshold=0.1,
                                      max_level=6, max_threshold=0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=false) # Adaption of initial condition not yet supported
=#

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
#callbacks = CallbackSet(summary_callback, analysis_callback, amr_callback, stepsize_callback)
#callbacks = CallbackSet(summary_callback, analysis_callback, amr_callback)
callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# run the simulation

#ode_algorithm = Trixi.CarpenterKennedy2N54()

#dtOptMin = 0.05
dtOptMin = 1

#A = jacobian_ad_forward(semi)
A, = linear_structure(semi)
A = Matrix(A)

Eigenvalues = eigvals(Matrix(A))
# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

# Sometimes due to numerical issues some eigenvalues have positive real part, which is erronous (for hyperbolic eqs)
#Eigenvalues = Eigenvalues[real(Eigenvalues) .< 0]

#ode_algorithm = Trixi.FE2S(6, 1, dtOptMin, "/home/daniel/Desktop/git/MA/Optim_Monomials/Matlab/", A)
ode_algorithm = Trixi.PERK(4, 2, 2, dtOptMin, 
                           "/home/daniel/Desktop/git/MA/Optim_Monomials/Matlab/Results/1D_Adv_ConvergenceTest/", 
                           A, Eigenvalues)

#exit()
#ode_algorithm = Trixi.CarpenterKennedy2N54()
# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = Trixi.solve(ode, ode_algorithm,
                  #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  dt = dtOptMin,
                  #dt = ode_algorithm.dtOptMin,
                  save_everystep=false, callback=callbacks);

pd = PlotData1D(sol)
plot(sol)
# Print the timer summary
summary_callback()

plot(getmesh(pd))