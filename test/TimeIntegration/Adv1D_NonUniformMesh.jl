
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

# semidiscretization for non-refined mesh (only for plotting eigenvalues)
#=
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)
A, = linear_structure(semi) # Potentially cheaper than jacobian
EigVals = eigvals(Matrix(A))

# Complex conjugate eigenvalues have same modulus
EigVals = EigVals[imag(EigVals) .>= 0]

plotdata = scatter(real.(EigVals), imag.(EigVals), label = "Non-refined discretization")
=#

LLID = Trixi.local_leaf_cells(mesh.tree)
num_leafs = length(LLID)

# Refine only one mesh cell
#Trixi.refine!(mesh.tree, LLID[end])

# Refine right one quarter of mesh
#@assert num_leafs % 4 == 0
#Trixi.refine!(mesh.tree, LLID[Int(3 * num_leafs/4)+1 : end])

# Refine right half of mesh
@assert num_leafs % 2 == 0 "Assume even number of leaf nodes/cells"
Trixi.refine!(mesh.tree, LLID[Int(num_leafs/2)+1 : end])

# Refine right three quarters of mesh
#@assert num_leafs % 4 == 0
#Trixi.refine!(mesh.tree, LLID[Int(num_leafs/4)+1 : end])

# Refine mesh completely
#Trixi.refine!(mesh.tree, LLID)

# Update num_leafs:
num_leafs = length(Trixi.local_leaf_cells(mesh.tree))

#=
LLID = Trixi.local_leaf_cells(mesh.tree)
for id in LLID
  println(Trixi.cell_coordinates(mesh.tree, id))
end
=#

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)

###############################################################################
# ODE solvers, callbacks etc.

StartTime = 0.0
EndTime = 0.068856907635927 * 1000
#EndTime = 100

# Create ODEProblem
ode = semidiscretize(semi, (StartTime, EndTime));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100, extra_analysis_errors=(:conservation_error,))

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.0)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
#callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution)

###############################################################################
# run the simulation

#ode_algorithm = Trixi.CarpenterKennedy2N54()

dtOptMin = 0.068856907635927/4

#A = jacobian_ad_forward(semi)
A, = linear_structure(semi)
A = Matrix(A)
#=
A1, = linear_structure(semi, 1)
A1 = Matrix(A1)
A2, = linear_structure(semi, 2)
A2 = Matrix(A2)

#display(eigvals(A1 + A2))
@assert(zeros(size(A, 1), size(A, 2)) == A - (A1 + A2))
=#

#EigVals = eigvals(Matrix(A))
EigVals = eigvals(Matrix(A), sortby=nothing)
# Complex conjugate eigenvalues have same modulus
EigVals = EigVals[imag(EigVals) .>= 0]

# Sometimes due to numerical issues some eigenvalues have positive real part, which is erronous (for hyperbolic eqs)
if findfirst(x -> real(x) > 0, EigVals) != nothing
  println("Somewhat erronous spectrum (eigenvalue with positive real part)!")
end
EigVals = EigVals[real(EigVals) .< 0]

ode_algorithm = Trixi.FE2S(6, 1, dtOptMin, "/home/daniel/Desktop/git/MA/Optim_Monomials/Matlab/", A)
#ode_algorithm = Trixi.PERK(6, 1, 12, dtOptMin, "/home/daniel/Desktop/git/MA/Optim_Monomials/Matlab/")

plotdata = scatter!(real.(EigVals), imag.(EigVals), label = "Refined discretization")

plotdata = scatter!(real.(ode_algorithm.dtOptMin / dtOptMin * EigVals), 
                    imag.(ode_algorithm.dtOptMin / dtOptMin * EigVals), label = "Refined & rescaled discretization",
                    legend=:topleft)
#display(plotdata)

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