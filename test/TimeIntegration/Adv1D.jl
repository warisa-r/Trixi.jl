# The same setup as tree_1d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh


using Trixi
using OrdinaryDiffEq # For CallbackSet etc.
using Plots

#=

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEM(polydeg=1, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0,) # minimum coordinate
coordinates_max = (1.0,) # maximum coordinate
cells_per_dimension = (32,) # discretization size automatically computed

# Create curved mesh with 16 cells
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)

###############################################################################

=#

advection_velocity = (0.5, -0.1)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
Degree = 3
solver = DGSEM(polydeg=Degree, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

NumCells = 6
cells_per_dimension = (NumCells, NumCells)

# Create curved mesh with 16 x 16 elements
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)

### Burgers Equation (nonlinear test case) ###
#=

equations = InviscidBurgersEquation1D()
Degree = 1
solver = DGSEM(polydeg=Degree, surface_flux=flux_lax_friedrichs)
coordinates_min = (0.0,) # minimum coordinate
coordinates_max = (1.0,) # maximum coordinate
cells_per_dimension = (32,) # discretization size automatically computed

# Create curved mesh with 16 cells
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)
=#


# ODE Solver

NumStages = 32
dtOpt = 2.052

#ode_algorithm = Trixi.PERK(NumStages, NumStages, "./")
ode_algorithm = Trixi.FE2S(NumStages, dtOpt, "/home/daniel/Desktop/git/MA/Optim_Roots/IpOPT_dco/Feas_Reals/")
#ode_algorithm = Trixi.CarpenterKennedy2N43()

#exit()

#dtOpt = convert(Float64, ode_algorithm.dtOpt)
#=
println("Maximum (untested!) dt ist: ", dtOpt)

CFL = NumStageEvals / 2
println("CFL number for this problem is: ", CFL)
=#

#exit()

StartTime = 0.0
EndTime = 2 * dtOpt
#EndTime = 100.0

# Create ODEProblem
ode = semidiscretize(semi, (StartTime, EndTime));

###############################################################################
# Callbacks etc.

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=20)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
#callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution) # For own solutions

###############################################################################
# run the simulation
#visnodes = range(StartTime, EndTime, length=2)
#exit()

sol = Trixi.solve(ode, ode_algorithm,
                  dt=dtOpt, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep=false, callback=callbacks);
exit()

#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=0.1, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
=#
# Print the timer summary
summary_callback()
#exit()

PlotData = plot(sol)
#savefig(PlotData,"CarpenterKennedy2N43.png")
display(PlotData)
readline()

exit()
#=
for step in 1:length(sol.u)
  default(titlefont = (20, "times"), legendfontsize = 6)
  PlotData = plot(LinRange(coordinates_min[1], coordinates_max[1], convert(Int64, 2 * cells_per_dimension[1])),
                  sol.u[step])

  display(PlotData)
  readline() # Hang on to show plot until enter is hit
end
readline() # Hang on to show plot until enter is hit
=#

PlotData = plot(sol)
#savefig(PlotData,"PERK.png")
display(PlotData)
readline() # Hang on to show plot until enter is hit
