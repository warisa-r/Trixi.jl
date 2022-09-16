# The same setup as tree_1d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh


using Trixi
using OrdinaryDiffEq # For CallbackSet etc.
using Plots

###############################################################################
# semidiscretization of the linear advection equation

rho_0 = 1.0
c_0   = 0.2
Mach  = 0.5
v_0   = Mach * c_0
equations = LinearizedEulerEquations1D(rho_0, v_0, c_0)

N = 1 # Degree
solver = DGSEM(polydeg = N, surface_flux = flux_lax_friedrichs)

coordinates_min = (0.0,) # minimum coordinate
coordinates_max = (2.0,) # maximum coordinate
cells_per_dimension = (200,) # discretization size automatically computed

StartTime = 0.0
EndTime = 5.0

# Create curved mesh with 16 cells
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

#=
initial_condition = initial_condition_rest

BCs = (x_neg=BoundaryConditionDirichlet(boundary_condition_inlet),
       x_pos=BoundaryConditionDirichlet(boundary_condition_inlet))           

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=BCs)
=#

semi = SemidiscretizationHyperbolic(mesh, equations, Trixi.initial_condition_acoustic_wave, solver)

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (StartTime, EndTime));

###############################################################################
# ODE Solver

dxMin = minimum((collect(coordinates_max) - collect(coordinates_min) ) ./ collect(cells_per_dimension))
AbsMaxEigVal = abs(v_0) + c_0
#CFL = 1 / (2 * N + 1) # CFL cond. according to https://scicomp.stackexchange.com/questions/26018/cfl-condition-in-discontinuous-galerkin-schemes
CFL = 1
dtCFL = CFL * dxMin / AbsMaxEigVal # TODO: Adapt CFL number from Preuhs MA
println("Expected timestep: ", dtCFL)


#dtMax = Double64(EndTime - StartTime)
dtMax = EndTime - StartTime
println("Supplied dtMax is: ", dtMax)
#dtEps = Double64(1e-9)
dtEps = 1e-9

NumStages     = 8
NumStageEvals = 8 # NumUnknonws = NumStageEvals - 2
ode_algorithm = Trixi.PERK(NumStages, NumStageEvals, semi, dtMax, WriteEvalsToFile = true, EvalsFileWrite = "./Evals.txt")

dtOpt = convert(Float64, ode_algorithm.dtOpt)
println("Maximum (untested!) dt ist: ", dtOpt)

#=
CFL = dtOpt / dtCFL1
println("CFL number for this problem is: ", CFL)
=#

#ode_algorithm = CarpenterKennedy2N54(williamson_condition=false)

###############################################################################
# Callbacks etc.

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
#stepsize_callback = StepsizeCallback(cfl=1.0)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution)

###############################################################################
# run the simulation

N_plot = convert(Int64, (2*N+1) * cells_per_dimension[1])
x_plot = LinRange(coordinates_min[1], coordinates_max[1], N_plot)
CharVars = zeros(3, N_plot)

visnodes = range(StartTime, EndTime, length=200)
sol = Trixi.solve(ode, ode_algorithm,
            dt=dtCFL, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, saveat=visnodes, callback=callbacks);

# Print the timer summary
summary_callback()

println(length(sol.u))
exit()


for step in 1:length(sol.u)

  x_char = Trixi.compute_char_initial_pos(x_plot, sol.t[step], equations)
  for p in 1:3
    CharVars[p,:] = Trixi.initial_condition_char_vars_acoustic_wave(x_char[p,:]', p, equations)
  end
  Solution = Trixi.compute_primal_sol(CharVars, equations)
  
  default(titlefont = (20, "times"), legendfontsize = 6)
  p1 = plot(LinRange(coordinates_min[1], coordinates_max[1], convert(Int64, 2 * cells_per_dimension[1])),
            sol.u[step][1:3:end], label = "rho'")
  p1 = plot!(x_plot, Solution[1,:], label = "Analytic Sol")

  p2 = plot(LinRange(coordinates_min[1], coordinates_max[1], convert(Int64, 2 * cells_per_dimension[1])),
            sol.u[step][2:3:end], label = "u'")
  p2 = plot!(x_plot, Solution[2,:], label = "Analytic Sol")

  p3 = plot(LinRange(coordinates_min[1], coordinates_max[1], convert(Int64, 2 * cells_per_dimension[1])),
            sol.u[step][3:3:end], label = "p'")
  p3 = plot!(x_plot, Solution[3,:], label = "Analytic Sol")

  l = @layout [a; b; c]
  PlotDataDisp = plot(p1, p2, p3, layout = l, plot_title = "Entropy or Acoustic Wave")

  display(PlotDataDisp)
  readline() # Hang on to show plot until enter is hit
end
readline() # Hang on to show plot until enter is hit

#=
PlotData = plot(sol)
savefig(PlotData,"EntropyWave.png")
display(PlotData)
readline() # Hang on to show plot until enter is hit
=#