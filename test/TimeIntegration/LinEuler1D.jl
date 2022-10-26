# The same setup as tree_1d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh


using Trixi
using OrdinaryDiffEq # For CallbackSet etc.
using Plots

###############################################################################
# semidiscretization of the linear advection equation

# 
rho_0 = 1.255
c_0   = 343
Mach  = 10
v_0   = Mach * c_0
#equations = LinearizedEulerEquations1D(rho_0, v_0, c_0)
equations = Trixi.LinearizedEulerEquations1D_PosDepMach(rho_0, 0.1 * c_0, 0.3 * c_0, c_0)

N = 3 # Degree
# HLL advisable over Lax-Friedrichs for system of Consevation Laws
solver = DGSEM(polydeg = N, surface_flux = flux_hll)


coordinates_min = (0.0,) # minimum coordinate
coordinates_max = (1.0,) # maximum coordinate
#=
cells_per_dimension = (64,) # discretization size automatically computed
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)
=#

InitialRefinement = 6
# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                # Start from one cell => Results in 1 + 2 + 4 + 8 + 16 + 32 + 64 = 2^7 - 1 = 255 cells (but only 64 elements)
                initial_refinement_level=InitialRefinement,
                n_cells_max=30_000) # set maximum capacity of tree data structure

#=
initial_condition = initial_condition_rest

BCs = (x_neg=BoundaryConditionDirichlet(boundary_condition_inlet),
       x_pos=BoundaryConditionDirichlet(boundary_condition_inlet))           

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=BCs)
=#

#ic = initial_condition_convergence_test
#ic = initial_condition_acoustic_wave
ic = initial_condition_entropy_wave
#ic = Trixi.initial_condition_custom
#ic = initial_condition_rest
semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver)

# Create ODE problem 
StartTime = 0.0
EndTime = 1e-1

ode = semidiscretize(semi, (StartTime, EndTime))

###############################################################################
# Callbacks etc.

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100, extra_analysis_errors=(:conservation_error,))

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.8)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)
#callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# run the simulation

#N_plot = convert(Int64, (2*N+1) * cells_per_dimension[1])
N_plot = convert(Int64, (2*N+1) * 2^InitialRefinement)
x_plot = LinRange(coordinates_min[1], coordinates_max[1], N_plot)
CharVars = zeros(3, N_plot)

visnodes = range(StartTime, EndTime, length=400)


#ode_algorithm = SSPRK22()
ode_algorithm = CarpenterKennedy2N54(williamson_condition=false)
sol = solve(ode, ode_algorithm,
            dt=42, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, saveat=visnodes, callback=callbacks);

pd = PlotData1D(sol)
plot(sol)

#=
# 4 Stage, Mach = 0.1
dtOptMin = 1.8046855181455613e-05

# 32 Stage, Mach = 10
#dtOptMin = 1.5296e-05
ode_algorithm = PERK(4, 0, dtOptMin,
                     "/home/daniel/Desktop/git/MA/Optim_Monomials/Cpp/Results/LinEuler1D_Mach01/")
sol = Trixi.solve(ode, ode_algorithm,
                  dt=dtOptMin, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback=callbacks);
=#

# Print the timer summary
summary_callback()

@gif for step in 1:length(sol.u)

  #x_char = Trixi.compute_char_initial_pos(x_plot, sol.t[step], equations)
  for p in 1:3
    #CharVars[p,:] = Trixi.initial_condition_char_vars_convergence_test(x_char[p,:]', p, equations)
    #CharVars[p,:] = Trixi.initial_condition_char_vars_acoustic_wave(x_char[p,:]', p, equations)
    #CharVars[p,:] = Trixi.initial_condition_char_vars_entropy_wave(x_char[p,:]', p, equations)
    #CharVars[p,:] = Trixi.initial_condition_char_vars_custom(x_char[p,:]', p, equations)
  end
  #Solution = Trixi.compute_primal_sol(CharVars, equations)
  
  default(titlefont = (20, "times"), legendfontsize = 6)
  p1 = plot(LinRange(coordinates_min[1], coordinates_max[1], convert(Int64, length(sol.u[step][1:4:end]))),
            sol.u[step][1:4:end], label = "rho'")
  #p1 = plot!(x_plot, Solution[1,:], label = "Analytic Sol")

  p2 = plot(LinRange(coordinates_min[1], coordinates_max[1], convert(Int64, length(sol.u[step][2:4:end]))),
            sol.u[step][2:4:end], label = "u'")
  #p2 = plot!(x_plot, Solution[2,:], label = "Analytic Sol")

  p3 = plot(LinRange(coordinates_min[1], coordinates_max[1], convert(Int64, length(sol.u[step][3:4:end]))),
            sol.u[step][3:4:end], label = "p'")
  #p3 = plot!(x_plot, Solution[3,:], label = "Analytic Sol")

  p4 = plot(LinRange(coordinates_min[1], coordinates_max[1], convert(Int64, length(sol.u[step][4:4:end]))),
            sol.u[step][4:4:end], label = "v_0(x)")

  l = @layout [a; b; c; d]
  #PlotDataDisp = plot(p1, p2, p3, p4, layout = l, plot_title = "Acoustic Wave")
  PlotDataDisp = plot(p1, p2, p3, p4, layout = l, plot_title = "Entropy Wave")

  display(PlotDataDisp)
  #readline() # Hang on to show plot until enter is hit
end
readline() # Hang on to show plot until enter is hit

#=
PlotData = plot(sol)
savefig(PlotData,"EntropyWave.png")
display(PlotData)
readline() # Hang on to show plot until enter is hit
=#