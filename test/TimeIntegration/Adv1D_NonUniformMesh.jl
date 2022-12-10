
using OrdinaryDiffEq, Plots, LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux

PolyDeg = 3
surface_flux = flux_lax_friedrichs
solver = DGSEM(polydeg=PolyDeg, surface_flux=surface_flux)


#=
numerical_flux = flux_lax_friedrichs
PolyDeg = 0
solver = DGSEM(polydeg=PolyDeg, surface_flux=numerical_flux,
               volume_integral=VolumeIntegralPureLGLFiniteVolume(numerical_flux))
=#

coordinates_min = -1.0 # minimum coordinate
coordinates_max =  1.0 # maximum coordinate

InitialRefinement = 6
# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                # Start from one cell => Results in 1 + 2 + 4 + 8 + 16 = 2^5 - 1 = 31 cells
                initial_refinement_level=InitialRefinement,
                n_cells_max=30_000) # set maximum capacity of tree data structure

#=
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
Trixi.refine!(mesh.tree, LLID[Int(2*num_leafs/4) : Int(3*num_leafs/4)])

# For testing: Have transition: Coarse -> Fine (without medium in between - bot supported by Trixi)
#Trixi.refine!(mesh.tree, LLID[5 : Int(3*num_leafs/4)])


# Third refinement
# Refine mesh locally
LLID = Trixi.local_leaf_cells(mesh.tree)
num_leafs = length(LLID)

Trixi.refine!(mesh.tree, LLID[18 : 22])
=#

initial_condition = initial_condition_convergence_test
#initial_condition = initial_condition_gauss

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

#for i = 1:100
  StartTime = 0.0
  EndTime = 0.054593 / (2^(InitialRefinement - 4))
  #EndTime = 10*rand()
  EndTime = 10


  # Create ODEProblem
  ode = semidiscretize(semi, (StartTime, EndTime));

  # At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
  # and resets the timers
  summary_callback = SummaryCallback()

  # The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
  analysis_callback = AnalysisCallback(semi, interval=1000, extra_analysis_errors=(:conservation_error,))

  # The SaveSolutionCallback allows to save the solution to a file in regular intervals
  #=
  save_solution = SaveSolutionCallback(interval=1,
                                      solution_variables=cons2prim)
  =#
  # The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
  #stepsize_callback = StepsizeCallback(cfl=1.0)

  # Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
  #callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)
  #callbacks = CallbackSet(summary_callback, analysis_callback)

  callbacks = CallbackSet(summary_callback, analysis_callback)

  ###############################################################################
  # run the simulation

  #dtOptMin = 0.05
  # For s = 4
  dtOptMin = 0.0545930 / (2^(InitialRefinement - 4))

  # s= 12
  #dtOptMin = 0.0432664315456349892

  #=
  # s = 24
  #dtOptMin = 0.0432872764758940345 * 0.999
  dtOptMin = 0.04328

  # s = 64
  dtOptMin = 0.04328 * 1.0

  # s = 96 ("three-based")
  dtOptMin = 0.0432873
  =#

  #=
  ode_algorithm = FE2S(96, 
                    "/home/daniel/Desktop/git/MA/EigenspectraGeneration/Spectra/1D_Adv_ConvergenceTest/")
                    #"/home/daniel/Desktop/git/MA/Optim_Roots/IpOPT_dco/Feas_Reals/")
  =#

  #=
  ode_algorithm = PERK(4, 
                      "/home/daniel/Desktop/git/MA/EigenspectraGeneration/Spectra/1D_Adv_ConvergenceTest/")
  =#

  
  ode_algorithm = PERK_Multi(4, 3, 
                  "/home/daniel/Desktop/git/MA/EigenspectraGeneration/Spectra/1D_Adv_ConvergenceTest/")
  
  sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
              dt = 1e-2/2,
              save_everystep=false, callback=callbacks);

  sol = Trixi.solve(ode, ode_algorithm,
                    #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                    dt = dtOptMin/32,
                    save_everystep=false, callback=callbacks);

  # Print the timer summary
  summary_callback()

  #PlotData = plot(sol)
  #savefig(PlotData, string(i) * ".png")
#end

plot(sol)


pd = PlotData1D(sol)
plot!(getmesh(pd))


#=
plot!(sol)

# Analytical solution at T = 100
x = range(-1, 1; length = 100)
y = 1 .+ 0.5*sinpi.(x .- advection_velocity * EndTime)
plot!(x, y)
=#
