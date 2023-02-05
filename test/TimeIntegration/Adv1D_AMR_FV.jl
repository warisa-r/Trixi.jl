
using OrdinaryDiffEq, Plots, LinearAlgebra
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
coordinates_max =  1.0 # maximum coordinate

RefinementLevel = 6
# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                # Start from one cell => Results in 1 + 2 + 4 + 8 + 16 = 2^5 - 1 = 31 cells
                initial_refinement_level=RefinementLevel,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# Manual refinement (see NonUniformMesh)
#=
LLID = Trixi.local_leaf_cells(mesh.tree)
num_leafs = length(LLID)

# Refine center of mesh
@assert num_leafs % 4 == 0
Trixi.refine!(mesh.tree, LLID[Int(num_leafs/4)+1 : Int(3*num_leafs/4)])
=#


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


#initial_condition = initial_condition_gauss
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

Interval = 5
# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=Interval, extra_analysis_errors=(:conservation_error,))

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
#stepsize_callback = StepsizeCallback(cfl=1.0)


amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first),
                                      base_level=RefinementLevel,
                                      med_level=RefinementLevel+1, med_threshold=0.1,
                                      max_level=RefinementLevel+2, max_threshold=0.6)


#=
amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first),
                                      base_level=RefinementLevel,
                                      med_level=RefinementLevel+1, med_threshold=0.1,
                                      max_level=RefinementLevel+3, max_threshold=0.6)
=#

amr_callback = AMRCallback(semi, amr_controller,
                           interval=Interval,
                           adapt_initial_condition=false) # Adaption of initial condition not yet supported

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacksPERK = CallbackSet(summary_callback, analysis_callback, amr_callback)
callbacksPERK = CallbackSet(summary_callback, analysis_callback)

stepsize_callback = StepsizeCallback(cfl=1.0)     
callbacksSSPRK22 = CallbackSet(summary_callback, analysis_callback,
                               stepsize_callback)


###############################################################################
# run the simulation

#=
sol = solve(ode, 
            #CarpenterKennedy2N54(williamson_condition=false),
            SSPRK22(),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacksSSPRK22);
=#

NumStagesBase = 2
NumDoublings = 2

# p = 0, NumStagesBase = 4
dtOptMin = 0.0937500079744495451 / (2.0^(RefinementLevel - 6))

# p = 0, NumStagesBase = 2
dtOptMin = 0.0312500000003637967 / (2.0^(RefinementLevel - 6))


CFL = 1.0

ode_algorithm = PERK_Multi(NumStagesBase, NumDoublings,
                           "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/p0/")

sol = Trixi.solve(ode, ode_algorithm,
                  #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  dt = dtOptMin * CFL,
                  save_everystep=false, callback=callbacksPERK);

# Print the timer summary
summary_callback()

plot(sol, plot_title = EndTime)

pd = PlotData1D(sol)

#=
io = open("x_final.txt", "w") do io
  for Pos in pd.x
    println(io, Pos)
  end
end
=#

plot!(getmesh(pd))