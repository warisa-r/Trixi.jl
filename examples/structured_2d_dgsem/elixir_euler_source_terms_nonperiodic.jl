
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

# you can either use a single function to impose the BCs weakly in all
# 2*ndims == 4 directions or you can pass a tuple containing BCs for each direction
boundary_condition = BoundaryConditionDirichlet(initial_condition)
#=


coordinates_min = (0.0, 0.0)
coordinates_max = (2.0, 2.0)
mesh = StructuredMesh((16, 16), coordinates_min, coordinates_max, periodicity=false)
=#

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000,
                periodicity=false)

LLID = Trixi.local_leaf_cells(mesh.tree)
num_leafs = length(LLID)
@assert num_leafs % 4 == 0
Trixi.refine!(mesh.tree, LLID[1:Int(num_leafs/4)])

LLID = Trixi.local_leaf_cells(mesh.tree)
num_leafs = length(LLID)
@assert num_leafs % 4 == 0
Trixi.refine!(mesh.tree, LLID[1:Int(num_leafs/4)])

boundary_conditions = (x_neg=boundary_condition,
                      x_pos=boundary_condition,
                      y_neg=boundary_condition,
                      y_pos=boundary_condition,)                


mesh_file = "out/box.inp"
mesh = P4estMesh{2}(mesh_file)

boundary_conditions = Dict( :Bottom  => boundary_condition,
                            :Top     => boundary_condition,
                            :Right   => boundary_condition,
                            :Left    => boundary_condition )


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_convergence_test,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution,
                        stepsize_callback)
###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

#=
dt = 0.0355976722988998537 * 0.29 # 4, 6, 8, 10, 12, 14, 16
dt = 0.0355976722988998537 * 0.25 # 4, 8, 16

b1 = 0.0
bS = 1 - b1
cEnd = 0.5/bS

ode_algorithm = PERK_Multi(4, 2, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_CEE_P4est/",
                            bS, cEnd)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks);
=#
                  
summary_callback() # print the timer summary

Plots.plot(sol)
Plots.plot!(getmesh(PlotData2D(sol)))