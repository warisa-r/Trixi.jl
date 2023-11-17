using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

# TODO: parabolic; unify names of these accessor functions
prandtl_number() = 0.72
mu() = 1e-4 # Re = 10000

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu=mu(),
                                                          Prandtl=prandtl_number())

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
surf_fux = FluxHLL(min_max_speed_davis)
solver = DGSEM(polydeg=2, surface_flux=surf_fux)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                periodicity=false,
                n_cells_max=30_000) # set maximum capacity of tree data structure


function initial_condition_cavity(x, t, equations::CompressibleEulerEquations2D)
  Ma = 0.1
  rho = 1.0
  u, v = 0.0, 0.0
  p = 1.0 / (Ma^2 * equations.gamma)
  return prim2cons(SVector(rho, u, v, p), equations)
end
initial_condition = initial_condition_cavity

# BC types
velocity_bc_lid = NoSlip((x, t, equations) -> SVector(1.0, 0.0))
velocity_bc_cavity = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_lid = BoundaryConditionNavierStokesWall(velocity_bc_lid, heat_bc)
boundary_condition_cavity = BoundaryConditionNavierStokesWall(velocity_bc_cavity, heat_bc)

# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_slip_wall

boundary_conditions_parabolic = (; x_neg = boundary_condition_cavity,
                                   y_neg = boundary_condition_cavity,
                                   y_pos = boundary_condition_lid,
                                   x_pos = boundary_condition_cavity)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions=(boundary_conditions,
                                                                  boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan, split_form = false);

summary_callback = SummaryCallback()
analysis_interval = 50
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
amr_indicator = IndicatorLöhner(semi, variable=Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 2,
                                      med_level  = 5, med_threshold=0.00003,
                                      max_level  = 7, max_threshold=0.00008)
                                
amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)


###############################################################################
# run the simulation

#S = 3, p = 2, d = 2, Ref = 3
CFL_Ref = 2
dt = 0.00739935292367590632 * CFL_Ref

#dt *= 2 # S = 6
#dt *= 4 # S = 12


time_int_tol = 1e-8
callbacks = CallbackSet(summary_callback, amr_callback, analysis_callback)
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacks);


#=
callbacks = CallbackSet(summary_callback, amr_callback, analysis_callback)
#callbacks = CallbackSet(summary_callback, analysis_callback)

b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS

#=
ode_algorithm = PERK(12, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/LidDrivenCavity/",
                     bS, cEnd)
=#

Integrator_Mesh_Level_Dict = Dict([(42, 42)])
ode_algorithm = PERK_Multi(3, 2, 
                           "/home/daniel/git/MA/EigenspectraGeneration/Spectra/LidDrivenCavity/",
                           bS, cEnd,
                           stage_callbacks = ())


sol = Trixi.solve(ode, ode_algorithm, 
                  dt = dt,
                  save_everystep=false, callback=callbacks)
=#

summary_callback() # print the timer summary
plot(sol)

pd = PlotData2D(sol)
plot(pd["rho"])
plot(pd["v1"])
plot(pd["v2"])
plot!(getmesh(pd))