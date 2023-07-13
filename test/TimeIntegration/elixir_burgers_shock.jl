using OrdinaryDiffEq, Plots, LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = InviscidBurgersEquation1D()

PolyDeg = 3
basis = LobattoLegendreBasis(PolyDeg)
# Use shock capturing techniques to supress oscillations at discontinuities
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=first)

volume_flux  = flux_ec
surface_flux = flux_lax_friedrichs
#surface_flux = flux_godunov

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=surface_flux,
                                                 #volume_flux_fv=volume_flux) # This gives a spike for some reason
                                                 volume_flux_fv=surface_flux)
                                                 
solver = DGSEM(basis, surface_flux, volume_integral)
#solver = DGSEM(polydeg=PolyDeg, surface_flux=surface_flux)

coordinate_min = 0.0
coordinate_max = 1.0

# Make sure to turn periodicity explicitly off as special boundary conditions are specified
InitialRefinement = 6
mesh = TreeMesh(coordinate_min, coordinate_max,
                initial_refinement_level=InitialRefinement,
                n_cells_max=10_000,
                periodicity=false)

#=
# First refinement
# Refine mesh locally 
LLID = Trixi.local_leaf_cells(mesh.tree)
num_leafs = length(LLID)

# Refine middle of mesh
@assert num_leafs % 4 == 0
Trixi.refine!(mesh.tree, LLID[Int(num_leafs/4)+1 : Int(3*num_leafs/4)])                
=#

# Discontinuous initial condition (Riemann Problem) leading to a shock to test e.g. correct shock speed.
function initial_condition_shock(x, t, equation::InviscidBurgersEquation1D)
  scalar = x[1] < 0.5 ? 1.5 : 0.5

  return SVector(scalar)
end


###############################################################################
# Specify non-periodic boundary conditions

function inflow(x, t, equations::InviscidBurgersEquation1D)
  return initial_condition_shock(coordinate_min, t, equations)
end
boundary_condition_inflow = BoundaryConditionDirichlet(inflow)

function boundary_condition_outflow(u_inner, orientation, normal_direction, x, t,
                                    surface_flux_function, equations::InviscidBurgersEquation1D)
  # Calculate the boundary flux entirely from the internal solution state
  flux = Trixi.flux(u_inner, normal_direction, equations)

  return flux
end


boundary_conditions = (x_neg=boundary_condition_inflow,
                       x_pos=boundary_condition_outflow)
                       
initial_condition = initial_condition_shock

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.25)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.9)

amr_controller = ControllerThreeLevel(semi, 
                                      #IndicatorMax(semi, variable=first),
                                      indicator_sc,
                                      base_level=InitialRefinement,
                                      med_level=InitialRefinement+1, med_threshold=0.1, #0.1
                                      max_level=InitialRefinement+2, max_threshold=0.6) #0.6


amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=false) # Adaption of initial condition not yet supported


callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

callbacksRED = CallbackSet(summary_callback,
                           analysis_callback, alive_callback, amr_callback)

###############################################################################
# run the simulation

# TODO: Check with non SSP integrator - maybe possible to produce oscillations
#=
solHeun = Trixi.solve(ode, Trixi.Heun(), #CarpenterKennedy2N54(williamson_condition=false),
            dt=1e-3, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacksRED);

solSSPRKS2 = Trixi.solve(ode, Trixi.SSPRKS2(2), #CarpenterKennedy2N54(williamson_condition=false),
            dt=1e-3, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacksRED);            

summary_callback() # print the timer summary

println(norm(solHeun.u[end] - solSSPRKS2.u[end]))
=#

b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS

ode_algorithm = PERK_Multi(4, 2, "/home/daniel/git/MA/EigenspectraGeneration/BurgersRiemannProb/",
                           bS, cEnd, stage_callbacks = ())

# S = 2
CFL = 0.39 # NOTE: Almost same CFL as for S=4 => I attribute this to the optimization
dt = 0.00289164227200672036 / (2.0^(InitialRefinement - 6)) * CFL


# S = 4
#CFL = 0.4 # S = 4, standalone
CFL = 0.33 # Two levels, b1 = 0.5
CFL = 0.23 # Three levels, b1 = 0.5

CFL = 0.4 # Two levels, b1 = 0.0
CFL = 0.26 # Three Levels, b1 = 0.0 (actually CFL = 0.27 possible, but then the mesh is only once refined)


# S = 4 only
CFL = 0.51

dt  = 0.00632331863453146105 / (2.0^(InitialRefinement+2 - 6)) * CFL

ode_algorithm = PERK(4, "/home/daniel/git/MA/EigenspectraGeneration/BurgersRiemannProb/")

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacksRED);
plot(sol)
plot!(getmesh(PlotData1D(sol)))