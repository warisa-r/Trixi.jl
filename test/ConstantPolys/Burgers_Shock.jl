
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = InviscidBurgersEquation1D()

PolyDegree = 0
numerical_flux = flux_lax_friedrichs
solver = DGSEM(polydeg=PolyDegree, surface_flux=numerical_flux)

coordinate_min = 0.0
coordinate_max = 1.0

# Make sure to turn periodicity explicitly off as special boundary conditions are specified
mesh = TreeMesh(coordinate_min, coordinate_max,
                initial_refinement_level=6,
                n_cells_max=10_000,
                periodicity=false)

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

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.9)


callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation


sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=42, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
plot(sol)