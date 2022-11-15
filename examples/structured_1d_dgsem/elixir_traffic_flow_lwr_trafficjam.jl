
using OrdinaryDiffEq
using Trixi
using Plots

###############################################################################

# Example taken from http://www.clawpack.org/riemann_book/html/Traffic_flow.html#Example:-Traffic-jam

equations = TrafficFlowLWR1D()

basis = LobattoLegendreBasis(3)

surface_flux = flux_lax_friedrichs
                                                 
solver = DGSEM(basis, surface_flux)

coordinate_min = (-1.0,) # minimum coordinate
coordinate_max = (1.0,) # maximum coordinate
cells_per_dimension = (32,)

# Create curved mesh with 16 cells
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max, periodicity = false)

# Discontinuous initial condition (Riemann Problem) leading to a shock that moves to the left
function initial_condition_traffic_jam(x, t, equation::TrafficFlowLWR1D)
  scalar = x[1] < 0.0 ? 0.5 : 1.0

  return SVector(scalar)
end

###############################################################################
# Specify non-periodic boundary conditions

function outflow(x, t, equations::TrafficFlowLWR1D)
  return initial_condition_traffic_jam(coordinate_min, t, equations)
end
boundary_condition_outflow = BoundaryConditionDirichlet(outflow)

function boundary_condition_inflow(u_inner, orientation, normal_direction, x, t,
                                   surface_flux_function, equations::TrafficFlowLWR1D)
  # Calculate the boundary flux entirely from the internal solution state
  flux = Trixi.flux(u_inner, orientation, equations)

  return flux
end


boundary_conditions = (x_neg=boundary_condition_outflow,
                       x_pos=boundary_condition_inflow)
                       
initial_condition = initial_condition_traffic_jam

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
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
plot(sol)

summary_callback() # print the timer summary
