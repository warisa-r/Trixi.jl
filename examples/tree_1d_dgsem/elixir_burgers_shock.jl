
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = InviscidBurgersEquation1D()

initial_condition = initial_condition_shock

solver = DGSEM(polydeg=1, surface_flux=flux_lax_friedrichs)

coordinates_min = 0.0
coordinates_max = 1.0

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000,
                periodicity=false) # CARE: This has to be explicitly set!

function inflow(x, t, equations::InviscidBurgersEquation1D)
  return initial_condition_shock(coordinates_min, t, equations)
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

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
#tspan = (0.0, 0.0) # For plotting initial condition
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors=(:l2_error_primitive,
                                                            :linf_error_primitive,
                                                            :conservation_error))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1)


callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

#=
# Try positivity limiter to prevent oscillations - seems not to give anything
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(1.6,),
                                                     variables=(Trixi.scalar,))

ode_algorithm = SSPRK33(stage_limiter!)
=#

ode_algorithm = SSPRK33()
sol = solve(ode, ode_algorithm,
            dt=1, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary