
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = Trixi.MoshpitEquations1D()

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=0, surface_flux=flux_lax_friedrichs)

coordinate_min = 0.0
coordinate_max = 2.0

# Make sure to turn periodicity explicitly off as special boundary conditions are specified
mesh = TreeMesh(coordinate_min, coordinate_max,
                initial_refinement_level=7,
                n_cells_max=10_000,
                periodicity=false)

function initial_condition_shock(x, t, equation::Trixi.MoshpitEquations1D)
  rho = x[1] <= 0.2 || x[1] >= 1.8 ? 5.0 : 0.01
  if x[1] <= 0.2
    v = 1.0
  elseif  x[1] >= 1.8
    v = -1.0
  else
    v = 0.0
  end
    

  return SVector(rho, rho*v)
end

###############################################################################
# Specify non-periodic boundary conditions

function boundary_condition_outflow(u_inner, orientation, normal_direction, x, t,
                                    surface_flux_function, equations::Trixi.MoshpitEquations1D)
  # Calculate the boundary flux entirely from the internal solution state
  flux = Trixi.flux(u_inner, normal_direction, equations)

  return flux
end

# TODO: Work out reflective BC!
function boundary_condition_reflect(u_inner, orientation_or_normal, direction,
                                    x, t,
                                    surface_flux_function,
                                    equations::Trixi.MoshpitEquations1D)

  #return SVector(u_inner[1], - u_inner[2])
    # create the "external" boundary solution state
    u_boundary = SVector(u_inner[1],
                        -u_inner[2])

  # calculate the boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation_or_normal, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation_or_normal, equations)
  end

  return flux
end

function boundary(x, t, equations::Trixi.MoshpitEquations1D)
  return SVector(1, 0)
end
boundary_condition_const = BoundaryConditionDirichlet(boundary)


boundary_conditions = (x_neg=boundary_condition_reflect,
                       x_pos=boundary_condition_reflect)
                       
initial_condition = initial_condition_shock

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.8)


callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=42, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
plot(sol)