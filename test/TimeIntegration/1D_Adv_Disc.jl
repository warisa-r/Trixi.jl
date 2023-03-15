using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

PolyDegree = 0
numerical_flux = flux_lax_friedrichs
solver = DGSEM(polydeg=PolyDegree, surface_flux=numerical_flux)
               #volume_integral=VolumeIntegralPureLGLFiniteVolume(numerical_flux))

coordinates_min = -1.0 # minimum coordinate
coordinates_max =  1.0 # maximum coordinate

RefinementLevel = 8
# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                # Start from one cell => Results in 1 + 2 + 4 + 8 + 16 = 2^5 - 1 = 31 cells
                initial_refinement_level=RefinementLevel,
                n_cells_max=30_000,
                periodicity=false) # set maximum capacity of tree data structure

# Discontinuous initial condition (Riemann Problem) leading to a shock to test e.g. correct shock speed.
function initial_condition_shock(x, t, equation::LinearScalarAdvectionEquation1D)
  scalar = x[1] < -0.75 ? 1.5 : 0.5

  return SVector(scalar)
end

###############################################################################
# Specify non-periodic boundary conditions

function inflow(x, t, equations::LinearScalarAdvectionEquation1D)
  return initial_condition_shock(coordinates_min, t, equations)
end
boundary_condition_inflow = BoundaryConditionDirichlet(inflow)

function boundary_condition_outflow(u_inner, orientation, normal_direction, x, t,
                                    surface_flux_function, equations::LinearScalarAdvectionEquation1D)
  # Calculate the boundary flux entirely from the internal solution state
  flux = Trixi.flux(u_inner, orientation, equations)

  return flux
end

boundary_conditions = (x_neg=boundary_condition_inflow,
                       x_pos=boundary_condition_outflow)
                       
initial_condition = initial_condition_shock

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

dtRef = 0.00781250000072759619
StagesRef = 2

#=
dtRef = 0.101582545940618735
StagesRef = 14
=#

NumStages = 2
CFL = 1.0

dt = dtRef * NumStages / StagesRef * CFL

#=
sol = solve(ode, SSPRK22(),
            dt=dt, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
=#

#ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Adv_FV/")
ode_algorithm = SSPRK2S(NumStages)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks)

summary_callback() # print the timer summary
plot(sol)

TV0 = 0
for i in 1:length(sol.u[1])-1
  TV0 += abs(sol.u[1][i+1] - sol.u[1][i])
end

println("Initial Total Variation:\t", TV0)

TV = 0
for i in 1:length(sol.u[end])-1
  TV += abs(sol.u[end][i+1] - sol.u[end][i])
end

println("Final Total Variation:\t\t", TV)

#=
pd = PlotData1D(sol)
plot!(getmesh(pd))
=#