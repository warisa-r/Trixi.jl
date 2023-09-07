using Downloads: download
using OrdinaryDiffEq, LinearAlgebra, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

prandtl_number() = 0.72
mu() = 0.0005

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu=mu(), Prandtl=prandtl_number(),
                                                          gradient_variables=GradientVariablesPrimitive())

@inline function initial_condition(x, t, equations)
  # set the freestream flow parameters
  #rho_freestream = 1.4
  rho_freestream = 1.0
  v1 = 0.05
  Ms = 0.1
  v2 = 0.0
  p_freestream = (v1 / Ms)^2 * rho_freestream / equations.gamma # scaling to get Ms

  prim = SVector(rho_freestream, v1, v2, p_freestream)
  return prim2cons(prim, equations)
end

initial_condition = initial_condition

# Supersonic inflow boundary condition.
# Calculate the boundary flux entirely from the external solution state, i.e., set
# external solution state values for everything entering the domain.
@inline function boundary_condition_inflow(u_inner, normal_direction::AbstractVector, x, t,
                                           surface_flux_function, equations)
  u_boundary = initial_condition(x, t, equations)
  flux = Trixi.flux(u_boundary, normal_direction, equations)

  return flux
end


# Supersonic outflow boundary condition.
# Calculate the boundary flux entirely from the internal solution state. Analogous to supersonic inflow
# except all the solution state values are set from the internal solution as everything leaves the domain
@inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                            surface_flux_function, equations)
  flux = Trixi.flux(u_inner, normal_direction, equations)

  return flux
end

boundary_conditions = Dict( :Bottom  => boundary_condition_slip_wall,
                            :Circle  => boundary_condition_slip_wall,
                            :Top     => boundary_condition_slip_wall,
                            :Right   => boundary_condition_outflow,
                            :Left    => boundary_condition_inflow )

velocity_bc_inflow = NoSlip() do x, t, equations
  u = initial_condition(x, t, equations)
  return SVector(u[2], u[3])
end

velocity_bc_solid = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc_solid = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_solid = BoundaryConditionNavierStokesWall(velocity_bc_solid, heat_bc_solid)

velocity_bc_top_bottom = NoSlip() do x, t, equations
  u = initial_condition(x, t, equations)
  return SVector(u[2], u[3])
end
heat_bc_solid = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_top_bottom = BoundaryConditionNavierStokesWall(velocity_bc_top_bottom, heat_bc_solid)

heat_bc_inflow = Isothermal((x, t, equations) -> 
                  Trixi.temperature(initial_condition(x, t, equations), 
                                    equations_parabolic))
boundary_condition_inflow_para = BoundaryConditionNavierStokesWall(velocity_bc_inflow, heat_bc_inflow)

# TODO: True outflow for non-viscous things needed!
boundary_condition_outflow_para = BoundaryConditionNavierStokesWall(velocity_bc_inflow, heat_bc_solid)

boundary_conditions_para = Dict(:Bottom  => boundary_condition_top_bottom,
                                :Circle  => boundary_condition_solid,
                                :Top     => boundary_condition_top_bottom,
                                :Right   => boundary_condition_outflow_para,
                                :Left    => boundary_condition_inflow_para )                            

volume_flux = flux_ranocha_turbo
surface_flux = flux_lax_friedrichs

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
#=
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max=0.5,
                                            alpha_min=0.001,
                                            alpha_smooth=true,
                                            variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
=#
volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha_turbo)                                               
solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux, volume_integral=volume_integral)


# Get the unstructured quad mesh from a file (downloads the file if not available locally)
mesh_file = "out/cylinder.inp"

mesh = P4estMesh{2}(mesh_file, initial_refinement_level=0, polydeg=1)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver;
                                             boundary_conditions=(boundary_conditions, boundary_conditions_para))

###############################################################################
# ODE solvers

tspan = (0.0, 22)
ode = semidiscretize(semi, tspan)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)


amr_indicator = IndicatorLöhner(semi, variable=Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=0,
                                      med_level=1, med_threshold=0.05,
                                      max_level=2, max_threshold=0.1)

amr_callback = AMRCallback(semi, amr_controller,
                          interval=1,
                          adapt_initial_condition=true,
                          adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=0.01)


callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = Trixi.solve(ode, Trixi.CarpenterKennedy2N54(),
            dt=5e-3, callback=callbacks);

#=
b1 = 0.0
bS = 1 - b1
cEnd = 0.5/bS

ode_algorithm = PERK_Multi(4, 2, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_CEE_P4est/",
                            bS, cEnd)

dt = 0.0398892400284239579 * 0.01
sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks);
=#

summary_callback() # print the timer summary
Plots.plot(sol)

pd = PlotData2D(sol)
Plots.plot(pd["rho"])
Plots.plot(getmesh(pd))