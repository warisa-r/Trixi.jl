
using Trixi, OrdinaryDiffEq, Plots

###############################################################################
# semidiscretization of the compressible Euler equations

gamma = 5/3
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_rayleigh_taylor_instability(coordinates, t, equations::CompressibleEulerEquations2D)

Setup used for the Rayleigh-Taylor instability. Initial condition adapted from
- Shi, Jing, Yong-Tao Zhang, and Chi-Wang Shu (2003).
  Resolution of high order WENO schemes for complicated flow structures.
  [DOI](https://doi.org/10.1016/S0021-9991(03)00094-9).

- Remacle, Jean-François, Joseph E. Flaherty, and Mark S. Shephard (2003).
  An adaptive discontinuous Galerkin technique with an orthogonal basis applied to compressible
  flow problems.
  [DOI](https://doi.org/10.1137/S00361445023830)

The domain is [0, 0.25] x [0, 1]. Boundary conditions can be reflective wall boundary conditions on
all boundaries or
- periodic boundary conditions on the left/right boundaries
- Dirichlet boundary conditions on the top/bottom boundaries

This should be used together with `source_terms_rayleigh_taylor_instability`, which is
defined below.
"""
@inline function initial_condition_rayleigh_taylor_instability(x, t,
                                                               equations::CompressibleEulerEquations2D,
                                                               slope=1000)
  tol = 1e2*eps()

  if x[2] < 0.5
    p = 2*x[2] + 1
  else
    p = x[2] + 3/2
  end

  # smooth the discontinuity to avoid ambiguity at element interfaces
  smoothed_heaviside(x, left, right) = left + 0.5*(1 + tanh(slope * x)) * (right-left)
  rho = smoothed_heaviside(x[2] - 0.5, 2.0, 1.0)

  c = sqrt(equations.gamma * p / rho)
  # the velocity is multiplied by sin(pi*y)^6 as in Remacle et al. 2003 to ensure that the
  # initial condition satisfies reflective boundary conditions at the top/bottom boundaries.
  v = -0.025 * c * cos(8*pi*x[1]) * sin(pi*x[2])^6
  u = 0.0

  return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function boundary_condition_dirichlet_top(x, t,
                                                  equations::CompressibleEulerEquations2D)
  rho = 1.0
  u = 0.0
  v = 0.0
  p = 2.5
  return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function boundary_condition_dirichlet_bottom(x, t,
                                                     equations::CompressibleEulerEquations2D)
  rho = 2.0
  u = 0.0
  v = 0.0
  p = 1.0
  return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function source_terms_rayleigh_taylor_instability(u, x, t,
                                                          equations::CompressibleEulerEquations2D)
  g = 1.0
  rho, rho_v1, rho_v2, rho_e = u

  return SVector(0.0, 0.0, g*rho, g*rho_v2)
end

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

volume_flux = flux_ranocha
surface_flux = flux_hlle # flux_hllc
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max=0.5,
                                            alpha_min=0.001,
                                            alpha_smooth=true,
                                            variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux, volume_integral=volume_integral)

num_elements = 12
trees_per_dimension = (num_elements, 4 * num_elements)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=3, initial_refinement_level=0,
                 coordinates_min=(0.0, 0.0), coordinates_max=(0.25, 1.0),
                 periodicity=false)
                 #periodicity=(true, false))

initial_condition = initial_condition_rayleigh_taylor_instability

boundary_conditions = Dict( :x_neg => boundary_condition_slip_wall,
                            :y_neg => boundary_condition_slip_wall,
                            #:y_neg => BoundaryConditionDirichlet(boundary_condition_dirichlet_bottom),
                            #:y_pos => BoundaryConditionDirichlet(boundary_condition_dirichlet_top),
                            :y_pos => boundary_condition_slip_wall,
                            :x_pos => boundary_condition_slip_wall
                            )

#=                            
boundary_conditions_parabolic = Dict( :x_neg => boundary_condition_slip_wall,
                                      :y_neg => boundary_condition_slip_wall,
                                      :y_pos => boundary_condition_slip_wall,
                                      :x_pos => boundary_condition_slip_wall)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions=(boundary_conditions,
                                                                  boundary_conditions_parabolic))
=#

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition, solver;
                                    boundary_conditions=boundary_conditions,
                                    source_terms = source_terms_rayleigh_taylor_instability)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1.0) # p = 2, E = 3, 5, 10
stepsize_callback = StepsizeCallback(cfl=1.3) # p = 3, E = 3, 4, 6, 11

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=true,
                                          variable=Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=0,
                                      med_level =3, med_threshold=0.00125,
                                      max_level =6, max_threshold=0.0025)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=20,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        amr_callback)

###############################################################################
# run the simulation

#=
sol = solve(ode, RDPK3SpFSAL49(); abstol=1.0e-6, reltol=1.0e-6,
            ode_default_options()..., callback=callbacks);
=#

# S = 3, p = 2
dt = 0.00142227114120032641
# S = 10 p = 2
dt = 0.00656249996216502054

LevelCFL = Dict([(42, 42.0)])
Integrator_Mesh_Level_Dict = Dict([(42, 42)])
b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS

Stages = [10, 5, 3]

ode_algorithm = PERK_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/RayleighTaylorInstability/p2/",
                           bS, cEnd,
                           LevelCFL, Integrator_Mesh_Level_Dict)
#=
ode_algorithm = PERK(10, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/RayleighTaylorInstability/",
                     bS, cEnd)                        
=#

# S = 11, p = 3
dt = 0.007080

Stages = [11, 6, 4, 3]

cS2 = 1.0
ode_algorithm = PERK3_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/RayleighTaylorInstability/p3/", cS2,
                            LevelCFL, Integrator_Mesh_Level_Dict)

#ode_algorithm = PERK3(11, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/RayleighTaylorInstability/p3/")

sol = Trixi.solve(ode, ode_algorithm, dt = dt,
                  save_everystep=false, callback=callbacks)



#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
=#

summary_callback() # print the timer summary
plot(sol)

pd = PlotData2D(sol)
plot(pd["rho"], title = "\$ ρ, t_f = 3.0 \$")
plot(getmesh(pd), xlabel = "\$x\$", ylabel="\$y\$", title = "Mesh at \$t_f = 3.0\$")