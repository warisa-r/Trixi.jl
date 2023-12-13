
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
prandtl_number() = 1.0

mu() = 1e-2
eta() = 1e-2

gamma = 5/3
equations = IdealGlmMhdEquations2D(gamma)
equations_parabolic = ViscoResistiveMhdDiffusion2D(equations, mu = mu(),
                                          Prandtl = prandtl_number(),
                                          eta = eta(),
                                          gradient_variables = GradientVariablesPrimitive())

"""
    initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations2D)

The classical Orszag-Tang vortex test case. Here, the setup is taken from
- https://onlinelibrary.wiley.com/doi/pdf/10.1002/fld.4681
"""
function initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations2D)
  rho = 1.0
  v1 = -2 * sqrt(pi) * sin(x[2])
  v2 =  2 * sqrt(pi) * sin(x[1])
  v3 = 0.0
  p = 15/4 + 0.25 * cos(4*x[1]) + 0.8 * cos(2*x[1])*cos(x[2]) - cos(x[1])*cos(x[2]) + 0.25 * cos(2*x[2])
  B1 = -sin(x[2])
  B2 =  sin(2.0*x[1])
  B3 = 0.0
  psi = 0.0
  return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end
initial_condition = initial_condition_orszag_tang

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
volume_flux  = (flux_hindenlang_gassner, flux_nonconservative_powell)
basis = LobattoLegendreBasis(3)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (2*pi, 2*pi)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=7,
                n_cells_max=100000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, analysis_errors = Symbol[])

cfl = 1.7
stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        glm_speed_callback)

sol = solve(ode, DGLDDRK73_C(;thread = OrdinaryDiffEq.True()), dt = 42.0,
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary

plot(sol)

pd = PlotData2D(sol)

V1 = pd.data[2]
V2 = pd.data[3]

B1 = pd.data[6]
B2 = pd.data[7]

using DelimitedFiles

# Export points
writedlm("x.csv", pd.x, ',')
writedlm("y.csv", pd.y, ',')

# Export Velocity field
writedlm("V1.csv", V1, ',')
writedlm("V2.csv", V2, ',')

# Export Magnetic field
writedlm("B1.csv", B1, ',')
writedlm("B2.csv", B2, ',')
