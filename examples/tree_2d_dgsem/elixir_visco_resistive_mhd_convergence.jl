
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
prandtl_number() = 1.0

mu() = 1e-5
eta() = 1e-5

gamma = 5/3
equations = IdealGlmMhdEquations2D(gamma)
equations_parabolic = ViscoResistiveMhdDiffusion2D(equations, mu = mu(),
                                                   Prandtl = prandtl_number(),
                                                   eta = eta(),
                                                   gradient_variables = GradientVariablesPrimitive())

"""
initial_condition_convergence_test(x, t, equations::IdealGlmMhdEquations2D)

Manufactured solution due to Andrew Winters.
"""
function initial_condition_convergence_test(x, t, equations::IdealGlmMhdEquations2D)
  alpha = 2.0 * pi * (x[1] + x[2]) - 4.0*t
  phi = sin(alpha) + 4.0

  rho = phi
  rho_v1  = phi
  rho_v2  = phi
  rho_v3  = 0.0
  rho_e   = 2.0*phi^2
  B1  = phi
  B2  = -phi
  B3  = 0.0
  psi = 0.0
  return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi)
end
initial_condition = initial_condition_convergence_test

function source_terms(u, x, t, equations)
  alpha = 2.0*pi*(x[1] + x[2])-4.0*t

  phi = sin(alpha) + 4.0
  phi_t = -4.0*cos(alpha)
  phi_x = 2.0*pi*cos(alpha)
  phi_xx = -4.0*pi^2*sin(alpha)

  du1 = phi_t+2.0*phi_x
  du2 = phi_t+4.0*phi*phi_x+phi_x
  du3 = du2
  du4 = 0.0
  du5 = 4.0*phi*phi_t+16.0*phi*phi_x-2.0*phi_x-4.0*eta()*(phi_x^2+phi*phi_xx)-4.0*mu()/prandtl_number()*phi_xx
  du6 = du1-2.0*eta()*phi_xx
  du7 = -du6
  du8 = 0.0
  du9 = 0.0

  return SVector(du1, du2, du3, du4, du5, du6, du7, du8, du9)
end

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=100000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver,
                                             source_terms = source_terms)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

cfl = 0.8
stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        glm_speed_callback)

sol = solve(ode, SSPRK104(;thread = OrdinaryDiffEq.True()), dt = 42.0,
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary

#plot(sol)
