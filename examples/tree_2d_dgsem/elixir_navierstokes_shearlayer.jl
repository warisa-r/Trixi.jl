
using OrdinaryDiffEq, Plots
using Trixi, LinearAlgebra

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

# TODO: parabolic; unify names of these accessor functions
prandtl_number() = 0.72
mu() = 1.0/3.0 * 10^(-5)

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu=mu(),
                                                          Prandtl=prandtl_number())
"""
A compressible version of the double shear layer initial condition. Adapted from
Brown and Minion (1995).

- David L. Brown and Michael L. Minion (1995)
  Performance of Under-resolved Two-Dimensional Incompressible Flow Simulations.
  [DOI: 10.1006/jcph.1995.1205](https://doi.org/10.1006/jcph.1995.1205)
"""
function initial_condition_shear_layer(x, t, equations::CompressibleEulerEquations2D)
  # Shear layer parameters
  k = 80
  delta = 0.05
  u0 = 1.0
  
  Ms = 0.1 # maximum Mach number

  rho = 1.0
  v1  = x[2] <= 0.5 ? u0 * tanh(k*(x[2] - 0.25)) : u0 * tanh(k*(0.75 -x[2]))
  v2  = u0 * delta * sin(2*pi*(x[1]+ 0.25))
  p   = (u0 / Ms)^2 * rho / equations.gamma # scaling to get Ms

  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_shear_layer

volume_flux = flux_ranocha
solver = DGSEM(polydeg=3, surface_flux=flux_hllc,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
InitialRefinement = 3
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=InitialRefinement,
                n_cells_max=100_000)


semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver)
#=
A = jacobian_ad_forward(semi)

Eigenvalues = eigvals(A)

# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

# Sometimes due to numerical issues some eigenvalues have positive real part, which is erronous (for hyperbolic eqs)
Eigenvalues = Eigenvalues[real(Eigenvalues) .< 0]

EigValsReal = real(Eigenvalues)
EigValsImag = imag(Eigenvalues)

plotdata = nothing
plotdata = scatter!(EigValsReal, EigValsImag, label = "Spectrum")
display(plotdata)
=#

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.8)
#tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan; split_form = false)
#ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval,)

amr_indicator = IndicatorLöhner(semi, variable=v1)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = InitialRefinement,
                                      med_level  = InitialRefinement+4, med_threshold=0.15,
                                      max_level  = InitialRefinement+6, max_threshold=0.45)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=20,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        amr_callback)

###############################################################################
# run the simulation

Integrator_Mesh_Level_Dict = Dict([(42, 42)])


# S = 3 base

# Levels 3-7
LevelCFL = Dict([(3, 4.0), (4, 2.0), (5, 1.0 * 0.9), (6, 0.5 * 0.9), (7, 0.25 * 0.9)])
# Longer tspan
#LevelCFL = Dict([(3, 4.0), (4, 2.0), (5, 1.0 * 0.9), (6, 0.5 * 0.9), (7, 0.25 * 0.8)])

# Levels 3-9
LevelCFL = Dict([(3, 4.0), (4, 2.0), (5, 1.0 * 0.9), (6, 0.5 * 0.9), (7, 0.25 * 0.9), (8, 0.125 * 0.8), (9, 1/16 * 0.8)])

# S= 3, p = 2
dt = 0.000896571863268036445 / (2.0^(InitialRefinement - 4))

b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS

ode_algorithm = PERK_Multi(3, 2, "/home/daniel/git/Paper_AMR_PERK/Data/2D_NavierStokes_ShearLayer/", 
                           #"/home/daniel/git/MA/Optim_Monomials/SecOrdCone_EiCOS/",
                           bS, cEnd, 
                           LevelCFL, Integrator_Mesh_Level_Dict,
                           stage_callbacks = ())


#=
# S = 6
# handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=3.6)
S = 6

#stepsize_callback = StepsizeCallback(cfl=7.2)
#S = 12


callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        amr_callback,
                        stepsize_callback)

ode_algorithm = PERK(S, "/home/daniel/git/Paper_AMR_PERK/Data/2D_NavierStokes_ShearLayer/", bS, cEnd)
=#

sol = Trixi.solve(ode, ode_algorithm, dt = dt, save_everystep=false, callback=callbacks);


#=
time_int_tol = 1e-6 # InitialRefinement = 4
#time_int_tol = 1e-7 # InitialRefinement = 5
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacks)
=#


summary_callback() # print the timer summary

plot(sol)
pd = PlotData2D(sol)
plot(pd["v1"], title = "\$v_x, t_f=0.8\$")
plot(pd["v2"], title = "\$v_x, t=0\$")
plot(getmesh(pd), xlabel = "\$x\$", ylabel="\$y\$", title = "Mesh at \$t_f = 0.8\$")

plot(pd["v2"])