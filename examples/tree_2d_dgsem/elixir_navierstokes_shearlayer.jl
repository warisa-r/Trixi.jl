
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
InitialRefinement = 5
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

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan; split_form = false)
#ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval,)

amr_indicator = IndicatorLöhner(semi, variable=v1)                                          
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = InitialRefinement,
                                      med_level  = InitialRefinement+1, med_threshold=0.2,
                                      max_level  = InitialRefinement+3, max_threshold=0.5)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=10,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        amr_callback)

###############################################################################
# run the simulation

CFL = 1.0
CFL = 0.96
CFL = 0.61
CFL = 0.35

Integrator_Mesh_Level_Dict = Dict([(5, 4), (6, 3), (7, 2), (8, 1)])

LevelCFL = [0.35, 0.61, 0.96, 1.0]

# 4: dt 0.00156784012855496261
#dt = 0.00156784012855496261 / (2.0^(InitialRefinement - 4)) * CFL
# 8: dt 0.00342847820080351092
# 16: dt 0.00708093033754266813

dt = 0.00156784012855496261 / (2.0^(InitialRefinement - 4))

b1   = 0.5
bS   = 1.0 - b1
cEnd = 0.5/bS
ode_algorithm = PERK_Multi(4, 3, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_NavierStokes_ShearLayer/", 
                           #"/home/daniel/git/MA/Optim_Monomials/SecOrdCone_EiCOS/",
                           bS, cEnd, 
                           LevelCFL, Integrator_Mesh_Level_Dict,
                           stage_callbacks = ())

#=
# S = 8
# handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=4.8)
dt = 0.00342847820080351092 / (2.0^(InitialRefinement - 4))
S = 8
#=
stepsize_callback = StepsizeCallback(cfl=4.8*2*0.9)
dt = 0.00708093033754266813 / (2.0^(InitialRefinement - 4)) * CFL
S = 16


stepsize_callback = StepsizeCallback(cfl=4.8*4*0.5)
dt = 0.013813946938685265 / (2.0^(InitialRefinement - 4)) * CFL
S = 32
=#

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        amr_callback,
                        stepsize_callback)

#=
# S = 3 = p (similar to SSPRK3,3)
S = 3
stepsize_callback = StepsizeCallback(cfl=1.05)
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        amr_callback,
                        stepsize_callback)

dt = 0.000730023539508692935 / (2.0^(InitialRefinement - 4))
=#

ode_algorithm = PERK(S, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_NavierStokes_ShearLayer/", bS, cEnd)
=#

sol = Trixi.solve(ode, ode_algorithm, dt = dt, save_everystep=false, callback=callbacks);


#=
time_int_tol = 1e-6 # InitialRefinement = 4
#time_int_tol = 1e-7 # InitialRefinement = 5
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacks)
=#


summary_callback() # print the timer summary

plot(sol.u)
pd = PlotData2D(sol)
plot(pd["v1"], title = "\$v_x, t=0\$")
plot(getmesh(pd))

plot(pd["v2"])