
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
  v1  = x[2] <= 0.5 ? u0 * tanh(k*(x[2]*0.5 - 0.25)) : u0 * tanh(k*(0.75 -x[2]*0.5))
  v2  = u0 * delta * sin(2*pi*(x[1]*0.5 + 0.25))
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

tspan = (0.0, 1)
ode = semidiscretize(semi, tspan; split_form = false)
#ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval,)

amr_indicator = IndicatorLöhner(semi, variable=v_x)                                          
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

CFL = 0.57
CFL = 0.35 # Three refinements
# 4: dt 0.00156784012855496261
dt = 0.00156784012855496261 / (2.0^(InitialRefinement - 4)) * CFL
# 8: dt 0.00342847820080351092
# 16: dt 0.00708093033754266813

b1   = 0.5
bS   = 1.0 - b1
cEnd = 0.5/bS
ode_algorithm = PERK_Multi(4, 3, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_NavierStokes_ShearLayer/", 
                           #"/home/daniel/git/MA/Optim_Monomials/SecOrdCone_EiCOS/",
                           bS, cEnd, stage_callbacks = ())


# S = 8
CFL = 0.25 * 1.0
CFL = 0.125 * 1.0

dt = 0.00342847820080351092 / (2.0^(InitialRefinement - 4)) * CFL
S = 8

#=
CFL = 0.25 * 0.8
dt = 0.00708093033754266813 / (2.0^(InitialRefinement - 4)) * CFL
S = 16

CFL = 0.25 * 0.4
dt = 0.013813946938685265 / (2.0^(InitialRefinement - 4)) * CFL
S = 32
=#

ode_algorithm = PERK(S, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_NavierStokes_ShearLayer/", bS, cEnd)


sol = Trixi.solve(ode, ode_algorithm, dt = dt, save_everystep=false, callback=callbacks);


#=
A = jacobian_ad_forward(semi, 1.0, sol.u[end])

Eigenvalues = eigvals(A)

# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

# Sometimes due to numerical issues some eigenvalues have positive real part, which is erronous (for hyperbolic eqs)
Eigenvalues = Eigenvalues[real(Eigenvalues) .< 0]

EigValsReal = real(Eigenvalues)
EigValsImag = imag(Eigenvalues)

plotdata = nothing
plotdata = scatter(EigValsReal, EigValsImag, label = "Spectrum")
display(plotdata)
=#

#=
time_int_tol = 1e-6 # InitialRefinement = 4
#time_int_tol = 1e-7 # InitialRefinement = 5
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacks)
=#

#=
sol = solve(ode, SSPRK33(), dt=4e-5,
            save_everystep=false, callback=callbacks)
=#

summary_callback() # print the timer summary

plot(sol)
pd = PlotData2D(sol)
plot(pd["v1"])
plot!(getmesh(pd))

plot(pd["v2"])