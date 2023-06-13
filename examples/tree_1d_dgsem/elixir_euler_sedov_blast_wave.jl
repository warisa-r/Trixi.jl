
using OrdinaryDiffEq, Plots, LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations1D)

The Sedov blast wave setup based on Flash
- http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations1D)
  # Set up polar coordinates
  inicenter = SVector(0.0)
  x_norm = x[1] - inicenter[1]
  r = abs(x_norm)

  # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
  r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
  # r0 = 0.5 # = more reasonable setup
  E = 1.0
  p0_inner = 6 * (equations.gamma - 1) * E / (3 * pi * r0)
  p0_outer = 1.0e-5 # = true Sedov setup
  # p0_outer = 1.0e-3 # = more reasonable setup

  # Calculate primitive variables
  rho = 1.0
  v1  = 0.0
  p   = r > r0 ? p0_outer : p0_inner

  return prim2cons(SVector(rho, v1, p), equations)
end
initial_condition = initial_condition_sedov_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux  = flux_chandrashekar
basis = LobattoLegendreBasis(3)
shock_indicator_variable = density_pressure
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=shock_indicator_variable)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0,)
coordinates_max = ( 2.0,)
InitialRef = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=InitialRef,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

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
plotdata = scatter(EigValsReal, EigValsImag, label = "Spectrum")
display(plotdata)
=#

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 12.5)
ode = semidiscretize(semi, tspan)


summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=true,
                                          variable=density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=InitialRef,
                                      max_level=InitialRef+2, max_threshold=0.01)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback, stepsize_callback)

callbacks_No_CFL = CallbackSet(summary_callback,
                              analysis_callback, alive_callback,
                              amr_callback)

###############################################################################
# run the simulation

#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
plot(sol)
=#

plotdata = nothing

ode_algorithm = PERK_Multi(4, 2, 
                           "/home/daniel/git/MA/EigenspectraGeneration/SedovBlast/", 
                           #"/home/daniel/git/MA/EigenspectraGeneration/SedovBlast/Joint/", 
                           1.0, 0.5)

# S_base = 3
dtOptMin = 0.0184927567373961221 / (2.0^(InitialRef - 6))
CFL = 0.26


# S_base = 4
dtOptMin = 0.0245742732349754078 / (2.0^(InitialRef - 6))
#dtOptMin = 0.012899075796546 / (2.0^(InitialRef - 6))
CFL = 0.2


#=
# S_base = 6
dtOptMin = 0.0380284561062580927
=#

sol = Trixi.solve(ode, ode_algorithm,
                  #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  dt = dtOptMin * CFL,
                  save_everystep=false, callback=callbacks_No_CFL);


summary_callback() # print the timer summary

plot(sol)
pd = PlotData1D(sol)
plot!(getmesh(pd))

A = jacobian_ad_forward(semi, 12.5, sol.u[end])

Eigenvalues = eigvals(A)

# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

# Sometimes due to numerical issues some eigenvalues have positive real part, which is erronous (for hyperbolic eqs)
Eigenvalues = Eigenvalues[real(Eigenvalues) .< 0]

EigValsReal = real(Eigenvalues)
EigValsImag = imag(Eigenvalues)


plotdata = scatter!(EigValsReal, EigValsImag, label = "Spectrum")
display(plotdata)