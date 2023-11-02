
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  slope = 15
  amplitude = 0.02
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  v2 = 0.1 * sin(2 * pi * x[1])
  p = 1.0
  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability

#surface_flux = flux_lax_friedrichs
surface_flux = flux_hlle
volume_flux  = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.002,
                                         alpha_min=0.0001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
Refinement = 4
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=Refinement,
                n_cells_max=100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
#tspan = (0.0, 25.0) # Endtime in paper above
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=1.0,
                                          alpha_min=0.0001,
                                          alpha_smooth=false,
                                          variable=Trixi.density)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=Refinement,
                                      med_level=Refinement+3, med_threshold=0.7, # med_level = current level
                                      max_level=Refinement+5, max_threshold=0.9)

amr_callback = AMRCallback(semi, amr_controller,
                           #interval=9,
                           #interval=18, #RDPK3SpFSAL35, SSPRK43
                           #interval=12, #DGLDDRK73_C
                           interval=32, #SSPRK33
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

analysis_interval = 50000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        amr_callback)

CFL_Stability = 1.0
BaseRefinement = 4
# S = 4, p = 3, Ref = 4  
dt = 0.0126464843742724049 * 2.0^(BaseRefinement - Refinement) * CFL_Stability

# S = 3, p = 3
dt = 0.0078961056005209686 * 2.0^(BaseRefinement - Refinement) * CFL_Stability

# S = 16, p = 3, Ref = 4
#dt = 0.0722396842546004376 * 2.0^(BaseRefinement - Refinement) * CFL_Stability

###############################################################################
# run the simulation


Integrator_Mesh_Level_Dict = Dict([(42, 42)])

# Two level
LevelCFL = Dict([(4, 2.0), (4, 1.0), (5, 0.5), (6, 0.25), (7, 1/8), (8, 1/16), (9, 1/32 * 1.1)])

cS2 = 1.0
ode_algorithm = PERK3_Multi(3, 2, "/home/daniel/git/Paper_AMR_PERK/Data/Kelvin_Helmholtz_Euler/Own_SSPRK33_Style/",
                           cS2,
                           LevelCFL, Integrator_Mesh_Level_Dict,
                           stage_callbacks = ())

stepsize_callback = StepsizeCallback(cfl=3.805) # S = 12

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        amr_callback)

#ode_algorithm = PERK3(12, "/home/daniel/git/Paper_AMR_PERK/Data/Kelvin_Helmholtz_Euler/Own_SSPRK33_Style/")

#=
sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks);
=#

callbacksDE = CallbackSet(summary_callback,
                          analysis_callback,
                          amr_callback)

#=
tol = 1e-6 # maximum error tolerance
sol = solve(ode, RDPK3SpFSAL35(); abstol=tol, reltol=tol,
            ode_default_options()..., callback=callbacksDE,
            thread = OrdinaryDiffEq.True());
=#

#=
sol = solve(ode, SSPRK43();
            ode_default_options()..., callback=callbacksDE,
            thread = OrdinaryDiffEq.True());
=#

#=
stepsize_callback = StepsizeCallback(cfl=3.0) # DGLDDRK73_C

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        amr_callback)

sol = solve(ode, DGLDDRK73_C();
            dt = 1.0,
            ode_default_options()..., callback=callbacks,
            thread = OrdinaryDiffEq.True())
=#

stepsize_callback = StepsizeCallback(cfl=1.0) # SSPRK33

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        amr_callback)

sol = solve(ode, SSPRK33();
            dt = 1.0,
            ode_default_options()..., callback=callbacks,
            thread = OrdinaryDiffEq.True())

summary_callback() # print the timer summary

plot(sol)

pd = PlotData2D(sol)
#=
plot(pd["rho"], title = "", colorbar_title = "     \$ρ\$", xlabel = "\$x\$", ylabel = "\$y \$", 
     colorbar_titlefontrotation = 270, colorbar_titlefontsize = 12)
=#
plot(pd["rho"], title = "\$ρ, t_f = 3.0\$", xlabel = "\$x\$", ylabel = "\$y \$")
plot(getmesh(pd), xlabel = "\$x\$", ylabel="\$y\$", title = "Mesh at \$t_f = 3.0\$")

