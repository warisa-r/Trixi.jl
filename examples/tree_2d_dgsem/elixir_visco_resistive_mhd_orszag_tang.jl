
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
prandtl_number() = 1.0

# Make less diffusive to still have convection-dominated spectra
#mu() = 1e-2
mu() = 1e-3
#eta() = 1e-2
eta() = 1e-3

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
                initial_refinement_level=4,
                n_cells_max=100000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan; split_form = false)

summary_callback = SummaryCallback()

analysis_interval = 200000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=false,
                                          variable=density_pressure)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=3,
                                      med_level =7, med_threshold=0.04,
                                      max_level =9, max_threshold=0.4)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=10, # PERK, DGLDDRK73_C
                           #interval=31, # SSPRK33
                           #interval = 15, # ParsaniKetchesonDeconinck3S53
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)


cfl = 1.9 # p = 2, S = 12
cfl = 1.9 # p = 3, S = 10
cfl = 1.9 # p = 3, S = 6
cfl = 1.9 # p = 3, S = 8

stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        amr_callback,
                        stepsize_callback,
                        save_restart,
                        glm_speed_callback)

###############################################################################
# run the simulation

# S = 3, p = 2 Ref = 4
dt = 0.0161709425439767083

# S = 4, p = 2 Ref = 4
#dt = 0.0274786205467535198

b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS

# TODO: Do also p=3

# S = 4, p = 3, Ref = 4
dt = 0.0155320306832436481
# Series: 4, 6, 11

#=
ode_algorithm = PERK_Multi(3, 2, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p2/",
                           bS, cEnd,
                           LevelCFL, Integrator_Mesh_Level_Dict)


ode_algorithm = PERK(12, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p2/",
                     bS, cEnd)
=#


Stages = [11, 6, 4]
Stages = [10, 6, 4]
Stages = [6, 4, 3]
#Stages = [8, 5, 4]

cS2 = 1.0
ode_algorithm = PERK3_Multi(Stages, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p3/", cS2)

#ode_algorithm = PERK3(6, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p3/")

#for i = 1:1
  mesh = TreeMesh(coordinates_min, coordinates_max,
                  initial_refinement_level=4,
                  n_cells_max=100000)
  
  semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver) 
  
  ode = semidiscretize(semi, tspan; split_form = false)
  sol = Trixi.solve(ode, ode_algorithm, dt = dt,
                    save_everystep=false, callback=callbacks);

#end

#=
#cfl = 1.9 # DGLDDRK73_C Max Level 9, base lvl = 3
#cfl = 0.6 # SSPRK33 Max Level 9, base lvl = 3

#cfl = 1.2 # ParsaniKetchesonDeconinck3S53

stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        amr_callback,
                        stepsize_callback,
                        glm_speed_callback)

for i = 1:11
  mesh = TreeMesh(coordinates_min, coordinates_max,
                  initial_refinement_level=4,
                  n_cells_max=100000)
  
  semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver) 
  
  ode = semidiscretize(semi, tspan; split_form = false)

  #=
  sol = solve(ode, DGLDDRK73_C(;thread = OrdinaryDiffEq.True());
                dt = 1.0,
                ode_default_options()..., callback=callbacks)
  =#

  #=
  sol = solve(ode, ParsaniKetchesonDeconinck3S53(;thread = OrdinaryDiffEq.True());
                dt = 1.0,
                ode_default_options()..., callback=callbacks)                
  =#
  
  #=
  sol = solve(ode, SSPRK33(;thread = OrdinaryDiffEq.True());
              dt = 1.0,
              ode_default_options()..., callback=callbacks);
  =#
end
=#
summary_callback() # print the timer summary


#plot(sol)

pd = PlotData2D(sol)

using Printf

mkpath("out")  # Create output directory automatically

MacroVars_String = "out/MacroVars.vtk"
MacroVars = open(MacroVars_String, "w")

write(MacroVars, "# vtk DataFile Version 3.0\n")
write(MacroVars, "vtk output\n")
write(MacroVars, "ASCII\n")
write(MacroVars, "DATASET STRUCTURED_GRID\n")

Nx = length(pd.x)
Ny = length(pd.y)
NumPoints = Int(Nx * Ny)
NumPointsString = string(Int(Nx * Ny))

write(MacroVars, "DIMENSIONS ", string(Nx), " ", string(Ny), " 1\n")
write(MacroVars, "POINTS ", NumPointsString, " float\n")
for i in 1:Nx
  for j in 1:Ny
    write(MacroVars, "$(pd.x[i]) $(pd.y[j]) 0\n")
  end
end

write(MacroVars, "\n")
write(MacroVars, "POINT_DATA ", NumPointsString, "\n")

#=
write(MacroVars, "SCALARS rho float\n")
rho = pd.data[1]
write(MacroVars, "LOOKUP_TABLE default\n")
for i in 1:Nx
  for j in 1:Ny
      write(MacroVars, string(rho[i,j]), "\n")
  end
end

write(MacroVars, "\n")
write(MacroVars, "SCALARS p float\n")
p = pd.data[5]
write(MacroVars, "LOOKUP_TABLE default\n")
for i in 1:Nx
  for j in 1:Ny
      write(MacroVars, string(p[i,j]), "\n")
  end
end
=#

write(MacroVars, "\n")
write(MacroVars, "VECTORS U float\n")
v1 = pd.data[2]
v2 = pd.data[3]
for i in 1:Nx
  for j in 1:Ny
    write(MacroVars, string(v1[i,j]), " ", string(v2[i,j]), " 0\n")
  end
end
close(MacroVars)


using CairoMakie

x = pd.x
y = pd.y

f(xP, x, y, v1, v2) = Point2f(
  v1[findmin(abs.(xP[1] .- x))[2], findmin(abs.(xP[2] .- y))[2]],
  v2[findmin(abs.(xP[1] .- x))[2], findmin(abs.(xP[2] .- y))[2]]
)

f(xP) = f(xP, x, y, v1, v2)

fig, ax, pl = streamplot(f, 0.0..2*pi, 0.0..2*pi; 
                         stepsize = 1e-3, gridsize = (100, 100),
                         arrow_size = 0)

Plots.plot(pd["rho"], c = :jet, title = "\$ ρ, t_f = 3.0 \$", 
           xticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]),
           yticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]))

plot(pd["p"], c = :jet, title = "\$ p, t_f = 2.0 \$",
     xticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]),
     yticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]))

Plots.plot(getmesh(pd), xlabel = "\$x\$", ylabel="\$y\$", title = "Mesh at \$t_f = 2.0\$",
           xticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]),
           yticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]))