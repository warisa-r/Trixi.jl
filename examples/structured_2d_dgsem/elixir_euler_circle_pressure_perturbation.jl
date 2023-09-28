using OrdinaryDiffEq, Plots
using Trixi

equations = CompressibleEulerEquations2D(1.4)

boundary_conditions = boundary_condition_slip_wall

flux = Trixi.FluxHLL(min_max_speed_davis)
solver = DGSEM(polydeg=2, surface_flux=flux,
               volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha))

r0 = 0.5 # inner radius
r1 = 9.0 # outer radius

function initial_condition_pressure_perturbation(x, t, equations::CompressibleEulerEquations2D)
  #xs = 1.5 # location of the initial disturbance on the x axis
  xs = 0.6 * r1 # location of the initial disturbance on the x axis
  w = 1/8 * r1/2.5 # half width
  p = exp(-log(2) * ((x[1]-xs)^2 + x[2]^2)/w^2) + 1.0
  v1 = 0.0
  v2 = 0.0
  rho = 1.0

  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_pressure_perturbation


f1(xi)  = SVector( r0 + 0.5 * (r1 - r0) * (xi + 1), 0.0) # right line
f2(xi)  = SVector(-r0 - 0.5 * (r1 - r0) * (xi + 1), 0.0) # left line
f3(eta) = SVector(r0 * cos(0.5 * pi * (eta + 1)), r0 * sin(0.5 * pi * (eta + 1))) # inner circle (Bottom line)
f4(eta) = SVector(r1 * cos(0.5 * pi * (eta + 1)), r1 * sin(0.5 * pi * (eta + 1))) # outer circle (Top line)

cells_per_dimension = (192, 128)
mesh = StructuredMesh(cells_per_dimension, (f1, f2, f3, f4), periodicity=false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.9)

callbacks = CallbackSet(analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        summary_callback);
#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
=#

b1 = 0.0
bS = 1 - b1
cEnd = 0.5/bS

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback)

S_min = 4

Add_Levels = 0 # S_max = 4
#=
Add_Levels = 1 # S_max = 6
Add_Levels = 2 # S_max = 8
Add_Levels = 3 # S_max = 10
Add_Levels = 4 # S_max = 12
Add_Levels = 5 # S_max = 14
Add_Levels = 6 # S_max = 16
=#
#=
Add_Levels = 7 # S_max = 18
Add_Levels = 8 # S_max = 20
Add_Levels = 9 # S_max = 22
Add_Levels = 10 # S_max = 24
Add_Levels = 11 # S_max = 26
Add_Levels = 12 # S_max = 28
Add_Levels = 13 # S_max = 30
Add_Levels = 14 # S_max = 32
=#
Integrator_Mesh_Level_Dict = Dict([(1, 1)])
for i = 2:Add_Levels+1
  Integrator_Mesh_Level_Dict[i] = i
end
Integrator_Mesh_Level_Dict
LevelCFL = ones(Add_Levels+1)

ode_algorithm = PERK_Multi(S_min, Add_Levels, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_CEE_Structured/",
                           bS, cEnd,
                           LevelCFL, Integrator_Mesh_Level_Dict)

CFL_PERK = ((4 + 2*Add_Levels)/4)/8

CFL_Stab = 0.47 # S_max = 4
#=
CFL_Stab = 0.48 # S_max = 6
CFL_Stab = 0.48 # S_max = 8
CFL_Stab = 0.47 # S_max = 10
CFL_Stab = 0.47 # S_max = 12
CFL_Stab = 0.47 # S_max = 14
CFL_Stab = 0.46 # S_max = 16
=#
#=
CFL_Stab = 0.37 # S_max = 18
CFL_Stab = 0.36 # S_max = 20
CFL_Stab = 0.33 # S_max = 22
CFL_Stab = 0.31 # S_max = 24
CFL_Stab = 0.28 # S_max = 26
CFL_Stab = 0.27 # S_max = 28
CFL_Stab = 0.25 # S_max = 30
CFL_Stab = 0.24 # S_max = 32
=#

#=
# Standalone checks (many-stage methods)
CFL_Stab = 0.37 # S_max = 18
CFL_Stab = 0.35 # S_max = 20
CFL_Stab = 0.33 # S_max = 22
CFL_Stab = 0.30 # S_max = 24
CFL_Stab = 0.28 # S_max = 26
CFL_Stab = 0.28 # S_max = 28
CFL_Stab = 0.26 # S_max = 30
CFL_Stab = 0.24 # S_max = 32

ode_algorithm = PERK(S_min+2*Add_Levels, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_CEE_Structured/",
                     bS, cEnd)
=#

CFL = CFL_Stab * CFL_PERK

# S = 4, p = 2, NumCells = 12
dt = 0.0830890595862001646 * CFL

sol = Trixi.solve(ode, ode_algorithm, dt = dt, save_everystep=false, callback=callbacks);

pd = PlotData2D(sol)
plot(pd["p"])
plot!(getmesh(pd))