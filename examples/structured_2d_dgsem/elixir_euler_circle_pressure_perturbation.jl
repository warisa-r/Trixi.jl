using OrdinaryDiffEq, Plots
using Trixi

equations = CompressibleEulerEquations2D(1.4)

function initial_condition_pressure_perturbation(x, t, equations::CompressibleEulerEquations2D)
  xs = 1.5 # location of the initial disturbance on the x axis
  w = 1/8 # half width
  p = exp(-log(2) * ((x[1]-xs)^2 + x[2]^2)/w^2) + 1.0
  v1 = 0.0
  v2 = 0.0
  rho = 1.0

  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_pressure_perturbation

boundary_conditions = boundary_condition_slip_wall

solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs,
               volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha))

r0 = 0.5 # inner radius
r1 = 5.0 # outer radius
f1(xi)  = SVector( r0 + 0.5 * (r1 - r0) * (xi + 1), 0.0) # right line
f2(xi)  = SVector(-r0 - 0.5 * (r1 - r0) * (xi + 1), 0.0) # left line
f3(eta) = SVector(r0 * cos(0.5 * pi * (eta + 1)), r0 * sin(0.5 * pi * (eta + 1))) # inner circle (Bottom line)
f4(eta) = SVector(r1 * cos(0.5 * pi * (eta + 1)), r1 * sin(0.5 * pi * (eta + 1))) # outer circle (Top line)

cells_per_dimension = (32, 16)
mesh = StructuredMesh(cells_per_dimension, (f1, f2, f3, f4), periodicity=false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

tspan = (0.0, 5)
ode = semidiscretize(semi, tspan)

analysis_interval = 100
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

Integrator_Mesh_Level_Dict = Dict([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)])
LevelCFL = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback)

S_min = 4
Add_Levels = 6
ode_algorithm = PERK_Multi(S_min, Add_Levels, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_CEE_Structured/",
                           bS, cEnd,
                           LevelCFL, Integrator_Mesh_Level_Dict)

_, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)

nnodes = length(dg.basis.nodes)
n_elements = nelements(dg, cache)
n_dims = ndims(mesh) # Spatial dimension

h_min = 42;
h_max = 0;

h_min_per_element = zeros(n_elements)

for element_id in 1:n_elements
  # pull the four corners numbered as right-handed
  P0 = cache.elements.node_coordinates[:, 1     , 1     , element_id]
  P1 = cache.elements.node_coordinates[:, nnodes, 1     , element_id]
  P2 = cache.elements.node_coordinates[:, nnodes, nnodes, element_id]
  P3 = cache.elements.node_coordinates[:, 1     , nnodes, element_id]
  # compute the four side lengths and get the smallest
  L0 = sqrt( sum( (P1-P0).^2 ) )
  L1 = sqrt( sum( (P2-P1).^2 ) )
  L2 = sqrt( sum( (P3-P2).^2 ) )
  L3 = sqrt( sum( (P0-P3).^2 ) )
  h = min(L0, L1, L2, L3)
  h_min_per_element[element_id] = h
  if h > h_max 
    h_max = h
  end
  if h < h_min
    h_min = h
  end
end

n_levels = Add_Levels + 1 # Linearly increasing levels
#=
if n_levels == 1
  h_bins = [h_max]
else
  h_bins = LinRange(h_min, h_max, n_levels)
end
=#
h_bins = LinRange(h_min, h_max, n_levels+1)

h_min_ref = 0.19509032201612822
CFL_grid = h_bins[end-1] / h_min_ref

CFL_Stab = 0.4
CFL = CFL_grid * CFL_Stab

# S = 4, p = 2, NumCells = 8
dt = 0.044526100157963813 * CFL                           

sol = Trixi.solve(ode, ode_algorithm, dt = dt, save_everystep=false, callback=callbacks);

pd = PlotData2D(sol)
plot(pd["p"])
plot(pd["rho"])
plot!(getmesh(pd))