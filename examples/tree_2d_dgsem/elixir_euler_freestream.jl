
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_constant

solver = DGSEM(polydeg=3, surface_flux=flux_ranocha)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000,
                periodicity=true)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_condition_periodic)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

using Plots
plot(sol)

pd = PlotData2D(sol)

V1 = pd.data[2]
V2 = pd.data[3]

V1_rot = copy(V2)
V2_rot = copy(V1)

N = size(V1_rot)[1]
# Rotate counter-clockwise
for i = 1:N
  V1_rot[i, :] = V2[:, i]
  V2_rot[i, :] = V1[:, i]
end

V1_rot_flipped = copy(V1_rot)
V2_rot_flipped = copy(V2_rot)

for i = 1:N
  V1_rot_flipped[i, :] =  V1_rot[N - i + 1, :]
  V2_rot_flipped[i, :] = -V2_rot[N - i + 1, :]
end

using DelimitedFiles

# Export points
writedlm("x.csv", pd.x, ',')
writedlm("y.csv", pd.y, ',')

# Export Velocity field
writedlm("V1.csv", V1, ',')
writedlm("V2.csv", V2, ',')

writedlm("V1_rot.csv", V1_rot, ',')
writedlm("V2_rot.csv", V2_rot, ',')

writedlm("V1_rot_flipped.csv", V1_rot_flipped, ',')
writedlm("V2_rot_flipped.csv", V2_rot_flipped, ',')