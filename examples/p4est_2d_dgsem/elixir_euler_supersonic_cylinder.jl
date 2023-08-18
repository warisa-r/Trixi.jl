# Channel flow around a cylinder at Mach 3
#
# Boundary conditions are supersonic Mach 3 inflow at the left portion of the domain
# and supersonic outflow at the right portion of the domain. The top and bottom of the
# channel as well as the cylinder are treated as Euler slip wall boundaries.
# This flow results in strong shock reflections / interactions as well as Kelvin-Helmholtz
# instabilities at later times as two Mach stems form above and below the cylinder.
#
# For complete details on the problem setup see Section 5.7 of the paper:
# - Jean-Luc Guermond, Murtazo Nazarov, Bojan Popov, and Ignacio Tomas (2018)
#   Second-Order Invariant Domain Preserving Approximation of the Euler Equations using Convex Limiting.
#   [DOI: 10.1137/17M1149961](https://doi.org/10.1137/17M1149961)
#
# Keywords: supersonic flow, shock capturing, AMR, unstructured curved mesh, positivity preservation, compressible Euler, 2D

using Downloads: download
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_mach3_flow(x, t, equations::CompressibleEulerEquations2D)
  # set the freestream flow parameters
  rho_freestream = 1.4
  v1 = 3.0
  v2 = 0.0
  p_freestream = 1.0

  prim = SVector(rho_freestream, v1, v2, p_freestream)
  return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach3_flow

# Supersonic inflow boundary condition.
# Calculate the boundary flux entirely from the external solution state, i.e., set
# external solution state values for everything entering the domain.
@inline function boundary_condition_supersonic_inflow(u_inner, normal_direction::AbstractVector, x, t,
                                                      surface_flux_function, equations::CompressibleEulerEquations2D)
  u_boundary = initial_condition_mach3_flow(x, t, equations)
  flux = Trixi.flux(u_boundary, normal_direction, equations)

  return flux
end


# Supersonic outflow boundary condition.
# Calculate the boundary flux entirely from the internal solution state. Analogous to supersonic inflow
# except all the solution state values are set from the internal solution as everything leaves the domain
@inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                            surface_flux_function, equations::CompressibleEulerEquations2D)
  flux = Trixi.flux(u_inner, normal_direction, equations)

  return flux
end

boundary_conditions = Dict( :Bottom  => boundary_condition_slip_wall,
                            :Circle  => boundary_condition_slip_wall,
                            :Top     => boundary_condition_slip_wall,
                            :Right   => boundary_condition_outflow,
                            :Left    => boundary_condition_supersonic_inflow )

volume_flux = flux_ranocha_turbo
surface_flux = flux_lax_friedrichs

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max=0.5,
                                            alpha_min=0.001,
                                            alpha_smooth=true,
                                            variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux, volume_integral=volume_integral)

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
default_mesh_file = joinpath(@__DIR__, "abaqus_cylinder_in_channel.inp")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/a08f78f6b185b63c3baeff911a63f628/raw/addac716ea0541f588b9d2bd3f92f643eb27b88f/abaqus_cylinder_in_channel.inp",
                                      default_mesh_file)
mesh_file = default_mesh_file

mesh = P4estMesh{2}(mesh_file)

nnodes = length(mesh.nodes)
n_elements = last(size(mesh.tree_node_coordinates))
h_min = 42;
h_max = 0;
for k in 1:n_elements
  # pull the four corners numbered as right-handed
  P0 = mesh.tree_node_coordinates[:, 1     , 1     , k]
  P1 = mesh.tree_node_coordinates[:, nnodes, 1     , k]
  P2 = mesh.tree_node_coordinates[:, nnodes, nnodes, k]
  P3 = mesh.tree_node_coordinates[:, 1     , nnodes, k]
  # compute the four side lengths and get the smallest
  L0 = sqrt( sum( (P1-P0).^2 ) )
  L1 = sqrt( sum( (P2-P1).^2 ) )
  L2 = sqrt( sum( (P3-P2).^2 ) )
  L3 = sqrt( sum( (P0-P3).^2 ) )
  h = min(L0, L1, L2, L3)
  if h > h_max 
    h_max = h
  end
  if h < h_min
    h_min = h
  end
end
println("h_min, h_max: ", h_min, " ", h_max)
println("ratio: ", h_max/h_min)

S_min = 4
S_max = 32
N_bins = Int((S_max - S_min)/2) + 1
h_bins = LinRange(h_min, h_max, N_bins)
#bar(1:N_bins, h_bins)

level_u_indices_elements = [Vector{Int64}() for _ in 1:N_bins]
for k in 1:n_elements
  # pull the four corners numbered as right-handed
  P0 = mesh.tree_node_coordinates[:, 1     , 1     , k]
  P1 = mesh.tree_node_coordinates[:, nnodes, 1     , k]
  P2 = mesh.tree_node_coordinates[:, nnodes, nnodes, k]
  P3 = mesh.tree_node_coordinates[:, 1     , nnodes, k]
  # compute the four side lengths and get the smallest
  L0 = sqrt( sum( (P1-P0).^2 ) )
  L1 = sqrt( sum( (P2-P1).^2 ) )
  L2 = sqrt( sum( (P3-P2).^2 ) )
  L3 = sqrt( sum( (P0-P3).^2 ) )
  h = min(L0, L1, L2, L3)

  level = findfirst(x-> x >= h, h_bins)
  append!(level_u_indices_elements[level], k)
end
level_u_indices_elements_count = Vector{Int64}(undef, N_bins)
for i in eachindex(level_u_indices_elements)
  level_u_indices_elements_count[i] = length(level_u_indices_elements[i])
end

bar(1:N_bins, level_u_indices_elements_count)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

@unpack cache = semi
@unpack elements, interfaces, boundaries, mortars = cache

n_interfaces = last(size(interfaces.u))

level_info_interfaces_acc = [Vector{Int64}() for _ in 1:N_bins]
# Determine level for each interface
for interface_id in 1:n_interfaces
  # Get element ids
  element_id_left  = interfaces.neighbor_ids[1, interface_id]

  # pull the four corners numbered as right-handed
  P0 = mesh.tree_node_coordinates[:, 1     , 1     , element_id_left]
  P1 = mesh.tree_node_coordinates[:, nnodes, 1     , element_id_left]
  P2 = mesh.tree_node_coordinates[:, nnodes, nnodes, element_id_left]
  P3 = mesh.tree_node_coordinates[:, 1     , nnodes, element_id_left]
  # compute the four side lengths and get the smallest
  L0 = sqrt( sum( (P1-P0).^2 ) )
  L1 = sqrt( sum( (P2-P1).^2 ) )
  L2 = sqrt( sum( (P3-P2).^2 ) )
  L3 = sqrt( sum( (P0-P3).^2 ) )
  h_left = min(L0, L1, L2, L3)

  element_id_right = interfaces.neighbor_ids[2, interface_id]

  # pull the four corners numbered as right-handed
  P0 = mesh.tree_node_coordinates[:, 1     , 1     , element_id_right]
  P1 = mesh.tree_node_coordinates[:, nnodes, 1     , element_id_right]
  P2 = mesh.tree_node_coordinates[:, nnodes, nnodes, element_id_right]
  P3 = mesh.tree_node_coordinates[:, 1     , nnodes, element_id_right]
  # compute the four side lengths and get the smallest
  L0 = sqrt( sum( (P1-P0).^2 ) )
  L1 = sqrt( sum( (P2-P1).^2 ) )
  L2 = sqrt( sum( (P3-P2).^2 ) )
  L3 = sqrt( sum( (P0-P3).^2 ) )
  h_right = min(L0, L1, L2, L3)

  # Determine level
  h = min(h_left, h_right)
  level = findfirst(x-> x >= h, h_bins)

  for l in level:N_bins
    push!(level_info_interfaces_acc[l], interface_id)
  end
end
@assert length(level_info_interfaces_acc[end]) == 
  n_interfaces "highest level should contain all interfaces"

n_boundaries = last(size(boundaries.u))
level_info_boundaries_acc = [Vector{Int64}() for _ in 1:N_bins]
# For efficient treatment of boundaries we need additional datastructures
n_dims = ndims(mesh) # Spatial dimension
level_info_boundaries_orientation_acc = [[Vector{Int64}() for _ in 1:2*n_dims] for _ in 1:N_bins]

# Determine level for each boundary
for boundary_id in 1:n_boundaries
  # Get element id (boundaries have only one unique associated element)
  element_id = boundaries.neighbor_ids[boundary_id]

  # pull the four corners numbered as right-handed
  P0 = mesh.tree_node_coordinates[:, 1     , 1     , element_id]
  P1 = mesh.tree_node_coordinates[:, nnodes, 1     , element_id]
  P2 = mesh.tree_node_coordinates[:, nnodes, nnodes, element_id]
  P3 = mesh.tree_node_coordinates[:, 1     , nnodes, element_id]
  # compute the four side lengths and get the smallest
  L0 = sqrt( sum( (P1-P0).^2 ) )
  L1 = sqrt( sum( (P2-P1).^2 ) )
  L2 = sqrt( sum( (P3-P2).^2 ) )
  L3 = sqrt( sum( (P0-P3).^2 ) )
  h = min(L0, L1, L2, L3)

  # Determine level
  level = findfirst(x-> x >= h, h_bins)

  # Add to accumulated container
  for l in level:N_bins
    push!(level_info_boundaries_acc[l], boundary_id)
  end
end
@assert length(level_info_boundaries_acc[end]) == 
  n_boundaries "highest level should contain all boundaries"

###############################################################################
# ODE solvers

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

amr_indicator = IndicatorLöhner(semi, variable=Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=0,
                                      med_level=3, med_threshold=0.05,
                                      max_level=5, max_threshold=0.1)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=1,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)
#=
callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback)
=#
callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

# positivity limiter necessary for this example with strong shocks. Very sensitive
# to the order of the limiter variables, pressure must come first.
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-7, 1.0e-6),
                                                     variables=(pressure, Trixi.density))

###############################################################################
# run the simulation
sol = solve(ode, SSPRK43(stage_limiter!);
            ode_default_options()..., callback=callbacks);
summary_callback() # print the timer summary
pd = PlotData2D(sol)
plot(pd["rho"])
plot!(getmesh(pd))