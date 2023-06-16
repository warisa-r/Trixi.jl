# TODO: Currently hard-coded to second order accurate methods!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

### Based on file "methods_2N.jl", use this as a template for P-ERK RK methods

"""
    PRK()

The following structures and methods provide a minimal implementation of
the 'ForwardEulerTwoStep' temporal integrator.

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{54, 43} methods.
"""

mutable struct PRK
  Stages::Int
  R::Int # Number partitions

  A::Array{Float64} # Butcher Tableaus
  b::Vector{Float64}
  c::Vector{Float64}

  # Constructor for previously computed A Coeffs
  function PRK()

    Stages = 4
    R = 2
    # Store from a_{3,j} row-wise, until a_{i,i-1} (explicit method)
    A = zeros(R, Stages-2, Stages-1)
    A[1, 1, 1] = 0.656796879144715
    A[1, 1, 2] = 0.343201934078188
    A[1, 2, 1] = 0.537422905930307
    A[1, 2, 2] = 0.251418725645745
    A[1, 2, 3] = 0.211157181642298

    A[2, 1, 1] = 1.0
    A[2, 2, 1] = 1.0

    b = [0.499999406841893; 0.220692431117136; 0.151540432848441; 0.127767729206276]
    c = [1; 1; 1; 1]


    for r = 1:R
      display(A[r, :, :])
    end
    newPRK = new(Stages, R, A, b, c)

    return newPRK
  end

end # struct PRK


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct PRK_IntegratorOptions{Callback}
  callback::Callback # callbacks; used in Trixi
  adaptive::Bool # whether the algorithm is adaptive; ignored
  dtmax::Float64 # ignored
  maxiters::Int # maximal numer of time steps
  tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function PRK_IntegratorOptions(callback, tspan; maxiters=typemax(Int), kwargs...)
  PRK_IntegratorOptions{typeof(callback)}(callback, false, Inf, maxiters, [last(tspan)])
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PRK_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, PRK_IntegratorOptions}
  u::uType
  du::uType
  u_tmp::uType
  t::RealT
  dt::RealT # current time step
  dtcache::RealT # ignored
  iter::Int # current number of time steps (iteration)
  p::Params # will be the semidiscretization from Trixi
  sol::Sol # faked
  f::F # This is the ODE function from ODEProblem; within Trixi, this amounts to "rhs!"
  alg::Alg # This is our own class written above; Abbreviation for ALGorithm
  opts::PRK_IntegratorOptions
  finalstep::Bool # added for convenience
  # Variables managing level-depending integration
  level_info_elements_acc::Vector{Vector{Int64}}
  level_info_interfaces_acc::Vector{Vector{Int64}}
  level_info_boundaries_acc::Vector{Vector{Int64}}
  level_info_mortars_acc::Vector{Vector{Int64}}
  level_u_indices_elements::Vector{Vector{Int64}}
end

# Forward integrator.destats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PRK_Integrator, field::Symbol)
  if field === :destats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PRK;
               dt, callback=nothing, kwargs...)

  u0 = copy(ode.u0)
  du = similar(u0)
  u_tmp = similar(u0)

  t0 = first(ode.tspan)
  iter = 0

  ### Set datastructures for handling of level-dependent integration ###
  @unpack mesh, cache = ode.p
  @unpack elements, interfaces, boundaries = cache

  n_elements   = length(elements.cell_ids)
  n_interfaces = length(interfaces.orientations)
  n_boundaries = length(boundaries.orientations) # TODO Not sure if adequate

  min_level = minimum_level(mesh.tree)
  max_level = maximum_level(mesh.tree)
  n_levels = max_level - min_level + 1

  # NOTE: Next-to-fine is NOT integrated with fine integrator
  
  # Initialize storage for level-wise information
  # Set-like datastructures more suited then vectors (Especially for interfaces)
  level_info_elements     = [Vector{Int}() for _ in 1:n_levels]
  level_info_elements_acc = [Vector{Int}() for _ in 1:n_levels]

  # Determine level for each element
  for element_id in 1:n_elements
    # Determine level
    level = mesh.tree.levels[elements.cell_ids[element_id]]
    # Convert to level id
    level_id = max_level + 1 - level

    push!(level_info_elements[level_id], element_id)
    # Add to accumulated container
    for l in level_id:n_levels
      push!(level_info_elements_acc[l], element_id)
    end
  end
  @assert length(level_info_elements_acc[end]) == 
    n_elements "highest level should contain all elements"


  # Use sets first to avoid double storage of interfaces
  level_info_interfaces_set = [Set{Int}() for _ in 1:n_levels]
  level_info_interfaces_set_acc = [Set{Int}() for _ in 1:n_levels]
  # Determine level for each interface
  for interface_id in 1:n_interfaces
    # Get element ids
    element_id_left  = interfaces.neighbor_ids[1, interface_id]
    element_id_right = interfaces.neighbor_ids[2, interface_id]

    # Determine level
    level_left  = mesh.tree.levels[elements.cell_ids[element_id_left]]
    level_right = mesh.tree.levels[elements.cell_ids[element_id_right]]

    # Convert to level id
    level_id_left  = max_level + 1 - level_left
    level_id_right = max_level + 1 - level_right

    push!(level_info_interfaces_set[level_id_left], interface_id)
    push!(level_info_interfaces_set[level_id_right], interface_id)

    # Add to accumulated container
    for l in level_id_left:n_levels
      push!(level_info_interfaces_set_acc[l], interface_id)
    end
    for l in level_id_right:n_levels
      push!(level_info_interfaces_set_acc[l], interface_id)
    end
  end

  level_info_interfaces = [Vector{Int}() for _ in 1:n_levels]
  for level in 1:n_levels
    level_info_interfaces[level] = sort(collect(level_info_interfaces_set[level]))
  end

  # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
  level_info_interfaces_acc = [Vector{Int}() for _ in 1:n_levels]
  for level in 1:n_levels
    level_info_interfaces_acc[level] = sort(collect(level_info_interfaces_set_acc[level]))
  end
  @assert length(level_info_interfaces_acc[end]) == 
    n_interfaces "highest level should contain all interfaces"


  # Use sets first to avoid double storage of boundaries
  level_info_boundaries_set_acc = [Set{Int}() for _ in 1:n_levels]
  # Determine level for each boundary
  for boundary_id in 1:n_boundaries
    # Get element ids
    element_id_left  = boundaries.neighbor_ids[1, boundary_id]
    element_id_right = boundaries.neighbor_ids[2, boundary_id]

    # Determine level
    level_left  = mesh.tree.levels[elements.cell_ids[element_id_left]]
    level_right = mesh.tree.levels[elements.cell_ids[element_id_right]]

    # Convert to level id
    level_id_left  = max_level + 1 - level_left
    level_id_right = max_level + 1 - level_right

    # Add to accumulated container
    for l in level_id_left:n_levels
      push!(level_info_boundaries_set_acc[l], boundary_id)
    end
    for l in level_id_right:n_levels
      push!(level_info_boundaries_set_acc[l], boundary_id)
    end
  end

  # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
  level_info_boundaries_acc = [Vector{Int}() for _ in 1:n_levels]
  for level in 1:n_levels
    level_info_boundaries_acc[level] = sort(collect(level_info_boundaries_set_acc[level]))
  end
  @assert length(level_info_boundaries_acc[end]) == 
    n_boundaries "highest level should contain all boundaries"

  level_info_mortars_acc = [Vector{Int}() for _ in 1:n_levels]

  dimensions = ndims(mesh.tree) # Spatial dimension

  if dimensions > 1
    # Determine level for each mortar
    # Since mortars belong by definition to two levels, theoretically we have to
    # add them twice: Once for each level of its neighboring elements. However,
    # as we store the accumulated mortar ids, we only need to consider the one of
    # the small neighbors (here: the lower one), is it has the higher level and
    # thus the lower level id.

    @unpack mortars = cache
    n_mortars = length(mortars.orientations)

    for mortar_id in 1:n_mortars
      # Get element ids
      element_id_lower = mortars.neighbor_ids[1, mortar_id]

      # Determine level
      level_lower = mesh.tree.levels[elements.cell_ids[element_id_lower]]

      # Convert to level id
      level_id_lower = max_level + 1 - level_lower

      # Add to accumulated container
      for l in level_id_lower:n_levels
        push!(level_info_mortars_acc[l], mortar_id)
      end
    end
    @assert length(level_info_mortars_acc[end]) == 
      n_mortars "highest level should contain all mortars"
  end
  
  
  # NOTE: Next-to-fine is also integrated with fine integrator
  #=
  # Initialize storage for level-wise information
  # Set-like datastructures more suited then vectors
  level_info_elements_set     = [Set{Int}() for _ in 1:n_levels]
  level_info_elements_set_acc = [Set{Int}() for _ in 1:n_levels]
  # Loop over interfaces to have access to its neighbors
  for interface_id in 1:n_interfaces
    # Get element ids
    element_id_left  = interfaces.neighbor_ids[1, interface_id]
    element_id_right = interfaces.neighbor_ids[2, interface_id]

    # Determine level
    level_left  = mesh.tree.levels[elements.cell_ids[element_id_left]]
    level_right = mesh.tree.levels[elements.cell_ids[element_id_right]]

    # Neighbors of finer cells should be integrated with same integrator
    ode_level = max(level_left, level_right)

    # Convert to level id
    level_id = max_level + 1 - ode_level

    # Assign elements according to their neighbors
    push!(level_info_elements_set[level_id], element_id_left)
    push!(level_info_elements_set[level_id], element_id_right)
    # Add to accumulated container
    for l in level_id:n_levels
      push!(level_info_elements_set_acc[l], element_id_left)
      push!(level_info_elements_set_acc[l], element_id_right)
    end
  end
  
  # Turn sets into sorted vectors to have (hopefully) faster accesses due to contiguous storage
  level_info_elements = [Vector{Int}() for _ in 1:n_levels]
  for level in 1:n_levels
    # Make sure elements are only stored once: In the finest level
    for fine_level in 1:level-1
      level_info_elements_set[level] = setdiff(level_info_elements_set[level], 
                                               level_info_elements_set[fine_level])
    end

    level_info_elements[level] = sort(collect(level_info_elements_set[level]))
  end

  # Set up dictionary to set later ODE level for interfaces
  element_ODE_level_dict = Dict{Int, Int}()
  for level in 1:n_levels
    for element_id in level_info_elements[level]
      push!(element_ODE_level_dict, element_id=>level)
    end
  end
  display(element_ODE_level_dict); println()

  level_info_elements_acc = [Vector{Int}() for _ in 1:n_levels]
  for level in 1:n_levels
    level_info_elements_acc[level] = sort(collect(level_info_elements_set_acc[level]))
  end
  @assert length(level_info_elements_acc[end]) == 
    n_elements "highest level should contain all elements"

  # Use sets first to avoid double storage of interfaces
  level_info_interfaces_set = [Set{Int}() for _ in 1:n_levels]
  level_info_interfaces_set_acc = [Set{Int}() for _ in 1:n_levels]
  # Determine ODE level for each interface
  for interface_id in 1:n_interfaces
    # Get element ids
    element_id_left  = interfaces.neighbor_ids[1, interface_id]
    element_id_right = interfaces.neighbor_ids[2, interface_id]

    # Interface neighboring two distinct ODE levels belong to fines one
    ode_level = min(get(element_ODE_level_dict, element_id_left, -1), 
                    get(element_ODE_level_dict, element_id_right, -1))
    
    #=
    ode_level = max(get(element_ODE_level_dict, element_id_left, -1), 
                    get(element_ODE_level_dict, element_id_right, -1))                              
    =#

    @assert ode_level != -1 "Errors in datastructures for ODE level assignment"
    
    push!(level_info_interfaces_set[ode_level], interface_id)

    #=
    # TODO: Not sure if correct in this setting
    level_left  = mesh.tree.levels[elements.cell_ids[element_id_left]]
    level_right = mesh.tree.levels[elements.cell_ids[element_id_right]]
    level_id_left  = max_level + 1 - level_left
    level_id_right = max_level + 1 - level_right
    push!(level_info_interfaces_set[level_id_left], interface_id)
    push!(level_info_interfaces_set[level_id_right], interface_id)
    =#

    # Add to accumulated container
    for l in ode_level:n_levels
      push!(level_info_interfaces_set_acc[l], interface_id)
    end
  end

  # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
  level_info_interfaces = [Vector{Int}() for _ in 1:n_levels]
  for level in 1:n_levels
    # Make sure elements are only stored once: In the finest level
    for fine_level in 1:level-1
      level_info_interfaces_set[level] = setdiff(level_info_interfaces_set[level], 
                                                 level_info_interfaces_set[fine_level])
    end

    level_info_interfaces[level] = sort(collect(level_info_interfaces_set[level]))
  end

  #=
  level_info_interfaces = [Vector{Int}() for _ in 1:n_levels]
  for level in 1:n_levels
    level_info_interfaces[level] = sort(collect(level_info_interfaces_set[level]))
  end 
  =#

  level_info_interfaces_acc = [Vector{Int}() for _ in 1:n_levels]
  for level in 1:n_levels
    level_info_interfaces_acc[level] = sort(collect(level_info_interfaces_set_acc[level]))
  end
  @assert length(level_info_interfaces_acc[end]) == 
    n_interfaces "highest level should contain all interfaces"

  
  # Use sets first to avoid double storage of boundaries
  level_info_boundaries_set_acc = [Set{Int}() for _ in 1:n_levels]
  # Determine level for each boundary
  for boundary_id in 1:n_boundaries
    # Get element ids
    element_id_left  = boundaries.neighbor_ids[1, boundary_id]
    element_id_right = boundaries.neighbor_ids[2, boundary_id]

    # Determine level
    level_left  = mesh.tree.levels[elements.cell_ids[element_id_left]]
    level_right = mesh.tree.levels[elements.cell_ids[element_id_right]]

    # Convert to level id
    level_id_left  = max_level + 1 - level_left
    level_id_right = max_level + 1 - level_right

    # Add to accumulated container
    for l in level_id_left:n_levels
      push!(level_info_boundaries_set_acc[l], boundary_id)
    end
    for l in level_id_right:n_levels
      push!(level_info_boundaries_set_acc[l], boundary_id)
    end
  end

  # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
  level_info_boundaries_acc = [Vector{Int}() for _ in 1:n_levels]
  for level in 1:n_levels
    level_info_boundaries_acc[level] = sort(collect(level_info_boundaries_set_acc[level]))
  end
  @assert length(level_info_boundaries_acc[end]) == n_boundaries "highest level should contain all boundaries"


  # TODO: Mortars need probably to be reconsidered! (sets, level-assignment, ...)
  level_info_mortars_acc = [Vector{Int}() for _ in 1:n_levels]
  dimensions = ndims(mesh.tree) # Spatial dimension
  if dimensions > 1
    # Determine level for each mortar
    # Since mortars belong by definition to two levels, theoretically we have to
    # add them twice: Once for each level of its neighboring elements. However,
    # as we store the accumulated mortar ids, we only need to consider the one of
    # the small neighbors (here: the lower one), is it has the higher level and
    # thus the lower level id.

    @unpack mortars = cache
    n_mortars = length(mortars.orientations)

    for mortar_id in 1:n_mortars
      # Get element ids
      element_id_lower = mortars.neighbor_ids[1, mortar_id]

      # Determine level
      level_lower = mesh.tree.levels[elements.cell_ids[element_id_lower]]

      # Convert to level id
      level_id_lower = max_level + 1 - level_lower

      # Add to accumulated container
      for l in level_id_lower:n_levels
        push!(level_info_mortars_acc[l], mortar_id)
      end
    end
    @assert length(level_info_mortars_acc[end]) == n_mortars "highest level should contain all mortars"
  end
  =#
  
  println("n_elements: ", n_elements)
  println("\nn_interfaces: ", n_interfaces)

  println("level_info_elements:")
  display(level_info_elements); println()
  println("level_info_elements_acc:")
  display(level_info_elements_acc); println()

  println("level_info_interfaces:")
  display(level_info_interfaces); println()
  println("level_info_interfaces_acc:")
  display(level_info_interfaces_acc); println()

  println("level_info_boundaries_acc:")
  display(level_info_boundaries_acc); println()

  println("level_info_mortars_acc:")
  display(level_info_mortars_acc); println()

  
  # Set initial distribution of DG Base function coefficients 
  @unpack equations, solver = ode.p
  u = wrap_array(u0, mesh, equations, solver, cache)

  level_u_indices_elements = [Vector{Int}() for _ in 1:n_levels]

  # Have if outside for performance reasons (this is also used in the AMR calls)
  if dimensions == 1
    for level in 1:n_levels
      for element_id in level_info_elements[level]
        indices = vec(transpose(LinearIndices(u)[:, :, element_id]))
        append!(level_u_indices_elements[level], indices)
      end
    end
  elseif dimensions == 2
    for level in 1:n_levels
      for element_id in level_info_elements[level]
        indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, element_id]))
        append!(level_u_indices_elements[level], indices)
      end
    end
  end
  display(level_u_indices_elements); println()
  

  #=
  # CARE: Hard-coded "artificial" mesh splitting in two halves
  @assert n_elements % 4 == 0
  level_info_elements = [Vector(Int(n_elements/2) + 1:Int(3*n_elements/4)),
                          vcat(Vector(Int(n_elements/4) + 1:Int(n_elements/2)), 
                              Vector(Int(3*n_elements/4) + 1:n_elements)),
                          Vector(1:Int(n_elements/4))]
  level_info_elements_acc = [level_info_elements[1], 
                              vcat(level_info_elements[1], level_info_elements[2]),
                              Vector(1:n_elements)]

  element_ODE_level_dict = Dict{Int, Int}()
  for level in 1:length(level_info_elements)
    for element_id in level_info_elements[level]
      push!(element_ODE_level_dict, element_id=>level)
    end
  end
  display(element_ODE_level_dict); println()                              

  level_info_interfaces_set_acc = [Set{Int}() for _ in 1:length(level_info_elements)]
  # Determine ODE level for each interface
  for interface_id in 1:n_interfaces
    # Get element ids
    element_id_left  = interfaces.neighbor_ids[1, interface_id]
    element_id_right = interfaces.neighbor_ids[2, interface_id]

    # Interface neighboring two distinct ODE levels belong to finest one
    ode_level = min(get(element_ODE_level_dict, element_id_left, -1), 
                    get(element_ODE_level_dict, element_id_right, -1))                           
    
    @assert ode_level != -1 "Errors in datastructures for ODE level assignment"           

    # Add to accumulated container
    for l in ode_level:length(level_info_elements)
      push!(level_info_interfaces_set_acc[l], interface_id)
    end
  end
  # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
  level_info_interfaces_acc = [Vector{Int}() for _ in 1:length(level_info_elements)]
  for level in 1:length(level_info_elements)
    level_info_interfaces_acc[level] = sort(collect(level_info_interfaces_set_acc[level]))
  end
  @assert length(level_info_interfaces_acc[end]) == n_interfaces "highest level should contain all interfaces"

  level_info_boundaries_acc = [Vector{Int}() for _ in 1:length(level_info_elements)]
  level_info_mortars_acc = [Vector{Int}() for _ in 1:length(level_info_elements)]

  println("level_info_elements:")
  display(level_info_elements); println()
  println("level_info_elements_acc:")
  display(level_info_elements_acc); println()

  println("level_info_interfaces_acc:")
  display(level_info_interfaces_acc); println()

  println("level_info_boundaries_acc:")
  display(level_info_boundaries_acc); println()

  # Set initial distribution of DG Base function coefficients 
  @unpack equations, solver = ode.p
  u = wrap_array(u0, mesh, equations, solver, cache)
  level_u_indices_elements = [Vector{Int}() for _ in 1:length(level_info_elements)]
  for level in 1:length(level_info_elements)
    for element_id in level_info_elements[level]
      indices = vec(transpose(LinearIndices(u)[:, :, element_id]))
      append!(level_u_indices_elements[level], indices)
    end
  end
  display(level_u_indices_elements); println()
  =#

  #=
  # CARE: Distribute level assignment randomly
  Random.seed!(42); # Needed to fix error constant
  level_info_elements     = [Vector{Int}() for _ in 1:alg.NumDoublings+1]
  level_info_elements_acc = [Vector{Int}() for _ in 1:alg.NumDoublings+1]
  
  for element_id in 1:n_elements
    level_id = Int(mod(round(1000 * rand()), alg.NumDoublings+1)) + 1

    push!(level_info_elements[level_id], element_id)
    # Add to accumulated container
    for l in level_id:alg.NumDoublings+1
      push!(level_info_elements_acc[l], element_id)
    end  
  end
  @assert length(level_info_elements_acc[end]) == n_elements "highest level should contain all elements"

  element_ODE_level_dict = Dict{Int, Int}()
  for level in 1:length(level_info_elements)
    for element_id in level_info_elements[level]
      push!(element_ODE_level_dict, element_id=>level)
    end
  end
  display(element_ODE_level_dict); println()

  level_info_interfaces_set_acc = [Set{Int}() for _ in 1:length(level_info_elements)]
  # Determine ODE level for each interface
  for interface_id in 1:n_interfaces
    # Get element ids
    element_id_left  = interfaces.neighbor_ids[1, interface_id]
    element_id_right = interfaces.neighbor_ids[2, interface_id]

    # Interface neighboring two distinct ODE levels belong to finest one
    ode_level = min(get(element_ODE_level_dict, element_id_left, -1), 
                    get(element_ODE_level_dict, element_id_right, -1))                           
    
    @assert ode_level != -1 "Errors in datastructures for ODE level assignment"           

    # Add to accumulated container
    for l in ode_level:length(level_info_elements)
      push!(level_info_interfaces_set_acc[l], interface_id)
    end
  end
  # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
  level_info_interfaces_acc = [Vector{Int}() for _ in 1:length(level_info_elements)]
  for level in 1:length(level_info_elements)
    level_info_interfaces_acc[level] = sort(collect(level_info_interfaces_set_acc[level]))
  end
  @assert length(level_info_interfaces_acc[end]) == n_interfaces "highest level should contain all interfaces"

  level_info_boundaries_acc = [Vector{Int}() for _ in 1:length(level_info_elements)]
  level_info_mortars_acc = [Vector{Int}() for _ in 1:length(level_info_elements)]

  println("level_info_elements:")
  display(level_info_elements); println()
  println("level_info_elements_acc:")
  display(level_info_elements_acc); println()

  println("level_info_interfaces_acc:")
  display(level_info_interfaces_acc); println()

  println("level_info_boundaries_acc:")
  display(level_info_boundaries_acc); println()

  # Set initial distribution of DG Base function coefficients 
  @unpack equations, solver = ode.p
  u = wrap_array(u0, mesh, equations, solver, cache)
  level_u_indices_elements = [Vector{Int}() for _ in 1:length(level_info_elements)]
  dimensions = ndims(mesh.tree) # Spatial dimension
  if dimensions == 1
    for level in 1:alg.NumDoublings+1
      for element_id in level_info_elements[level]
        indices = vec(transpose(LinearIndices(u)[:, :, element_id]))
        append!(level_u_indices_elements[level], indices)
      end
    end
  elseif dimensions == 2
    for level in 1:alg.NumDoublings+1
      for element_id in level_info_elements[level]
        indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, element_id]))
        append!(level_u_indices_elements[level], indices)
      end
    end
  end
  display(level_u_indices_elements); println()
  =#

  ### Done with setting up for handling of level-dependent integration ###

  integrator = PRK_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                (prob=ode,), ode.f, alg,
                PERK_Multi_IntegratorOptions(callback, ode.tspan; kwargs...), false,
                level_info_elements_acc, level_info_interfaces_acc, level_info_boundaries_acc,
                level_info_mortars_acc, level_u_indices_elements)
            
  # initialize callbacks
  if callback isa CallbackSet
    for cb in callback.continuous_callbacks
      error("unsupported")
    end
    for cb in callback.discrete_callbacks
      cb.initialize(cb, integrator.u, integrator.t, integrator)
    end
  elseif !isnothing(callback)
    error("unsupported")
  end

  # Start actual solve
  solve!(integrator)
end

function solve!(integrator::PRK_Integrator)
  @unpack prob = integrator.sol
  @unpack alg = integrator
  t_end = last(prob.tspan) # Final time
  callbacks = integrator.opts.callback

  println("Start time integration.")

  integrator.finalstep = false
  @trixi_timeit timer() "main loop" while !integrator.finalstep
    if isnan(integrator.dt)
      error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end || isapprox(integrator.t + integrator.dt, t_end)
      integrator.dt = t_end - integrator.t
      terminate!(integrator)
    end

    # Hard-code consistent, but not conservative RK (Tang & Warnecke)
    #=
    alg.StagesMax = 2
    alg.CommonStages = [1]

    alg.A = zeros(2, 2, 2)
    alg.A[1, :, :] = [0 0
                      0.5 0]

    alg.A[2, :, :] = [0 0
                      0.5 0]
    =#

    # Osher & Sanders: Conservative, but not internally consistent
    #=
    alg.StagesMax = 2
    alg.CommonStages = [2]

    alg.A = zeros(2, 2, 2)
    alg.A[1, :, :] = [0 0
                      0 0]

    alg.A[2, :, :] = [0 0
                      0.5 0]
    =#

    # TODO: Use this not in the multilevel context, but only throughout the entire domain to showcase optimization

    # TODO: Multi-threaded execution as implemented for other integrators instead of vectorized operations
    @trixi_timeit timer() "PRK" begin
      # Butcher-Tableau based approach
      kfast = zeros(length(integrator.u), alg.Stages)
      kslow = zeros(length(integrator.u), alg.Stages)
      
      # k1: Computed on all levels simultaneously
      integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
      kfast[integrator.level_u_indices_elements[1], 1] = integrator.du[integrator.level_u_indices_elements[1]] * integrator.dt
      kslow[integrator.level_u_indices_elements[2], 1] = integrator.du[integrator.level_u_indices_elements[2]] * integrator.dt

      # k2: same action for all levels
      tmp = copy(integrator.u)
      tmp += alg.c[2] * kfast[:, 1] + alg.c[2] * kslow[:, 1]
      integrator.f(integrator.du, tmp, prob.p, integrator.t)
      kfast[integrator.level_u_indices_elements[1], 2] = integrator.du[integrator.level_u_indices_elements[1]] * integrator.dt
      kslow[integrator.level_u_indices_elements[2], 2] = integrator.du[integrator.level_u_indices_elements[2]] * integrator.dt

      for i = 3:alg.Stages
        tmp = copy(integrator.u)
        # Partitioned Runge-Kutta approach: One state that contains updates from all levels
        for j = 1:i-1
          tmp += alg.A[1, i-2, j] * kfast[:, j] + alg.A[2, i-2, j] * kslow[:, j]
        end

        # Evaluate f with all elements
        integrator.f(integrator.du, tmp, prob.p, integrator.t)
        kfast[integrator.level_u_indices_elements[1], i] = integrator.du[integrator.level_u_indices_elements[1]] * integrator.dt
        kslow[integrator.level_u_indices_elements[2], i] = integrator.du[integrator.level_u_indices_elements[2]] * integrator.dt

      end

      # For Tang & Warneke
      #integrator.u += kslow[:, 1] + 0.5 *(kfast[:, 1] + kfast[:, 2])

      # For Osher & Sanders
      #integrator.u += 0.5 *(kslow[:, 1] + kslow[:, 2]) + 0.5 *(kfast[:, 1] + kfast[:, 2])
      
      for i = 1:alg.Stages
        integrator.u += alg.b[i] * (kfast[:, i] + kslow[:, i])
      end
      
    end # PRK step

    integrator.iter += 1
    integrator.t += integrator.dt

    # handle callbacks
    if callbacks isa CallbackSet
      for cb in callbacks.discrete_callbacks
        if cb.condition(integrator.u, integrator.t, integrator)
          cb.affect!(integrator)
        end
      end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
      @warn "Interrupted. Larger maxiters is needed."
      terminate!(integrator)
    end
  end

  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u),
                                integrator.sol.prob)
end

# get a cache where the RHS can be stored
get_du(integrator::PRK_Integrator) = integrator.du
get_tmp_cache(integrator::PRK_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::PRK_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::PRK_Integrator, dt)
  integrator.dt = dt
end

# stop the time integration
function terminate!(integrator::PRK_Integrator)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PRK_Integrator, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.u_tmp, new_size)
end

end # @muladd
