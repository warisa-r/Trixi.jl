# TODO: Currently hard-coded to second order accurate methods!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

using Random # NOTE: Only for tests

function ComputeACoeffs(NumStageEvals::Int,
                        SE_Factors::Vector{Float64}, MonCoeffs::Vector{Float64})
  ACoeffs = MonCoeffs

  for stage in 1:NumStageEvals - 2
    ACoeffs[stage] /= SE_Factors[stage]
    for prev_stage in 1:stage-1
      ACoeffs[stage] /= ACoeffs[prev_stage]
    end
  end

  return reverse(ACoeffs)
end

function ComputePERK_Multi_ButcherTableau(NumDoublings::Int, NumStages::Int, BasePathMonCoeffs::AbstractString)

  # c Vector form Butcher Tableau (defines timestep per stage)
  c = zeros(Float64, NumStages)
  for k in 2:NumStages
    c[k] = (k - 1)/(2.0*(NumStages - 1))
  end
  println("Timestep-split: "); display(c); println("\n")
  # TODO: Not sure if valid for general ConsOrder (not 2)!
  SE_Factors = reverse(c[2:end-1])

  # - 2 Since First entry of A is always zero (explicit method) and second is given by c (PERK_Multi specific)
  CoeffsMax = NumStages - 2

  AMatrices = zeros(NumDoublings+1, CoeffsMax, 2)
  for i = 1:NumDoublings+1
    AMatrices[i, :, 1] = c[3:end]
  end

  ActiveLevels = [Vector{Int}() for _ in 1:NumStages]
  # k1 is evaluated at all levels
  ActiveLevels[1] = 1:NumDoublings+1

  for level = 1:NumDoublings + 1
    
    
    PathMonCoeffs = BasePathMonCoeffs * "gamma_" * string(Int(NumStages / 2^(level - 1))) * ".txt"
    NumMonCoeffs, MonCoeffs = read_file(PathMonCoeffs, Float64)
    @assert NumMonCoeffs == NumStages / 2^(level - 1) - 2
    A = ComputeACoeffs(Int(NumStages / 2^(level - 1)), SE_Factors, MonCoeffs)
   

    #=
    # TODO: Not sure if I not rather want to read-in values (especially those from Many Stage C++ Optim)
    PathMonCoeffs = BasePathMonCoeffs * "a_" * string(Int(NumStages / 2^(level - 1))) * ".txt"
    NumMonCoeffs, A = read_file(PathMonCoeffs, Float64)
    @assert NumMonCoeffs == NumStages / 2^(level - 1) - 2
    =#

    AMatrices[level, CoeffsMax - Int(NumStages / 2^(level - 1) - 3):end, 1] -= A
    AMatrices[level, CoeffsMax - Int(NumStages / 2^(level - 1) - 3):end, 2]  = A

    # CARE: For linear PERK family: 4,5,6, and not 4, 8, 16, ...
    #=
    PathMonCoeffs = BasePathMonCoeffs * "a_" * string(Int(NumStages - level + 1)) * ".txt"
    NumMonCoeffs, A = read_file(PathMonCoeffs, Float64)
    AMatrices[level, CoeffsMax - Int(NumStages - level + 1 - 3):end, 1] -= A
    AMatrices[level, CoeffsMax - Int(NumStages - level + 1 - 3):end, 2]  = A
    =#

    # Add refinement levels to stages
    for stage = NumStages:-1:NumStages-NumMonCoeffs
      push!(ActiveLevels[stage], level)
    end
  end

  for i = 1:NumDoublings+1
    println("A-Matrix of Butcher tableau of level " * string(i))
    display(AMatrices[i, :, :]); println()
  end

  println("Check violation of internal consistency")
  for i = 1:NumDoublings+1
    for j = 1:i
      display(norm(AMatrices[i, :, 1] + AMatrices[i, :, 2] - AMatrices[j, :, 1] - AMatrices[j, :, 2], 1))
    end
  end

  println("\nActive Levels:"); display(ActiveLevels); println()

  return AMatrices, c, ActiveLevels
end

function ComputePERK_Multi_ButcherTableau(NumDoublings::Int, NumStages::Int, BasePathMonCoeffs::AbstractString, 
                                          bS::Float64, cEnd::Float64)
                                     
  # c Vector form Butcher Tableau (defines timestep per stage)
  c = zeros(Float64, NumStages)
  for k in 2:NumStages
    c[k] = cEnd * (k - 1)/(NumStages - 1)
  end
  println("Timestep-split: "); display(c); println("\n")

  SE_Factors = bS * reverse(c[2:end-1])

  # - 2 Since First entry of A is always zero (explicit method) and second is given by c (PERK_Multi specific)
  CoeffsMax = NumStages - 2

  AMatrices = zeros(NumDoublings+1, CoeffsMax, 2)
  for i = 1:NumDoublings+1
    AMatrices[i, :, 1] = c[3:end]
  end

  ActiveLevels = [Vector{Int}() for _ in 1:NumStages]
  # k1 is evaluated at all levels
  ActiveLevels[1] = 1:NumDoublings+1

  for level = 1:NumDoublings + 1
    
    PathMonCoeffs = BasePathMonCoeffs * "gamma_" * string(Int(NumStages / 2^(level - 1))) * ".txt"
    NumMonCoeffs, MonCoeffs = read_file(PathMonCoeffs, Float64)
    @assert NumMonCoeffs == NumStages / 2^(level - 1) - 2
    A = ComputeACoeffs(Int(NumStages / 2^(level - 1)), SE_Factors, MonCoeffs)

    AMatrices[level, CoeffsMax - Int(NumStages / 2^(level - 1) - 3):end, 1] -= A
    AMatrices[level, CoeffsMax - Int(NumStages / 2^(level - 1) - 3):end, 2]  = A

    # CARE: For linear PERK family: 4,5,6, and not 4, 8, 16, ...
    #=
    PathMonCoeffs = BasePathMonCoeffs * "a_" * string(Int(NumStages - level + 1)) * ".txt"
    NumMonCoeffs, A = read_file(PathMonCoeffs, Float64)
    AMatrices[level, CoeffsMax - Int(NumStages - level + 1 - 3):end, 1] -= A
    AMatrices[level, CoeffsMax - Int(NumStages - level + 1 - 3):end, 2]  = A
    =#

    # Add refinement levels to stages
    for stage = NumStages:-1:NumStages-NumMonCoeffs
      push!(ActiveLevels[stage], level)
    end
  end

  for i = 1:NumDoublings+1
    println("A-Matrix of Butcher tableau of level " * string(i))
    display(AMatrices[i, :, :]); println()
  end

  println("Check violation of internal consistency")
  for i = 1:NumDoublings+1
    for j = 1:i
      display(norm(AMatrices[i, :, 1] + AMatrices[i, :, 2] - AMatrices[j, :, 1] - AMatrices[j, :, 2], 1))
    end
  end

  println("\nActive Levels:"); display(ActiveLevels); println()

  return AMatrices, c, ActiveLevels
end


"""
    PERK_Multi()

The following structures and methods provide an implementation of
the paired explicit Runge-Kutta method (https://doi.org/10.1016/j.jcp.2019.05.014)
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).
In particular, these methods are tailored to a coupling with AMR.

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{54, 43} methods.
"""

mutable struct PERK_Multi
  const NumStageEvalsMin::Int64
  const NumDoublings::Int64
  const NumStages::Int64
  stage_callbacks

  AMatrices::Array{Float64, 3}
  c::Vector{Float64}
  ActiveLevels::Vector{Vector{Int64}}

  # Constructor for previously computed A Coeffs
  function PERK_Multi(NumStageEvalsMin_::Int, NumDoublings_::Int,
                      BasePathMonCoeffs_::AbstractString, bS::Float64, cEnd::Float64;
                      stage_callbacks=nothing)

    newPERK_Multi = new(NumStageEvalsMin_, NumDoublings_,
                        # Current convention: NumStages = MaxStages = S;
                        # TODO: Allow for different S >= Max {Stage Evals}
                        NumStageEvalsMin_ * 2^NumDoublings_,
                        stage_callbacks)
                        # CARE: Hack to eanble linear increasing PERK
                        #NumStageEvalsMin_ + NumDoublings_)

    newPERK_Multi.AMatrices, newPERK_Multi.c, newPERK_Multi.ActiveLevels = 
      ComputePERK_Multi_ButcherTableau(NumDoublings_, newPERK_Multi.NumStages, BasePathMonCoeffs_, bS, cEnd)

    return newPERK_Multi
  end
end # struct PERK_Multi


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct PERK_Multi_IntegratorOptions{Callback}
  callback::Callback # callbacks; used in Trixi
  adaptive::Bool # whether the algorithm is adaptive; ignored
  dtmax::Float64 # ignored
  maxiters::Int # maximal numer of time steps
  tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function PERK_Multi_IntegratorOptions(callback, tspan; maxiters=typemax(Int), kwargs...)
  PERK_Multi_IntegratorOptions{typeof(callback)}(callback, false, Inf, maxiters, [last(tspan)])
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK_Multi_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, PERK_Multi_IntegratorOptions}
  u::uType
  du::uType
  u_tmp::uType
  t::RealT
  dt::RealT # current time step
  dtcache::RealT # ignored
  iter::Int # current number of time steps (iteration)
  p::Params # will be the semidiscretization from Trixi
  sol::Sol # faked
  f::F
  alg::Alg # This is our own class written above; Abbreviation for ALGorithm
  opts::PERK_Multi_IntegratorOptions
  finalstep::Bool # added for convenience
  # PERK_Multi stages:
  k1::uType
  k_higher::uType
  # Variables managing level-depending integration
  level_info_elements_acc::Vector{Vector{Int64}}
  level_info_interfaces_acc::Vector{Vector{Int64}}
  level_info_boundaries_acc::Vector{Vector{Int64}}
  level_info_mortars_acc::Vector{Vector{Int64}}
  level_u_indices_elements::Vector{Vector{Int64}}
  t_stage::RealT
  coarsest_lvl::Int64
end

# Forward integrator.destats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PERK_Multi_Integrator, field::Symbol)
  if field === :destats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PERK_Multi;
               dt, callback=nothing, kwargs...)

  u0 = copy(ode.u0)
  du = similar(u0)
  u_tmp = similar(u0)

  # PERK_Multi stages
  k1       = similar(u0)
  k_higher = similar(u0)

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
  #=
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
  =#
  
  # NOTE: Next-to-fine is also integrated with fine integrator
  
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

  integrator = PERK_Multi_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                (prob=ode,), ode.f, alg,
                PERK_Multi_IntegratorOptions(callback, ode.tspan; kwargs...), false,
                k1, k_higher,   
                level_info_elements_acc, level_info_interfaces_acc, level_info_boundaries_acc,
                level_info_mortars_acc, level_u_indices_elements, t0, -1)
            
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


function minmod(a::Vector{Float64})
  if all(>=(0), a)
    return minimum(abs.(a))
  elseif all(<(0), a)
    -minimum(abs.(a))
  else
    return 0
  end
end

function Limiterp1!(u, integrator::PERK_Multi_Integrator)
  @unpack solver = integrator.p
  Weights = solver.basis.weights
  Nodes   = solver.basis.nodes
  NumNodes = length(Nodes)

  NumElements = Int(length(u)/NumNodes)

  # TODO: Get troubled elements from indicator!
  #TroubledElements = [15, 16]
  TroubledElements = 1:NumElements
  #TroubledElements = integrator.level_info_elements_acc[1]

  u_avg = zeros(NumElements, 1)
  for i in TroubledElements
    for j = 1:NumNodes
      u_avg[i] += Weights[j] * u[NumNodes * (i-1) + j]
    end
    u_avg[i] *= 0.5 # For 1D on [-1, 1] reference element
  end

  # Deltas
  u_Delta = zeros(NumElements, 2)
  for i in TroubledElements
    # Left
    # Assume periodicity
    if i == 1
      u_Delta[i, 1] = u_avg[i, 1] - u_avg[NumElements, 1]
    else
      u_Delta[i, 1] = u_avg[i, 1] - u_avg[i-1, 1]
    end

    # Right
    # Assume periodicity
    if i == NumElements
      u_Delta[i, 2] = u_avg[1, 1] - u_avg[NumElements, 1]
    else
      u_Delta[i, 2] = u_avg[i+1, 1] - u_avg[i, 1]
    end
  end

  sigma = zeros(NumElements)
  for i in TroubledElements
    sigma[i] = minmod(u_Delta[i,:])*0.5
  end

  
  for i in TroubledElements
    u[NumNodes * (i-1) + 1] = u_avg[i] - sigma[i] / sqrt(3)
    u[NumNodes * i]         = u_avg[i] + sigma[i] / sqrt(3)
  end
end

function Limiterp2!(u, integrator::PERK_Integrator)
  @unpack solver = integrator.p
  Weights = solver.basis.weights
  Nodes   = solver.basis.nodes
  NumNodes = length(Nodes)

  NumElements = Int(length(u)/NumNodes)

  #TroubledElements = [14, 15]
  TroubledElements = 1:NumElements
  #TroubledElements = integrator.level_info_elements_acc[1]

  u_avg = zeros(NumElements, 1)
  for i in TroubledElements
    for j = 1:NumNodes
      u_avg[i] += Weights[j] * u[NumNodes * (i-1) + j]
    end
    u_avg[i] *= 0.5 # For 1D on [-1, 1] reference element
  end

  # Deltas
  u_Delta = zeros(NumElements, 2)
  for i in TroubledElements
    # Left
    # Assume periodicity
    if i == 1
      u_Delta[i, 1] = u_avg[i, 1] - u_avg[NumElements, 1]
    else
      u_Delta[i, 1] = u_avg[i, 1] - u_avg[i-1, 1]
    end

    # Right
    # Assume periodicity
    if i == NumElements
      u_Delta[i, 2] = u_avg[1, 1] - u_avg[NumElements, 1]
    else
      u_Delta[i, 2] = u_avg[i+1, 1] - u_avg[i, 1]
    end
  end

  # TODO: Seems wrong!
  trace_values = zeros(NumElements, 2)
  for i in TroubledElements
    # Left
    trace_values[i, 1] = u[NumNodes * (i-1) + 1]
    # Right
    trace_values[i, 2] = u[NumNodes * i]
  end

  # Bar-quantities
  u_bar = zeros(NumElements, 2)

  ALegendre = [1 Nodes[1] 0.5 * (3*Nodes[1]^2 - 1)
               1 Nodes[2] 0.5 * (3*Nodes[2]^2 - 1)
               1 Nodes[3] 0.5 * (3*Nodes[3]^2 - 1)]
  
  for i in TroubledElements
    ULegendre = ALegendre\u[NumNodes * (i-1) + 1:NumNodes * i]

    # Left
    u_bar[i, 1] = ULegendre[2] * (-1) + ULegendre[3] * 0.5 *(3*1 - 1)

    # Right
    u_bar[i, 2] = ULegendre[2] *   1 + ULegendre[3] * 0.5 *(3*1 - 1)
  end 

  # Tilde-quantities
  u_tilde = zeros(NumElements, 2)
  for i in TroubledElements
    # Left
    u_tilde[i, 1] = minmod([u_bar[i, 1], u_Delta[i, 1], u_Delta[i, 2]])
    # Right
    u_tilde[i, 2] = minmod([u_bar[i, 2], u_Delta[i, 1], u_Delta[i, 2]])
  end

  for i in TroubledElements
    u[NumNodes * (i-1) + 1] = -u_tilde[i, 1] + u_avg[i]
    u[NumNodes * i]         =  u_tilde[i, 2] + u_avg[i]
  end
  
  # For p = 2 also required
  for i in TroubledElements
    # Conservation of Mass
    u[NumNodes * (i-1) + 2] = (2*u_avg[i] - Weights[1] * u[NumNodes * (i-1) + 1] 
                                          - Weights[3] * u[NumNodes * i]) / Weights[2]
  end
end


function solve!(integrator::PERK_Multi_Integrator)
  @unpack prob = integrator.sol
  @unpack alg = integrator
  t_end = last(prob.tspan)
  callbacks = integrator.opts.callback

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

    # TODO: Eliminate allocations!
    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
      
      # k1: Evaluated on entire domain / all levels
      integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
      @threaded for i in eachindex(integrator.du)
        integrator.k1[i] = integrator.du[i] * integrator.dt
      end

      #=
      # One scheme for whole domain (primarily for tests)
      tstage = integrator.t + alg.c[2] * integrator.dt
      # k2: Usually required for finest level [1]
      # (Although possible that no scheme has full stage evaluations)
      integrator.f(integrator.du, integrator.u + alg.c[2] * integrator.k1, prob.p, tstage)
      integrator.k_higher = integrator.du * integrator.dt
      
      for stage = 3:alg.NumStages
        # Construct current state

        # Use highest level
        
        integrator.u_tmp = integrator.u + alg.AMatrices[1, stage - 2, 1] * integrator.k1 + 
          alg.AMatrices[1, stage - 2, 2] * integrator.k_higher
        

        # Use lowest level
        #=
        integrator.u_tmp = integrator.u + (alg.c[stage] - alg.ACoeffs[stage - 2, end]) * integrator.k1 + 
            alg.ACoeffs[stage - 2, end] * integrator.k_higher
        =#

        tstage = integrator.t + alg.c[stage] * integrator.dt

        # Joint RHS evaluation with all elements sharing this timestep
        integrator.f(integrator.du, integrator.u_tmp, prob.p, tstage)

        integrator.k_higher = integrator.du * integrator.dt
      end
      =#
      
      integrator.t_stage = integrator.t + alg.c[2] * integrator.dt
      # k2: Here always evaluated for finest scheme (Allow currently only max. stage evaluations)
      @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
      end

      #integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)  
      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, 
                   integrator.level_info_elements_acc[1],
                   integrator.level_info_interfaces_acc[1],
                   integrator.level_info_boundaries_acc[1],
                   integrator.level_info_mortars_acc[1])
      
      @threaded for u_ind in integrator.level_u_indices_elements[1] # Update finest level
        integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
      end

      for stage = 3:alg.NumStages
        # Construct current state
        @threaded for i in eachindex(integrator.u)
          integrator.u_tmp[i] = integrator.u[i]
        end

        for level in eachindex(integrator.level_u_indices_elements) # Ensures only relevant levels are evaluated
          @threaded for u_ind in integrator.level_u_indices_elements[level]
            #integrator.u_tmp[u_ind] += alg.AMatrices[level, stage - 2, 1] * integrator.k1[u_ind]

            # Approach where one uses only the highest levels when needed
            integrator.u_tmp[u_ind] += alg.AMatrices[level + alg.NumDoublings + 1 - length(integrator.level_u_indices_elements), stage - 2, 1] * integrator.k1[u_ind]
          end

          # First attempt to be more effective
          if alg.AMatrices[level, stage - 2, 2] > 0 # TODO: Avoid if at some point (two for loops for stage < > E)
            @threaded for u_ind in integrator.level_u_indices_elements[level]
              #integrator.u_tmp[u_ind] += alg.AMatrices[level, stage - 2, 2] * integrator.k_higher[u_ind]

              # Approach where one uses only the highest levels when needed
              integrator.u_tmp[u_ind] += alg.AMatrices[level + alg.NumDoublings + 1 - length(integrator.level_u_indices_elements), stage - 2, 2] * integrator.k_higher[u_ind]
            end
          end
        end

        #Limiterp2!(integrator.u_tmp, integrator)

        integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt

        # "ActiveLevels" cannot be static for AMR, has to be checked with available levels
        #=
        integrator.coarsest_lvl = maximum(alg.ActiveLevels[stage][alg.ActiveLevels[stage] .<= 
                                                                  length(integrator.level_info_elements_acc)])
        =#

        # Allocation-free version
        for lvl in alg.ActiveLevels[stage]
          if alg.ActiveLevels[stage][lvl] > length(integrator.level_info_elements_acc)
            break
          else
            integrator.coarsest_lvl = lvl
          end
        end


        # Joint RHS evaluation with all elements sharing this timestep
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, 
                    integrator.level_info_elements_acc[integrator.coarsest_lvl],
                    integrator.level_info_interfaces_acc[integrator.coarsest_lvl],
                    integrator.level_info_boundaries_acc[integrator.coarsest_lvl],
                    integrator.level_info_mortars_acc[integrator.coarsest_lvl])
        
        # Update k_higher of relevant levels
        for level in 1:integrator.coarsest_lvl
          @threaded for u_ind in integrator.level_u_indices_elements[level]
            integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
          end
        end
      end
      
      # u_{n+1} = u_n + b_S * k_S = u_n + 1 * k_S
      @threaded for i in eachindex(integrator.u)
        # TODO: Adapt to b1 != 0, bS != 1 !
        integrator.u[i] += integrator.k_higher[i]
        #integrator.u[i] += 0.5 * (integrator.k1[i] + integrator.k_higher[i])
      end

      #=
      for stage_callback in alg.stage_callbacks
        stage_callback(integrator.u, integrator, prob.p, integrator.t_stage)
      end
      =#
    end # PERK_Multi step

    #Limiterp1!(integrator.u, integrator)

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

    for stage_callback in alg.stage_callbacks
      stage_callback(integrator.u, integrator, prob.p, integrator.t_stage)
    end

    #=
    TV = 0
    for i in 1:length(integrator.u)-1
      TV += abs(integrator.u[i+1] - integrator.u[i])
    end
    # Periodic domain
    TV += abs(integrator.u[1] - integrator.u[length(integrator.u)])
    # Normalize by number of gridpoints
    TV /= length(integrator.u)

    io = open("TV.txt", "a") do io
      println(io, TV)
    end
    =#

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
      @warn "Interrupted. Larger maxiters is needed."
      terminate!(integrator)
    end
  end # "main loop" timer
  
  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u),
                                integrator.sol.prob)
end

# get a cache where the RHS can be stored
get_du(integrator::PERK_Multi_Integrator) = integrator.du
get_tmp_cache(integrator::PERK_Multi_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::PERK_Multi_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::PERK_Multi_Integrator, dt)
  integrator.dt = dt
end

# stop the time integration
function terminate!(integrator::PERK_Multi_Integrator)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PERK_Multi_Integrator, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.u_tmp, new_size)

  resize!(integrator.k1, new_size)
  resize!(integrator.k_higher, new_size)
end


function ComputePERKSysMat(ode::ODEProblem, alg::PERK_Multi, A_ODE::Matrix;
                           dt, callback=nothing, kwargs...)

  u0 = copy(ode.u0)
  du = similar(u0)
  u_tmp = similar(u0)

  # PERK_Multi stages
  k1       = similar(u0)
  k_higher = similar(u0)

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
  
  # NOTE: Next-to-fine is also integrated with fine integrator
  
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


  ### Done with setting up for handling of level-dependent integration ###

  integrator = PERK_Multi_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                (prob=ode,), ode.f, alg,
                PERK_Multi_IntegratorOptions(callback, ode.tspan; kwargs...), false,
                k1, k_higher,   
                level_info_elements_acc, level_info_interfaces_acc, level_info_boundaries_acc,
                level_info_mortars_acc, level_u_indices_elements, t0, -1)
            
  N_Levels = length(level_u_indices_elements)
  N_ODE    = size(A_ODE, 1)

  I_Levels = zeros(N_Levels, N_ODE, N_ODE)
  for i = 1:N_Levels
    for j in eachindex(level_u_indices_elements[i])
      I_Levels[i, level_u_indices_elements[i][j], level_u_indices_elements[i][j]] = 1.0
    end
  end
  
  N_Control = 0
  for i = 1:N_Levels
    N_Control += LinearAlgebra.tr(I_Levels[i, :, :])
  end
  @assert N_Control == N_ODE

  # k1: Evaluated on entire domain / all levels
  K1 = dt * A_ODE
  
  # k2: Here always evaluated for finest scheme (Allow currently only max. stage evaluations)
  K_Higher = dt * I_Levels[1, :, :] * A_ODE * (I + alg.c[2] * K1)

  for stage = 3:alg.NumStages
    K_temp = zeros(N_ODE, N_ODE)
    for level in eachindex(integrator.level_u_indices_elements) # Ensures only relevant levels are evaluated
      K_temp += I_Levels[level, :, :] * (alg.AMatrices[level + alg.NumDoublings + 1 - length(integrator.level_u_indices_elements), stage - 2, 1] * K1 +
                                         alg.AMatrices[level + alg.NumDoublings + 1 - length(integrator.level_u_indices_elements), stage - 2, 2] * K_Higher)
    end

    # Allocation-free version
    for lvl in alg.ActiveLevels[stage]
      if alg.ActiveLevels[stage][lvl] > length(integrator.level_info_elements_acc)
        break
      else
        integrator.coarsest_lvl = lvl
      end
    end

    I_Coarsest = zeros(N_ODE, N_ODE)
    # Update k_higher of relevant levels
    for level in 1:integrator.coarsest_lvl
      I_Coarsest += I_Levels[level, :, :]
    end
    K_Higher = dt * I_Coarsest * A_ODE * (I + K_temp)
  end
  
  return I + K_Higher
end

end # @muladd