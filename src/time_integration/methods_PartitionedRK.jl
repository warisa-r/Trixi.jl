# TODO: Currently hard-coded to second order accurate methods!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function ComputePRK_Coefficients(StagesMin::Int, NumDoublings::Int, PathPseudoExtrema::AbstractString,
                                  TrueComplexPEMax::Int)

  ForwardEulerWeights = zeros(NumDoublings+1)
  a  = zeros(TrueComplexPEMax, NumDoublings+1)
  b1 = zeros(TrueComplexPEMax, NumDoublings+1)
  b2 = zeros(TrueComplexPEMax, NumDoublings+1)

  ### Set RKM / Butcher Tableau parameters corresponding to Base-Case (Minimal number stages) ### 

  PathPureReal = PathPseudoExtrema * "PureReal" * string(StagesMin) * ".txt"
  NumPureReal, PureReal = read_file(PathPureReal, Float64)

  @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
  @assert PureReal[1] <= -1.0 "Assume that pure-real pseudo-extremum is smaller then 1.0"
  ForwardEulerWeights[1] = -1.0 / PureReal[1]

  PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(StagesMin) * ".txt"
  NumTrueComplex, TrueComplex = read_file(PathTrueComplex, ComplexF64)
  @assert NumTrueComplex == StagesMin / 2 - 1 "Assume that all but one pseudo-extremum are complex"

  # Sort ascending => ascending timesteps (real part is always negative)
  perm = sortperm(real.(TrueComplex))
  TrueComplex = TrueComplex[perm]

  # Find first element where a timestep would be greater 1 => Special treatment
  IndGreater1 = findfirst(x->real(x) > -1.0, TrueComplex)

  if IndGreater1 === nothing
    # Easy: All can use negated inverse root as timestep/entry
    IndGreater1 = NumTrueComplex + 1
  end

  for i = 1:IndGreater1 - 1
    a[i, 1]  = -1.0 / real(TrueComplex[i])
    b1[i, 1] = -real(TrueComplex[i]) / (abs(TrueComplex[i]) .* abs.(TrueComplex[i]))
    b2[i, 1] = b1[i, 1]
  end

  # To avoid timesteps > 1, compute difference between current maximum timestep = a[IndGreater1 - 1] +  and 1.
  dtGreater1 = (1.0 - a[IndGreater1 - 1, 1]) / (NumTrueComplex - IndGreater1 + 1)

  for i = IndGreater1:NumTrueComplex
    # Fill gap a[IndGreater1 - 1] to 1 equidistantly
    a[i, 1]  = a[IndGreater1 - 1, 1] + dtGreater1 * (i - IndGreater1 + 1)

    b2[i, 1] = 1.0 / (abs(TrueComplex[i]) * abs(TrueComplex[i]) * a[i, 1])
    b1[i, 1] = -2.0 * real(TrueComplex[i]) / (abs(TrueComplex[i]) * abs(TrueComplex[i])) - b2[i, 1]
  end

  # Combine distinct (!) timesteps of pure real and true complex roots
  c = vcat(ForwardEulerWeights[1], a[:, 1])

  ### Set RKM / Butcher Tableau parameters corresponding for higher stages ### 

  for i = 1:NumDoublings
    Degree = StagesMin * 2^i

    PathPureReal = PathPseudoExtrema * "PureReal" * string(Degree) * ".txt"
    NumPureReal, PureReal = read_file(PathPureReal, Float64)

    @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
    @assert PureReal[1] <= -1.0 "Assume that pure-real pseudo-extremum is smaller then 1.0"
    ForwardEulerWeights[i+1] = -1.0 / PureReal[1]

    PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(Degree) * ".txt"
    NumTrueComplex, TrueComplex = read_file(PathTrueComplex, Complex{Float64})
    @assert NumTrueComplex == Degree / 2 - 1 "Assume that all but one pseudo-extremum are complex"

    # Sort ascending => ascending timesteps (real part is always negative)
    perm = sortperm(real.(TrueComplex))
    TrueComplex = TrueComplex[perm]

    # Different (!) timesteps of higher degree RKM
    c_higher = zeros(Int(Degree / 2))
    c_higher[1] = ForwardEulerWeights[i+1]
    for j = 2:Int(Degree / 2)
      if j % 2 == 0 # Copy timestep from lower degree
        c_higher[j] = c[Int(j / 2)]
      else # Interpolate timestep
        c_higher[j] = c[Int((j-1)/2)] + 0.5 * (c[Int((j+1)/2)] - c[Int((j-1)/2)])
      end
    end
    c = copy(c_higher)

    for j = 1:length(c_higher) - 1
      a[j, i+1]  = c_higher[j+1]
      b2[j, i+1] = 1.0 / (abs(TrueComplex[j]) * abs(TrueComplex[j]) * a[j, i+1])
      b1[j, i+1] = -2.0 * real(TrueComplex[j]) / (abs(TrueComplex[j]) * abs(TrueComplex[j])) - b2[j, i+1]
    end
  end

  # Re-order columns to comply with the level structure of the mesh quantities
  perm = Vector(NumDoublings+1:-1:1)
  permute!(ForwardEulerWeights, perm)
  Base.permutecols!!(a, perm)

  perm = Vector(NumDoublings+1:-1:1) # 'permutecols!!' "deletes" perm it seems
  Base.permutecols!!(b1, perm)
  
  perm = Vector(NumDoublings+1:-1:1) # 'permutecols!!' "deletes" perm it seems
  Base.permutecols!!(b2, perm)

  println("ForwardEulerWeights:\n"); display(ForwardEulerWeights); println("\n")
  println("a:\n"); display(a); println("\n")
  println("b1:\n"); display(b1); println("\n")
  println("b2:\n"); display(b2); println("\n")
  # CARE: Here, c_i is not sum_j a_{ij} !
  println("c:\n"); display(c); println("\n")

  # Compatibility condition (https://link.springer.com/article/10.1007/BF01395956)
  println("Sum of ForwardEuleWeight and b1, b2 pairs:")
  display(transpose(ForwardEulerWeights) + sum(b1, dims=1) + sum(b2, dims=1)); println()

  return ForwardEulerWeights, a, b1, b2, c
end

function BuildButcherTableaus(ForwardEulerWeights::Vector{Float64}, a::Matrix{Float64}, 
                              b1::Matrix{Float64}, b2::Matrix{Float64}, c_::Vector{Float64}, 
                              StagesMin::Int, NumDoublings::Int)
  StagesMax = StagesMin * 2^NumDoublings

  b = zeros(StagesMax)
  b[StagesMax] = 1

  c    = zeros(StagesMax)
  c[2] = c_[1] # Smalles forward Euler (time)step
  for i in 2:length(c_)
    c[2*i - 1] = c_[i]
    c[2*i]     = c_[i]
  end
  display(c); println()

  A = zeros(NumDoublings+1, StagesMin * 2^NumDoublings, StagesMin * 2^NumDoublings)

  # Try doing forward Euler step "one earlier"
  A[2, 3, 1] = ForwardEulerWeights[2]
  for i = 1:NumDoublings+1
    # Forward Euler contribution
    A[i, 2*i:end, 1] .= ForwardEulerWeights[i]

    # Intermediate two-step submethods
    for j = 1:Int((StagesMin * 2^(NumDoublings + 1 - i) - 2)/2)
      A[i, 2^i + 2^i*j - 1, 2^i * j] = a[j, i]

      A[i, (2^i + 2^i * j):end, 2^i * j]         .= b1[j, i]
      A[i, (2^i + 2^i * j):end, 2^i + 2^i*j - 1] .= b2[j, i]
    end
    display(A[i, :, :]); println()
    println("Sum of last row of A: ", sum(A[i, StagesMax, :], dims=1))
  end

  return A
end

### Based on file "methods_2N.jl", use this as a template for P-ERK RK methods

"""
    PRK()

The following structures and methods provide a minimal implementation of
the 'ForwardEulerTwoStep' temporal integrator.

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{54, 43} methods.
"""

mutable struct PRK
  # Reference = minimum number of stages
  StagesMin::Int
  # Determines how often one doubles the number of stages 
  # Number of methods = NumDoublings +1 
  NumDoublings::Int
  # Maximum number of stages = StagesMin * 2^NumDoublings
  StagesMax::Int
  # Maximum Number of True Complex Pseudo Extrema, relevant for many datastructures
  TrueComplexPEMax::Int
  # Timestep corresponding to lowest number of stages
  dtOptMin::Real
  # TODO:Add StagesMax member variable

  ForwardEulerWeights::Vector{Float64}
  a::Matrix{Float64}
  b1::Matrix{Float64}
  b2::Matrix{Float64}
  c::Vector{Float64}

  CommonStages::Vector{Int64}
  ActiveLevels::Vector{Vector{Int64}}

  A::Array{Float64} # Butcher Tableaus

  # Constructor for previously computed A Coeffs
  function PRK(StagesMin_::Int, NumDoublings_::Int, dtOptMin_::Real, 
                PathPseudoExtrema_::AbstractString)
    newPRK = new(StagesMin_, NumDoublings_, StagesMin_ * 2^NumDoublings_, 
                  Int((StagesMin_ * 2^NumDoublings_ - 2)/2), dtOptMin_)

    newPRK.ForwardEulerWeights, newPRK.a, newPRK.b1, newPRK.b2, newPRK.c = 
      ComputePRK_Coefficients(StagesMin_, NumDoublings_, PathPseudoExtrema_, 
                               newPRK.TrueComplexPEMax)

    # Stages at that we have to take common steps                                 
    newPRK.CommonStages = [4] # Forward Euler step
    for i = 1:Int((newPRK.StagesMax / 2 - 2)/2)
      push!(newPRK.CommonStages, 3 + i * 4)
      push!(newPRK.CommonStages, 4 + i * 4)
    end
    println("Commonstages"); display(newPRK.CommonStages); println()

    # TODO: Generalize this!
    newPRK.ActiveLevels = [Vector{Int}() for _ in 1:newPRK.StagesMax]
    newPRK.ActiveLevels[1] = Vector(1:newPRK.NumDoublings+1)
    newPRK.ActiveLevels[2] = [1]
    newPRK.ActiveLevels[3] = [1]
    newPRK.ActiveLevels[4] = [1, 2]
    newPRK.ActiveLevels[5] = [1]
    newPRK.ActiveLevels[6] = [1]
    newPRK.ActiveLevels[7] = [1, 2]
    newPRK.ActiveLevels[8] = [1, 2]


    newPRK.A = BuildButcherTableaus(newPRK.ForwardEulerWeights, newPRK.a, newPRK.b1, newPRK.b2, newPRK.c, 
    StagesMin_, NumDoublings_)

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
               dt::Real, callback=nothing, kwargs...)
  u0 = copy(ode.u0) # Initial value
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
  @assert length(level_info_elements_acc[end]) == n_elements "highest level should contain all elements"


  # Use sets first to avoid double storage of interfaces
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

    # Add to accumulated container
    for l in level_id_left:n_levels
      push!(level_info_interfaces_set_acc[l], interface_id)
    end
    for l in level_id_right:n_levels
      push!(level_info_interfaces_set_acc[l], interface_id)
    end
  end

  # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
  level_info_interfaces_acc = [Vector{Int}() for _ in 1:n_levels]
  for level in 1:n_levels
    level_info_interfaces_acc[level] = sort(collect(level_info_interfaces_set_acc[level]))
  end
  @assert length(level_info_interfaces_acc[end]) == n_interfaces "highest level should contain all interfaces"


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
      level_info_elements_set[level] = setdiff(level_info_elements_set[level], level_info_elements_set[fine_level])
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
  @assert length(level_info_elements_acc[end]) == n_elements "highest level should contain all elements"

  # Use sets first to avoid double storage of interfaces
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

    # Add to accumulated container
    for l in ode_level:n_levels
      push!(level_info_interfaces_set_acc[l], interface_id)
    end
  end
  # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
  level_info_interfaces_acc = [Vector{Int}() for _ in 1:n_levels]
  for level in 1:n_levels
    level_info_interfaces_acc[level] = sort(collect(level_info_interfaces_set_acc[level]))
  end
  @assert length(level_info_interfaces_acc[end]) == n_interfaces "highest level should contain all interfaces"


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

  println("level_info_interfaces_acc:")
  display(level_info_interfaces_acc); println()

  println("level_info_boundaries_acc:")
  display(level_info_boundaries_acc); println()
  
  
  # Set initial distribution of DG Base function coefficients 
  @unpack equations, solver = ode.p
  u = wrap_array(u0, mesh, equations, solver, cache)

  level_u_indices_elements = [Vector{Int}() for _ in 1:n_levels]
  for level in 1:n_levels
    for element_id in level_info_elements[level]
      indices = vec(transpose(LinearIndices(u)[:, :, element_id]))
      append!(level_u_indices_elements[level], indices)
    end
  end
  display(level_u_indices_elements); println()


  integrator = PRK_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, 
                 ode.p, # the semidiscretization
                 (prob=ode,), # Not really sure whats going on here
                 ode.f, # the right-hand-side of the ODE u' = f(u, p, t)
                 alg, # The ODE integration algorithm/method
                 PRK_IntegratorOptions(callback, ode.tspan; kwargs...), false,
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

  # TODO: Not really elegant way, maybe try to bundle 'indices' datastructures into 'integrator' 
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
    
    alg.StagesMax = 2
    alg.CommonStages = [1]

    alg.A = zeros(2, 2, 2)
    alg.A[1, :, :] = [0 0
                      0.5 0]

    alg.A[2, :, :] = [0 0
                      0.5 0]
    

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
    @trixi_timeit timer() "Forward Euler Two Stage ODE integration step" begin
      # Butcher-Tableau based approach
      kfast = zeros(length(integrator.u), alg.StagesMax)
      kslow = zeros(length(integrator.u), alg.StagesMax)
      
      # k1: Computed on all levels simultaneously
      integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
      kfast[integrator.level_u_indices_elements[1], 1] = integrator.du[integrator.level_u_indices_elements[1]] * integrator.dt
      kslow[integrator.level_u_indices_elements[2], 1] = integrator.du[integrator.level_u_indices_elements[2]] * integrator.dt

      for i = 2:alg.StagesMax
        tmp = integrator.u

        # Partitioned Runge-Kutta approach: One state that contains updates from all levels
        for j = 1:i-1
          tmp += alg.A[1, i, j] * kfast[:, j] + alg.A[2, i, j] * kslow[:, j]
        end

        if i in alg.CommonStages
          # Evaluate f with all elements
          integrator.f(integrator.du, tmp, prob.p, integrator.t)
          kfast[integrator.level_u_indices_elements[1], i] = integrator.du[integrator.level_u_indices_elements[1]] * integrator.dt
          kslow[integrator.level_u_indices_elements[2], i] = integrator.du[integrator.level_u_indices_elements[2]] * integrator.dt
        else
          # Evaluate only fine level
          integrator.f(integrator.du, tmp, prob.p, integrator.t,
                       integrator.level_info_elements_acc[1],
                       integrator.level_info_interfaces_acc[1],
                       integrator.level_info_boundaries_acc[1],
                       integrator.level_info_mortars_acc[1])

          kfast[integrator.level_u_indices_elements[1], i] = integrator.du[integrator.level_u_indices_elements[1]] * integrator.dt
        end
      end

      #integrator.u += kfast[:, alg.StagesMax] + kslow[:, alg.StagesMax]

      # For Tang & Warneke
      integrator.u += kslow[:, 1] + 0.5 *(kfast[:, 1] + kfast[:, 2])

      # For Osher & Sanders
      #integrator.u += 0.5 *(kslow[:, 1] + kslow[:, 2]) + 0.5 *(kfast[:, 1] + kfast[:, 2])
      
      
      #=
      # First Forward Euler step
      t_stage = integrator.t
      integrator.f(integrator.du, integrator.u, prob.p, t_stage) # du = k1
      integrator.u_tmp = integrator.u + alg.ForwardEulerWeight * integrator.dt * integrator.du
      t_stage += alg.ForwardEulerWeight * integrator.dt

      # Intermediate "two-step" sub methods
      for i in eachindex(alg.a)
        # Low-storage implementation (only one k = du):
        integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stage) # du = k1

        integrator.u_tmp += integrator.dt * alg.b1[i] * integrator.du

        t_stage += alg.a[i] * integrator.dt
        integrator.f(integrator.du, integrator.u_tmp + integrator.dt *(alg.a[i] - alg.b1[i]) * integrator.du, 
                     prob.p, t_stage) # du = k2
        integrator.u_tmp .+= integrator.dt .* alg.b2[i] .* integrator.du
      end
      # Final Euler step with step length of dt (Due to form of stability polynomial)
      integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stage) # k1
      ntegrator.u += integrator.dt * integrator.du
      =#
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
