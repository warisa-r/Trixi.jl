# TODO: Currently hard-coded to second order accurate methods!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

  function ReadInFile(FilePath::AbstractString, DataType::Type)
    @assert isfile(FilePath) "Couldn't find file"
    Data = zeros(DataType, 0)
    open(FilePath, "r") do File
      while !eof(File)     
        LineContent = readline(File)     
        append!(Data, parse(DataType, LineContent))
      end
    end
    NumLines = length(Data)
  
    return NumLines, Data
  end

  function ComputeACoeffs(NumStageEvals::Int, ConsOrder::Int,
                          SE_Factors::Vector{Float64}, MonCoeffs::Vector{Float64})
    ACoeffs = MonCoeffs

    for stage in 1:NumStageEvals - ConsOrder
      ACoeffs[stage] /= SE_Factors[stage]
      for prev_stage in 1:stage-1
        ACoeffs[stage] /= ACoeffs[prev_stage]
      end
    end

    return reverse(ACoeffs)
  end
  
  function ComputePERK_ButcherTableau(NumDoublings::Int, NumStages::Int, 
                                      ConsOrder::Int, BasePathMonCoeffs::AbstractString)
  
    # c Vector form Butcher Tableau (defines timestep per stage)
    c = zeros(Float64, NumStages)
    for k in 2:NumStages
      c[k] = (k - 1)/(2.0*(NumStages - 1))
    end
    println("Timestep-split: "); display(c); println("\n")
    # TODO: Not sure if valid for general ConsOrder (not 2)!
    SE_Factors = reverse(c[2:end-1])
  
    # - 2 Since First entry of A is always zero (explicit method) and second is given by c (PERK specific)
    CoeffsMax = NumStages - 2
  
    ACoeffs = zeros(CoeffsMax, NumDoublings+1)
  
    ActiveLevels = [Vector{Int}() for _ in 1:NumStages]
    # k1 is evaluated at all levels
    ActiveLevels[1] = 1:NumDoublings+1

    for level = 1:NumDoublings + 1
      
      PathMonCoeffs = BasePathMonCoeffs * "gamma_" * string(Int(NumStages / 2^(level - 1))) * ".txt"
      NumMonCoeffs, MonCoeffs = ReadInFile(PathMonCoeffs, Float64)
      @assert NumMonCoeffs == NumStages / 2^(level - 1) - 2
      A = ComputeACoeffs(Int(NumStages / 2^(level - 1)), ConsOrder, SE_Factors, MonCoeffs)
      

      #=
      # TODO: Not sure if I not rather want to read-in values (especcially those from Many Stage C++ Optim)
      PathMonCoeffs = BasePathMonCoeffs * "a_" * string(Int(NumStages / 2^(level - 1))) * ".txt"
      NumMonCoeffs, A = ReadInFile(PathMonCoeffs, Float64)
      @assert NumMonCoeffs == NumStages / 2^(level - 1) - 2
      =#

      ACoeffs[CoeffsMax - Int(NumStages / 2^(level - 1) - 3):end, level] = A
      # Add refinement levels to stages
      for stage = NumStages:-1:NumStages-NumMonCoeffs
        push!(ActiveLevels[stage], level)
      end
    end
    display(ACoeffs); println()
    display(ActiveLevels); println()

    return ACoeffs, c, ActiveLevels
  end


  """
      PERK()
  
  The following structures and methods provide a minimal implementation of
  the paired explicit Runge-Kutta method (https://doi.org/10.1016/j.jcp.2019.05.014)
  optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).
  
  This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
  CarpenterKennedy2N{54, 43} methods.
  """
  
  mutable struct PERK
    const NumStageEvalsMin::Int
    const NumDoublings::Int
    const NumStages::Int
    const dtOptMin::Real
  
    ACoeffs::Matrix{Real}
    c::Vector{Real}
    ActiveLevels::Vector{Vector{Int64}}

  
    # Constructor for previously computed A Coeffs
    function PERK(NumStageEvalsMin_::Int, NumDoublings_::Int, ConsOrder_::Int,
                  dtOptMin_::Real, BasePathMonCoeffs_::AbstractString)

      newPERK = new(NumStageEvalsMin_, NumDoublings_,
                    # Current convention: NumStages = MaxStages = S;
                    # TODO: Allow for different S >= Max {Stage Evals}
                    NumStageEvalsMin_ * 2^NumDoublings_, dtOptMin_)
  
      newPERK.ACoeffs, newPERK.c, newPERK.ActiveLevels = 
        ComputePERK_ButcherTableau(NumDoublings_, newPERK.NumStages, ConsOrder_, BasePathMonCoeffs_)

      return newPERK
    end
  end # struct PERK
  
  
  # This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
  mutable struct PERK_IntegratorOptions{Callback}
    callback::Callback # callbacks; used in Trixi
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal numer of time steps
    tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
  end
  
  function PERK_IntegratorOptions(callback, tspan; maxiters=typemax(Int), kwargs...)
    PERK_IntegratorOptions{typeof(callback)}(callback, false, Inf, maxiters, [last(tspan)])
  end
  
  # This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
  # This implements the interface components described at
  # https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
  # which are used in Trixi.
  mutable struct PERK_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, PERK_IntegratorOptions}
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
    opts::PERK_IntegratorOptions
    finalstep::Bool # added for convenience
    # PERK stages:
    k1::uType
    k_higher::uType
    # Variables managing level-depending integration
    level_info_elements_acc::Vector{Vector{Int64}}
    level_info_interfaces_acc::Vector{Vector{Int64}}
    level_info_boundaries_acc::Vector{Vector{Int64}}
    level_info_mortars_acc::Vector{Vector{Int64}}
    level_u_indices_elements::Vector{Vector{Int64}}
  end
  
  # Forward integrator.destats.naccept to integrator.iter (see GitHub PR#771)
  function Base.getproperty(integrator::PERK_Integrator, field::Symbol)
    if field === :destats
      return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
  end
  
  # Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
  function solve(ode::ODEProblem, alg::PERK;
                 dt, callback=nothing, kwargs...)
    u0 = copy(ode.u0)
    du = similar(u0)
    u_tmp = similar(u0)

    # PERK stages
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
    level_info_boundaries_set = [Set{Int}() for _ in 1:n_levels]
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
    level_info_elements = [Vector(1:Int(n_elements/4)), Vector(Int(n_elements/4)+1:n_elements)]
    level_info_elements_acc = [Vector(1:Int(n_elements/2)), Vector(1:n_elements)]

    level_info_interfaces_acc = [Vector(1:Int(n_elements/4)), Vector(1:n_elements)]
    append!(level_info_interfaces_acc[1], n_elements)

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
    # CARE: Hard-coded "artificial" mesh splitting to check 
    # whether integrating slow elements on boundaries to fast region helps 
    @assert n_elements % 2 == 0
    level_info_elements = [Vector(4:n_elements), [2, 3]]
    append!(level_info_elements[1], 1)
    level_info_elements_acc = [level_info_elements[1], Vector(1:n_elements)]

    level_info_interfaces_acc = [Vector(3:n_elements), Vector(1:n_elements)]
    append!(level_info_interfaces_acc[1], 1)

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

    ### Done with setting up for handling of level-dependent integration ###

    integrator = PERK_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                  (prob=ode,), ode.f, alg,
                  PERK_IntegratorOptions(callback, ode.tspan; kwargs...), false,
                  k1, k_higher,                
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
  
    solve!(integrator)
  end
  
  function solve!(integrator::PERK_Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback
  
    integrator.finalstep = false

    SSPStep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
      if isnan(integrator.dt)
        error("time step size `dt` is NaN")
      end
  
      # if the next iteration would push the simulation beyond the end time, set dt accordingly
      if integrator.t + integrator.dt > t_end || isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
      end
  
      # TODO: Try multi-threaded execution as implemented for other integrators!
      @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin

        # One scheme for all
        #=
        # k1: Evaluated on entire domain / all levels
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        integrator.k1 = integrator.du * integrator.dt

        tstage = integrator.t + alg.c[2] * integrator.dt
        # k2: Usually required for finest level [1]
        # (Although possible that no scheme has full stage evaluations)
        integrator.f(integrator.du, integrator.u + alg.c[2] * integrator.k1, prob.p, tstage)
        integrator.k_higher = integrator.du * integrator.dt

        for stage = 3:alg.NumStages
          # Construct current state
          integrator.u_tmp = integrator.u + (alg.c[stage] - alg.ACoeffs[stage - 2, 1]) * integrator.k1 + 
              alg.ACoeffs[stage - 2, 1] * integrator.k_higher

          tstage = integrator.t + alg.c[stage] * integrator.dt

          # Joint RHS evaluation with all elements sharing this timestep
          integrator.f(integrator.du, integrator.u_tmp, prob.p, tstage)

          integrator.k_higher = integrator.du * integrator.dt
        end
        integrator.u += integrator.k_higher
        =#
        #=
        if SSPStep
          println("SSP step")
          SSPStep = false

          # Heuristic to ensure this is a stable timestep. 
          # If Heun where optimal, one could multiply with a factor 2 here - but it is most likely not.
          dt = integrator.dt / (2 * alg.NumStages)
          
          # k1
          integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
          integrator.k1 = integrator.du * dt

          # k2
          integrator.f(integrator.du, integrator.u + integrator.k1, prob.p, integrator.t + dt)
          integrator.k_higher = integrator.du * dt

          integrator.u += 0.5 * (integrator.k1 + integrator.k_higher)
          integrator.t += dt
        else
          println("PERK step")
          SSPStep = true
          =#
          # k1: Evaluated on entire domain / all levels
          integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
          integrator.k1 = integrator.du * integrator.dt

          tstage = integrator.t + alg.c[2] * integrator.dt
          # k2: Usually required for finest level [1]
          # (Although possible that no scheme has full stage evaluations)
          integrator.f(integrator.du, integrator.u + alg.c[2] * integrator.k1, prob.p, tstage, 
                      integrator.level_info_elements_acc[1],
                      integrator.level_info_interfaces_acc[1],
                      integrator.level_info_boundaries_acc[1],
                      integrator.level_info_mortars_acc[1])

          integrator.k_higher = integrator.du * integrator.dt
          
          for stage = 3:alg.NumStages
            # Construct current state
            integrator.u_tmp = copy(integrator.u)

            #TODO: Implement fall-back for case when number of available integrators and mesh-partitions not match
            for level in eachindex(integrator.level_u_indices_elements)
              integrator.u_tmp[integrator.level_u_indices_elements[level]] += 
                (alg.c[stage] - alg.ACoeffs[stage - 2, level]) *
                  integrator.k1[integrator.level_u_indices_elements[level]] + 
                alg.ACoeffs[stage - 2, level] * 
                  integrator.k_higher[integrator.level_u_indices_elements[level]] 
            end

            tstage = integrator.t + alg.c[stage] * integrator.dt

            # "ActiveLevels" cannot be static for AMR, has to be checked with available levels
            CoarsestLevel = maximum(alg.ActiveLevels[stage][alg.ActiveLevels[stage] .<= 
                                    length(integrator.level_info_elements_acc)])
                                  
            # Joint RHS evaluation with all elements sharing this timestep
            integrator.f(integrator.du, integrator.u_tmp, prob.p, tstage, 
                        integrator.level_info_elements_acc[CoarsestLevel],
                        integrator.level_info_interfaces_acc[CoarsestLevel],
                        integrator.level_info_boundaries_acc[CoarsestLevel],
                        integrator.level_info_mortars_acc[CoarsestLevel])

            # Update k_higher of relevant levels
            for level in 1:CoarsestLevel
              integrator.k_higher[integrator.level_u_indices_elements[level]] = 
                integrator.du[integrator.level_u_indices_elements[level]] * integrator.dt
            end
          end
          
          integrator.u += integrator.k_higher
          integrator.t += integrator.dt
        #end
      end # PERK step
  
      integrator.iter += 1
      #integrator.t += integrator.dt
  
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
    end # "main loop" timer
  
    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
  end
  
  # get a cache where the RHS can be stored
  get_du(integrator::PERK_Integrator) = integrator.du
  get_tmp_cache(integrator::PERK_Integrator) = (integrator.u_tmp,)
  
  # some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
  u_modified!(integrator::PERK_Integrator, ::Bool) = false
  
  # used by adaptive timestepping algorithms in DiffEq
  function set_proposed_dt!(integrator::PERK_Integrator, dt)
    integrator.dt = dt
  end
  
  # stop the time integration
  function terminate!(integrator::PERK_Integrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
  end
  
  # used for AMR (Adaptive Mesh Refinement)
  function Base.resize!(integrator::PERK_Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)
  end
  
  end # @muladd