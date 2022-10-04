# TODO: Currently hard-coded to second order accurate methods!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

function ReadInFile(FilePath::AbstractString, DataType::Type)
  @assert isfile(FilePath)
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

function ComputeFE2S_Coefficients(StagesMin::Int, NumDoublings::Int, PathPseudoExtrema::AbstractString,
                                  MaxNumTwoStepSubstage::Int)

  ForwardEulerWeights = zeros(NumDoublings+1)
  a  = zeros(MaxNumTwoStepSubstage, NumDoublings+1)
  b1 = zeros(MaxNumTwoStepSubstage, NumDoublings+1)
  b2 = zeros(MaxNumTwoStepSubstage, NumDoublings+1)

  ### Set RKM / Butcher Tableau parameters corresponding to Base-Case (Minimal number stages) ### 

  PathPureReal = PathPseudoExtrema * "PureReal" * string(StagesMin) * ".txt"
  NumPureReal, PureReal = ReadInFile(PathPureReal, Float64)

  @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
  @assert PureReal[1] <= -1.0 "Assume that pure-real pseudo-extremum is smaller then 1.0"
  ForwardEulerWeights[1] = -1.0 / PureReal[1]

  PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(StagesMin) * ".txt"
  NumTrueComplex, TrueComplex = ReadInFile(PathTrueComplex, ComplexF64)
  @assert NumTrueComplex == StagesMin / 2 - 1 "Assume that all but one pseudo-extremum are complex"

  # Sort ascending => ascending timesteps (real part is always negative)
  perm = sortperm(real.(TrueComplex))
  TrueComplex = TrueComplex[perm]

  # Find first element where a timestep would be greater 1 => Special treatment
  IndGreater1 = findfirst(x->real(x) > -1.0, TrueComplex)

  if IndGreater1 === nothing
    # Easy: All can use negated inverse root as timestep
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
    NumPureReal, PureReal = ReadInFile(PathPureReal, Float64)

    @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
    @assert PureReal[1] <= -1.0 "Assume that pure-real pseudo-extremum is smaller then 1.0"
    ForwardEulerWeights[i+1] = -1.0 / PureReal[1]

    PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(Degree) * ".txt"
    NumTrueComplex, TrueComplex = ReadInFile(PathTrueComplex, Complex{Float64})
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
  display(transpose(ForwardEulerWeights) .+ sum(b1, dims=1) .+ sum(b2, dims=1)); println()

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

function ComputeAmpMatrix(StagesMin::Int, NumDoublings::Int, PathPseudoExtrema::AbstractString, 
                          A::Matrix{Float64}, DistributionMatrices)

  StagesMax = -1
  if NumDoublings == 0
    @assert StagesMin % 2 == 0 "Expect even number of Runge-Kutta steges!"
    StagesMax = Int(StagesMin / 2)
  else
    StagesMax = StagesMin * 2^(NumDoublings-1)
  end

  PseudoExtrema = zeros(ComplexF64, StagesMax, NumDoublings+1)

  PathPureReal = PathPseudoExtrema * "PureReal" * string(StagesMin) * ".txt"
  NumPureReal, PureReal = ReadInFile(PathPureReal, Float64)
  @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
  PseudoExtrema[1, 1] = PureReal[1]

  PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(StagesMin) * ".txt"
  NumTrueComplex, TrueComplex = ReadInFile(PathTrueComplex, ComplexF64)
  @assert NumTrueComplex == StagesMin / 2 - 1 "Assume that all but one pseudo-extremum are complex"
  PseudoExtrema[2:Int(StagesMin / 2), 1] = TrueComplex

  for i = 1:NumDoublings
    Degree = StagesMin * 2^i

    PathPureReal = PathPseudoExtrema * "PureReal" * string(Degree) * ".txt"
    NumPureReal, PureReal = ReadInFile(PathPureReal, Float64)
    @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
    PseudoExtrema[1, i+1] = PureReal[1]

    PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(Degree) * ".txt"
    NumTrueComplex, TrueComplex = ReadInFile(PathTrueComplex, Complex{Float64})
    @assert NumTrueComplex == Degree / 2 - 1 "Assume that all but one pseudo-extremum are complex"
    PseudoExtrema[2:Int(Degree / 2), i+1] = TrueComplex
  end

  # Keep same order as the datastructures in other files
  perm = Vector(NumDoublings+1:-1:1)
  Base.permutecols!!(PseudoExtrema, perm)
  #display(PseudoExtrema)

  AmpMatrices = zeros(2, size(A, 1), size(A, 2))
  for i = 1:NumDoublings+1
    # Forward Euler 
    AmpMatrices[i, :, :] = DistributionMatrices[i, :, :] * (I - A ./ PseudoExtrema[1, i])

    # Complex conjugated pseudo-extrema
    for j = 2:Int((StagesMin/2) * 2^(NumDoublings + 1 - i))
      AmpMatrices[i, :, :] *= DistributionMatrices[i, :, :] * real.((I - A ./ PseudoExtrema[j,i]) * 
                                                                    (I - A ./ conj(PseudoExtrema[j,i])))
    end

    # Last euler step
    AmpMatrices[i, :, :] *= DistributionMatrices[i, :, :] * A
    AmpMatrices[i, :, :] += DistributionMatrices[i, :, :] * I
  end
  AmpMatrix = AmpMatrices[1, :, :] + AmpMatrices[2, :, :]

  return AmpMatrix
end

function CheckStabilityJointMethod(StagesMin::Int, NumDoublings::Int, PathPseudoExtrema::String, 
                                   dtMax_::Float64, A::Matrix{Float64})

  DistributionMatrices = zeros(2, 96, 96)
  for i = 8 * 4 + 1:96
    DistributionMatrices[1, i, i] = 1
  end
  DistributionMatrices[2, :, :] = I - DistributionMatrices[1, :, :]

  dtMax = dtMax_
  dtMin = 0.0
  dtEps = 1e-9
  while dtMax - dtMin > dtEps
    dt = 0.5 * (dtMax + dtMin)

    AmpMat = ComputeAmpMatrix(StagesMin, NumDoublings, PathPseudoExtrema, 
                              dt.* A, DistributionMatrices)

    eigs = eigvals(AmpMat)
    SpectralRadius = maximum(abs.(eigs))

    #if SpectralRadius - 1.0 >= eps(Float64)
    if SpectralRadius < 1.0
      dtMax = dt
    else
      dtMin = dt
    end
  end

  return dtMin
end

### Based on file "methods_2N.jl", use this as a template for P-ERK RK methods

"""
    FE2S()

The following structures and methods provide a minimal implementation of
the 'ForwardEulerTwoStep' temporal integrator.

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{54, 43} methods.
"""

mutable struct FE2S
  # Reference = minimum number of stages
  StagesMin::Int
  # Determines how often one doubles the number of stages 
  # Number of methods = NumDoublings +1 
  NumDoublings::Int
  # Timestep corresponding to lowest number of stages
  dtOptMin::Real
  # TODO:Add StagesMax member variable

  ForwardEulerWeights::Vector{Float64}
  a::Matrix{Float64}
  b1::Matrix{Float64}
  b2::Matrix{Float64}
  c::Vector{Float64}

  A::Array{Float64} # Butcher Tableaus

  # Constructor for previously computed A Coeffs
  function FE2S(StagesMin_::Int, NumDoublings_::Int, dtOptMin_::Real, PathPseudoExtrema_::AbstractString, 
                # TODO: A: Only for testing
                A_::Matrix{Float64})
    newFE2S = new(StagesMin_, NumDoublings_, dtOptMin_)

    newFE2S.ForwardEulerWeights, newFE2S.a, newFE2S.b1, newFE2S.b2, newFE2S.c = 
      ComputeFE2S_Coefficients(StagesMin_, NumDoublings_, PathPseudoExtrema_, 
                               Int((StagesMin_ * 2^NumDoublings_ - 2)/2))

    #=                               
    println("Theoretical Max Timestep: ")
    dtMinStable = CheckStabilityJointMethod(StagesMin_, NumDoublings_, PathPseudoExtrema_, dtOptMin_, A_)
    println(dtMinStable)

    println("Which is fraction of theoretical: ", dtMinStable / dtOptMin_)
    =#

    newFE2S.A = BuildButcherTableaus(newFE2S.ForwardEulerWeights, newFE2S.a, newFE2S.b1, newFE2S.b2, newFE2S.c, 
                                     StagesMin_, NumDoublings_)

    return newFE2S
  end

end # struct FE2S


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct FE2S_IntegratorOptions{Callback}
  callback::Callback # callbacks; used in Trixi
  adaptive::Bool # whether the algorithm is adaptive; ignored
  dtmax::Float64 # ignored
  maxiters::Int # maximal numer of time steps
  tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function FE2S_IntegratorOptions(callback, tspan; maxiters=typemax(Int), kwargs...)
  FE2S_IntegratorOptions{typeof(callback)}(callback, false, Inf, maxiters, [last(tspan)])
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct FE2S_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, FE2S_IntegratorOptions}
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
  opts::FE2S_IntegratorOptions
  finalstep::Bool # added for convenience
end

# Forward integrator.destats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::FE2S_Integrator, field::Symbol)
  if field === :destats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::FE2S;
               dt::Real, callback=nothing, kwargs...)
  u0 = copy(ode.u0) # Initial value
  du = similar(u0)
  u_tmp = similar(u0)
  t0 = first(ode.tspan) # Initial time
  iter = 0 # We are in first iteration
  integrator = FE2S_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, 
                 ode.p, # the semidiscretization
                 (prob=ode,), # Not really sure whats going on here
                 ode.f, # the right-hand-side of the ODE u' = f(u, p, t)
                 alg, # The ODE integration algorithm/method
                 FE2S_IntegratorOptions(callback, ode.tspan; kwargs...), false)

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

function solve!(integrator::FE2S_Integrator)
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

    # TODO: Multi-threaded execution as implemented for other integrators instead of vectorized operations
    @trixi_timeit timer() "Forward Euler Two Stage ODE integration step" begin

      # Butcher-Tableau based approach
      # TODO: Correct timesteps not yet covered
      MaxStages = alg.StagesMin * 2^alg.NumDoublings

      kfast = zeros(length(integrator.u), MaxStages)
      kslow = zeros(length(integrator.u), MaxStages)
      integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
      kfast[33:96, 1] = integrator.du[33:96] * integrator.dt
      kslow[1:32, 1] = integrator.du[1:32] * integrator.dt

      for i = 2:MaxStages
        tmp     = integrator.u
        tmpfast = integrator.u
        tmpslow = integrator.u

        for j = 1:i-1
        #for j = 1:i-2
          tmp += alg.A[1, i, j] .* kfast[:, j] + alg.A[2, i, j] .* kslow[:, j]

          # See if splitting of f is correct
          #tmp += alg.A[1, i, j] .* kfast[:, j] + alg.A[1, i, j] .* kslow[:, j]
          
          #tmpfast += alg.A[1, i, j] .* kfast[:, j]
          #tmpslow += alg.A[2, i, j] .* kslow[:, j]
        end

        # Idea: Do not use "half step" of intermediate two-step methods
        #tmpslow += alg.A[2, i, i-1] .* kslow[:, i-1]
        #tmpfast += alg.A[1, i, i-1] .* kfast[:, i-1]

        integrator.f(integrator.du, tmp, prob.p, integrator.t, 1)
        #integrator.f(integrator.du, tmpfast, prob.p, integrator.t, 1)
        #integrator.f(integrator.du, tmpfast, prob.p, integrator.t)
        kfast[:, i] = integrator.du * integrator.dt

        integrator.f(integrator.du, tmp, prob.p, integrator.t, 2)
        #integrator.f(integrator.du, tmpslow, prob.p, integrator.t, 2)
        #integrator.f(integrator.du, tmpslow, prob.p, integrator.t)
        kslow[:, i] = integrator.du * integrator.dt
      end
      #display(kfast[:, MaxStages]); println()
      #display(kslow[:, MaxStages]); println()

      integrator.u += kfast[:, MaxStages] + kslow[:, MaxStages]
      
      #integrator.u += kslow[:, MaxStages]
      #integrator.u += kfast[:, MaxStages]

      #=
      t_stages = integrator.t .* ones(alg.NumDoublings + 1) # TODO: Make member variable?

      # Compute k1
      integrator.f(integrator.du, integrator.u, prob.p, t_stages[1])

      # TODO: Find way of storing only the relevant portions of u
      # Current state of the different levels
      u_levels = zeros(length(integrator.u), alg.NumDoublings+1) # TODO: Make member variable?

      # Perform Forward Euler steps
      for i = 1:alg.NumDoublings + 1
        u_levels[:, i] = integrator.u + integrator.dt .* alg.ForwardEulerWeights[i] .* integrator.du 
        t_stages[i] += alg.ForwardEulerWeights[i] * integrator.dt
      end

      # Intermediate "two-step" sub methods
      for i = 1:alg.NumDoublings + 1
        for step = 1:Int((alg.StagesMin * 2^(alg.NumDoublings + 1 - i) - 2)/2)
          # Low-storage implementation (only one k = du):
          integrator.f(integrator.du, u_levels[:, i], prob.p, t_stages[i], i) # du = k1
          u_levels[:, i] .+= integrator.dt .* alg.b1[step, i] .* integrator.du

          t_stages[i] = integrator.t + alg.a[step, i] * integrator.dt
          integrator.f(integrator.du, u_levels[:, i] .+ integrator.dt .*(alg.a[step, i] - alg.b1[step, i]) .* integrator.du, 
                       prob.p, t_stages[i], i)
          u_levels[:, i] .+= integrator.dt .* alg.b2[step, i] .* integrator.du
        end
      end
      # Final Euler step with step length of dt
      # TODO: Remove assert later (performance)
      @assert all(y->y == t_stages[1], t_stages) # Expect that every stage is at the same time by now

      integrator.f(integrator.du, u_levels[:, 1], prob.p, t_stages[1], 1)
      integrator.u += integrator.dt .* integrator.du
      integrator.f(integrator.du, u_levels[:, 2], prob.p, t_stages[2], 2)
      integrator.u += integrator.dt .* integrator.du
      =#
    end # FE2S step

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
get_du(integrator::FE2S_Integrator) = integrator.du
get_tmp_cache(integrator::FE2S_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::FE2S_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::FE2S_Integrator, dt)
  integrator.dt = dt
end

# stop the time integration
function terminate!(integrator::FE2S_Integrator)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::FE2S_Integrator, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.u_tmp, new_size)
end

end # @muladd