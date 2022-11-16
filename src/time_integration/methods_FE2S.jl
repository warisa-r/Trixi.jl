# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
  
function ComputeFE2S_Coefficients(Stages::Int, PathPseudoExtrema::AbstractString,
                                  NumTrueComplex_::Int)

  PathPureReal = PathPseudoExtrema * "PureReal" * string(Stages) * ".txt"
  NumPureReal, PureReal = read_file(PathPureReal, Float64)

  @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
  @assert PureReal[1] <= -1.0 "Assume that pure-real pseudo-extremum is smaller then 1.0"
  ForwardEulerWeight = -1.0 / PureReal[1]

  PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(Stages) * ".txt"
  NumTrueComplex, TrueComplex = read_file(PathTrueComplex, ComplexF64)
  @assert NumTrueComplex == NumTrueComplex_ "Assume that all but one pseudo-extremum are complex"

  # Sort ascending => ascending timesteps (real part is always negative)
  perm = sortperm(real.(TrueComplex))
  TrueComplex = TrueComplex[perm]

  InvAbsValsSquared     = zeros(NumTrueComplex_ + 1) # 1 / (|r_i|^2)
  TwoRealOverAbsSquared = zeros(NumTrueComplex_ + 1) # -2 * Re(r_i)/(|r_i|^2)
  TimeSteps             = zeros(NumTrueComplex_ + 1) # 1 / (|r_i|^2) + -2 * Re(r_i)/(|r_i|^2)

  TimeSteps[1] = ForwardEulerWeight
  # Dummy to fill position of ForwardEulerWeight
  InvAbsValsSquared[1]     = 42.0
  TwoRealOverAbsSquared[1] = 42.0

  for i = 1:NumTrueComplex
    InvAbsValsSquared[i+1]     = 1.0/(abs(TrueComplex[i])^2)
    TwoRealOverAbsSquared[i+1] = -2 * real(TrueComplex[i]) * InvAbsValsSquared[i+1]

    TimeSteps[i+1] = InvAbsValsSquared[i+1] + TwoRealOverAbsSquared[i+1]
  end

  println("ForwardEulerWeight:\n"); display(ForwardEulerWeight); println("\n")

  # Sort in ascending manner
  perm = sortperm(TimeSteps)
  TimeSteps = TimeSteps[perm]

  # Find position of ForwardEulerWeight after sorting, required to do steps in correct order
  IndexForwardEuler = findfirst(x -> x==ForwardEulerWeight, TimeSteps)

  InvAbsValsSquared     = InvAbsValsSquared[perm]
  TwoRealOverAbsSquared = TwoRealOverAbsSquared[perm]

  println("InvAbsValsSquared:\n"); display(InvAbsValsSquared); println("\n")
  println("TwoRealOverAbsSquared:\n"); display(TwoRealOverAbsSquared); println("\n")

  println("TimeSteps:\n"); display(TimeSteps); println("\n")
  println("Sum of Timesteps:\n");  println(sum(TimeSteps))

  return ForwardEulerWeight, InvAbsValsSquared, TwoRealOverAbsSquared, TimeSteps, IndexForwardEuler
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
  const Stages::Int
  # Maximum Number of True Complex Pseudo Extrema, relevant for many datastructures
  const NumTrueComplex::Int

  ForwardEulerWeight::Float64
  InvAbsValsSquared::Vector{Float64}
  TwoRealOverAbsSquared::Vector{Float64}
  TimeSteps::Vector{Float64}
  IndexForwardEuler::Int

  # Constructor for previously computed A Coeffs
  function FE2S(Stages_::Int, PathPseudoExtrema_::AbstractString)
    if Stages_ % 2 == 0 
      NumTrueComplex_ = Int(Stages_ / 2 - 1)
    else
      NumTrueComplex_ = Int((Stages_ - 1) / 2)
    end
    newFE2S = new(Stages_, NumTrueComplex_)

    newFE2S.ForwardEulerWeight, newFE2S.InvAbsValsSquared, newFE2S.TwoRealOverAbsSquared, 
    newFE2S.TimeSteps, newFE2S.IndexForwardEuler = 
      ComputeFE2S_Coefficients(Stages_, PathPseudoExtrema_, newFE2S.NumTrueComplex)

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
  k1::uType # Intermediate stage 
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

  u0    = copy(ode.u0) # Initial value
  du    = similar(u0)
  u_tmp = similar(u0)

  k1 = similar(u0)

  t0 = first(ode.tspan)
  iter = 0

  integrator = FE2S_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, 
                 ode.p, # the semidiscretization
                 (prob=ode,), # Not really sure whats going on here
                 ode.f, # the right-hand-side of the ODE u' = f(u, p, t)
                 alg, # The ODE integration algorithm/method
                 FE2S_IntegratorOptions(callback, ode.tspan; kwargs...), false, k1)

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
      @trixi_timeit timer() "Measure copy" begin
        integrator.u_tmp = copy(integrator.u) # Used for incremental stage update
      end

    t_stage = integrator.t
    # Two-stage substeps with smaller timestep than ForwardEuler
    for i = 1:alg.IndexForwardEuler-1
      if i > 1
        t_stage += integrator.dt * alg.TimeSteps[i-1]
      end

      integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stage)

      @threaded for j in eachindex(integrator.du)
        integrator.k1[j] = integrator.dt * integrator.du[j]
      end

      t_stage += alg.InvAbsValsSquared[i]
      @trixi_timeit timer() "Second rhs" begin
      integrator.f(integrator.du, integrator.u_tmp .* alg.TwoRealOverAbsSquared[i] .+ 
                                  integrator.k1 .* alg.InvAbsValsSquared[i], prob.p, t_stage)
      end
                                
      @threaded for j in eachindex(integrator.du)
        integrator.u_tmp[j] += integrator.dt * integrator.du[j]
      end
      t_stage += alg.TwoRealOverAbsSquared[i]
    end

    # Forward Euler step
    integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t) # du = k1
    
    @threaded for j in eachindex(integrator.du)
      integrator.u_tmp[j] += alg.ForwardEulerWeight * integrator.dt * integrator.du[j]
    end

    # Two-stage substeps with bigger timestep than ForwardEuler
    for i = alg.IndexForwardEuler+1:length(alg.InvAbsValsSquared)
      t_stage += integrator.dt * alg.TimeSteps[i-1]

      integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stage)
      @threaded for j in eachindex(integrator.du)
        integrator.k1[j] = integrator.dt * integrator.du[j]
        #=
        integrator.k1[j] = integrator.dt * integrator.du[j] * alg.InvAbsValsSquared[j] + 
                           integrator.u_tmp .* alg.TwoRealOverAbsSquared[j]
        =#
      end

      t_stage += alg.InvAbsValsSquared[i]
      @trixi_timeit timer() "Second rhs" begin
      integrator.f(integrator.du, integrator.u_tmp .* alg.TwoRealOverAbsSquared[i] .+ 
                                  integrator.k1 .* alg.InvAbsValsSquared[i], prob.p, t_stage)
      end

      @threaded for j in eachindex(integrator.du)                                  
        integrator.u_tmp[j] += integrator.dt * integrator.du[j]
      end
      t_stage += alg.TwoRealOverAbsSquared[i]
    end

    t_stage = integrator.t + alg.TimeSteps[end] * integrator.dt
    # Final Euler step with step length of dt (Due to form of stability polynomial)
    integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stage)
    @threaded for j in eachindex(integrator.du)
      integrator.u[j] += integrator.dt * integrator.du[j]
    end
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
