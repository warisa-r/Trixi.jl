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

function ComputeFE2S_Coefficients(NumStages::Int, PathPseudoExtrema::AbstractString)

  PathPureReal = PathPseudoExtrema * "PureReal" * string(NumStages) * ".txt"
  NumPureReal, PureReal = ReadInFile(PathPureReal, Float64)

  # CARE: This assumes there is only one pure real root (left end of the spectrum)
  @assert NumPureReal == 1
  ForwardEulerWeight = -1.0 / PureReal[1]
  #println(ForwardEulerWeight)

  PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(NumStages) * ".txt"
  NumTrueComplex, TrueComplex = ReadInFile(PathTrueComplex, Complex{Float64})
  @assert NumTrueComplex == NumStages / 2 - 1

  # Sort ascending => ascending timesteps (real part is always negative)
  perm = sortperm(real.(TrueComplex))
  TrueComplex = TrueComplex[perm]

  #=
  Variables of the NumTrueComplex different Butcher Tableaus of form 

    |
  a | a
  __|______
    | b1 b2 

  We have 3 degrees of freedom, but only two constraints.
  This leaves one with some flexibility.
  =#

  # Find first element where a timestep would be greater 1 => Avoid this
  IndGreater1 = findfirst(x->real(x) > -1.0, TrueComplex)

  a  = zeros(NumTrueComplex)
  b1 = zeros(NumTrueComplex)
  b2 = zeros(NumTrueComplex)

  # TODO: Performance gain possible by only saving one of b1 / b2 here and passing on "IndGreater1"!
  for i in 1:IndGreater1 - 1
    a[i]  = -1.0 / real(TrueComplex[i])
    b1[i] = -real(TrueComplex[i]) / (abs(TrueComplex[i]) .* abs.(TrueComplex[i]))
    b2[i] = b1[i]
  end

  # To avoid timesteps > 1, compute difference between current maximum timestep = a[IndGreater1 - 1] +  and 1.
  dtGreater1 = (1.0 - a[IndGreater1 - 1]) / (NumTrueComplex - IndGreater1 + 1)

  for i in IndGreater1:NumTrueComplex
    # Fill gap a[IndGreater1 - 1] to 1 equidistantly
    a[i]  = a[IndGreater1 - 1] + dtGreater1 * (i - IndGreater1 + 1)

    b2[i] = 1.0 / (abs(TrueComplex[i]) * abs(TrueComplex[i]) * a[i])
    b1[i] = -2.0 * real(TrueComplex[i]) / (abs(TrueComplex[i]) * abs(TrueComplex[i])) - b2[i]
  end

  #=
  display(a); println()
  display(b1); println()
  display(b2); println()
  =#

  return ForwardEulerWeight, a, b1, b2
end

### Based on file "methods_2N.jl", use this as a template for P-ERK RK methods

"""
    FE2S()

The following structures and methods provide a minimal implementation of
the 'ForwardEulerTwoStep' temporal integrator.

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{5,4}{4,3} methods.
"""

mutable struct FE2S
  NumStages::Int
  dtOpt::Real

  ForwardEulerWeight::AbstractFloat
  a::Vector{AbstractFloat}
  b1::Vector{AbstractFloat}
  b2::Vector{AbstractFloat}


  # Constructor for previously computed A Coeffs
  function FE2S(NumStages_::Int, dtOpt_::Real, PathPseudoExtrema_::AbstractString)
    newFE2S = new(NumStages_, dtOpt_)

    newFE2S.ForwardEulerWeight, newFE2S.a, newFE2S.b1, newFE2S.b2 = ComputeFE2S_Coefficients(NumStages_, PathPseudoExtrema_)

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

    # one time step
    # TODO: Multi-threaded execution as implemented for other integrators instead of vectorized operations
    @trixi_timeit timer() "Forward Euler Two Stage ODE integration step" begin
      t_stage = integrator.t

      # Call "rhs!" of the semidiscretization
      integrator.f(integrator.du, integrator.u, prob.p, t_stage) # du = k1

      integrator.u_tmp .= integrator.u .+ alg.ForwardEulerWeight * integrator.dt .* integrator.du
      t_stage += alg.ForwardEulerWeight * integrator.dt

      # Intermediate "two-step" sub methods
      for i in eachindex(alg.a)
        # Low-storage implementation (only one k = du):
        integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stage) # du = k1

        integrator.u_tmp .+= integrator.dt .* alg.b1[i] .* integrator.du

        t_stage += alg.a[i] * integrator.dt
        integrator.f(integrator.du, integrator.u_tmp .+ integrator.dt .*(alg.a[i] - alg.b1[i]) .* integrator.du, 
                     prob.p, t_stage) # du = k2
        integrator.u_tmp .+= integrator.dt .* alg.b2[i] .* integrator.du
      end
      # Final Euler step with step length of dt
      integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stage) # k1
      integrator.u .+= integrator.dt .* integrator.du
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

