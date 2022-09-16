# TODO: Currently hard-coded to second order accurate methods!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

function ComputePERK_ButcherTableau(NumStages::Int, NumStageEvals::Int, NumEvalReduction::Int,
  PathAUnknowns::AbstractString)

  # Now compute Butcher Tableau coefficients
  if NumStages > 2
    @assert isfile(PathAUnknowns)
    AUnknown = Vector{Float64}(undef, NumStageEvals)
    open(PathAUnknowns, "r") do AUnknownsFile
      index = 1
      while !eof(AUnknownsFile)
        LineContent = readline(AUnknownsFile)
        AUnknown[index] = parse(Float64, LineContent)
        index += 1
      end
    end
  end # Else: For order 2: Explicit midpoint: https://en.wikipedia.org/wiki/Midpoint_method

  # TODO: Read-in from C++ (for higher precision) ?
  # c Vector form Butcher Tableau (defines timestep per stage)
  c = zeros(Float64, NumStages)
  for k in 2:NumStages
    c[k] = (k - 1)/(2.0*(NumStages - 1))
  end
  #println("Timestep-split: "); display(c); println("\n")

  ### Assemble Butcher Tableau coeffcients in (relatively) sparse & easy to handle datastructure ###
  ACoeffs = zeros(Float64, NumStages, 2)
  # Rows with only entry in first column
  for k in 2:2 + NumEvalReduction
    ACoeffs[k, 1] = c[k]
    #ACoeffs[k, 2] = 0 # This is the core idea of reduced evaluations: Set sub-diagonal to zero
  end

  for k in 3 + NumEvalReduction:NumStages
    ACoeffs[k, 1] = c[k] - AUnknown[k - (3 + NumEvalReduction) + 1]
    ACoeffs[k, 2] = AUnknown[k - (3 + NumEvalReduction) + 1]
  end
  #println("Butcher Tableau Coefficient Matrix A:"); display(ACoeffs); println("\n")

  return ACoeffs, c
end

### Based on file "methods_2N.jl", use this as a template for P-ERK RK methods

"""
    PERK()

The following structures and methods provide a minimal implementation of
the paired explicit Runge-Kutta method optimized for a certain simulation setup.

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{5,4}{4,3} methods.
"""

mutable struct PERK
  NumStages::Int
  NumStageEvals::Int
  NumEvalReduction::Int
  # TODO: Consistency Order

  ACoeffs::Matrix{AbstractFloat}
  c::Vector{AbstractFloat}

  dtOpt::AbstractFloat

  # Constructor for previously computed A Coeffs
  function PERK(NumStages_::Int, NumStageEvals_::Int, OptFilesLoc::AbstractString)
    newPERK = new(NumStages_, NumStageEvals_, NumStages_ - NumStageEvals_)

    @assert isfile(OptFilesLoc * "dtOpt.txt")
    open(OptFilesLoc * "dtOpt.txt", "r") do dtOptFile
      while !eof(dtOptFile)
        lineContent = readline(dtOptFile)
        newPERK.dtOpt = parse(Float64, lineContent)
      end
    end

    newPERK.ACoeffs, newPERK.c = ComputePERK_ButcherTableau(NumStages_, NumStageEvals_,
                                                            newPERK.NumEvalReduction,
                                                            OptFilesLoc * "ACoeffs" * string(NumStages_) *
                                                            "_" * string(NumStageEvals_) * ".txt")

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
  # TODO: Add k1, khigher here?
  du::uType # Not used, for compliance with ODE solve
  u_tmp::uType # Not used, for compliance with ODE solve
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
  du = similar(u0) # Not used, for compliance with ODE solve
  u_tmp = similar(u0) # Not used, for compliance with ODE solve
  t0 = first(ode.tspan)
  iter = 0
  integrator = PERK_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                  (prob=ode,), ode.f, alg,
                  PERK_IntegratorOptions(callback, ode.tspan; kwargs...), false)

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

    # TODO: Try multi-threaded execution as implemented for other integrators!
    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
      # NOTE: Possible make these member variables of integrator to avoid repeated declaration ?
      k1      = similar(integrator.u)
      kHigher = similar(integrator.u)

      # Treat first two stages seperately
      # Stage 1: Note: Hard-coded to c[0] = 0! (tstage = t)
      integrator.f(k1, integrator.u, prob.p, integrator.t)

      # Stage 2: (Adapted for only coefficient a_{21})
      t_stage = integrator.t + integrator.dt * alg.c[2]
      integrator.f(kHigher, integrator.u .+ integrator.dt * alg.ACoeffs[2, 1] .* k1, prob.p, t_stage)

      for stage in 3:2+alg.NumEvalReduction # Here, the non-zero coeffs are only in the first column
        t_stage = integrator.t + integrator.dt * alg.c[stage]
        integrator.f(kHigher, integrator.u .+ integrator.dt * (alg.ACoeffs[stage, 1] .* k1), prob.p, t_stage)
      end

      # Higher stages with (in general) two coefficients
      for stage in 3+alg.NumEvalReduction:alg.NumStages # Here, there are non-zero coeffs in the first column and on sub-diagonal
        t_stage = integrator.t + integrator.dt * alg.c[stage]
        integrator.f(kHigher, integrator.u .+ integrator.dt * (alg.ACoeffs[stage, 1] .* k1 + alg.ACoeffs[stage, 2] .* kHigher), prob.p, t_stage)
      end

      # Final step: Update u (only b_s = 1, other b_i = 0)
      integrator.u += integrator.dt * kHigher
    end # PERK step

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
end

end # @muladd

