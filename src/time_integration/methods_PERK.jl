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

function ComputePERK_ButcherTableau(NumStageEvalsMin::Int, NumDoublings::Int, NumStages::Int, 
                                    PathAUnknowns::AbstractString)

  # c Vector form Butcher Tableau (defines timestep per stage)
  c = zeros(Float64, NumStages)
  for k in 2:NumStages
    c[k] = (k - 1)/(2.0*(NumStages - 1))
  end
  println("Timestep-split: "); display(c); println("\n")

  # - 2 Since First entry of A is always zero (explicit method)
  # and second is given by c (PERK specific)
  StagesMax = NumStageEvalsMin * 2^NumDoublings -2

  ACoeffs = zeros(StagesMax, NumDoublings+1)

  for i = NumDoublings+1:-1:1
    PathA = PathAUnknowns * "a" * string(Int(NumStageEvalsMin * 2^(i-1))) * ".txt"
    NumA, A = ReadInFile(PathA, Float64)
    @assert NumA == NumStageEvalsMin * 2^(i- 1) - 2
    ACoeffs[StagesMax - (NumStageEvalsMin * 2^(i- 1) - 3):end, i] = A
  end
  perm = Vector(NumDoublings+1:-1:1)
  Base.permutecols!!(ACoeffs, perm)
  display(ACoeffs); println()

  return ACoeffs, c
end

### Based on file "methods_2N.jl", use this as a template for P-ERK RK methods

"""
    PERK()

The following structures and methods provide a minimal implementation of
the paired explicit Runge-Kutta method optimized for a certain simulation setup.

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{54, 43} methods.
"""

mutable struct PERK
  NumStageEvalsMin::Int
  NumDoublings::Int
  NumStages::Int
  dtOptMin::Real

  ACoeffs::Matrix{AbstractFloat}
  c::Vector{AbstractFloat}

  dtOpt::AbstractFloat

  # Constructor for previously computed A Coeffs
  function PERK(NumStageEvalsMin_::Int, NumDoublings_::Int, NumStages_::Int, dtOptMin_::Real,
                PathACoeffs_::AbstractString)
    newPERK = new(NumStageEvalsMin_, NumDoublings_, NumStages_, dtOptMin_)

    newPERK.ACoeffs, newPERK.c = 
      ComputePERK_ButcherTableau(NumStageEvalsMin_, NumDoublings_, NumStages_, PathACoeffs_)

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

      # Partitioned RK approach
      kfast = zeros(length(integrator.u), alg.NumStages)
      kslow = zeros(length(integrator.u), alg.NumStages)

      # k1
      integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
      kfast[:, 1] = integrator.du * integrator.dt
      kslow[:, 1] = integrator.du * integrator.dt

      # k2
      integrator.f(integrator.du, integrator.u + alg.c[2] .* kfast[:, 1], prob.p, integrator.t)
      kfast[:, 2] = integrator.du * integrator.dt
      kslow[:, 2] = integrator.du * integrator.dt

      #for stage = 1:alg.NumStages - 2
      for stage = 1:1
        tmp = integrator.u

        tmp[33:96] += (alg.c[stage+2] - alg.ACoeffs[stage, 1]) .* kfast[33:96, 1] 
                     + alg.ACoeffs[stage, 1] .* kfast[33:96, stage + 1]
        tmp[1:32 ] += (alg.c[stage+2] - alg.ACoeffs[stage, 2]) .* kslow[1:32, 1] 
                     + alg.ACoeffs[stage, 2] .* kslow[1:32, stage + 1]

        #=
        # Testcase: Use same update rule throughout the domain
        tmp += (alg.c[stage+2] - alg.ACoeffs[stage, 1]) .* kfast[:, 1] 
              + alg.ACoeffs[stage, 1] .* kfast[:, stage + 1]
        =#

        integrator.f(integrator.du, tmp, prob.p, integrator.t)

        #kfast[33:96, stage+2] = integrator.du[33:96] * integrator.dt
        #kslow[1:32, stage+2]  = integrator.du[1:32] * integrator.dt
        kfast[:, stage+2] = integrator.du * integrator.dt
        kslow[:, stage+2] = integrator.du * integrator.dt

        #display(tmp[33:96]); println()
        #display(tmp[1:32]); println()
      end
      #display(kfast[:, alg.NumStages]); println()
      #display(kslow[:, alg.NumStages]); println()

      #integrator.u += kfast[:, alg.NumStages]
      integrator.u += kfast[:, 3]
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

