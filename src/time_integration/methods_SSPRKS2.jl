# TODO: Currently hard-coded to second order accurate methods!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

"""
    SSPRKS2()

The following structures and methods provide a minimal implementation of
the optimal second order accurate, S-stage method family.
The stability polynomial is the optimal second order accurate stability polynomial for the circle
and comes with many other advantageous properties.
See e.g.
https://doi.org/10.1016/j.jcp.2004.05.002
https://doi.org/10.1137/130936245
https://doi.org/10.1142/7498

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{54, 43} methods.
"""

function MaxTimeStep(NumStages::Int, dtMax::Float64, EigVals::Vector{<:ComplexF64})
  dtEps = 1e-9
  dt    = -1.0
  dtMin = 0.0

  while dtMax - dtMin > dtEps
    dt = 0.5 * (dtMax + dtMin)

    AbsPMax = -1
    for i in eachindex(EigVals)
      AbsP = abs(1.0/NumStages + (NumStages - 1)/NumStages * (1.0 + EigVals[i] * dt/(NumStages - 1))^NumStages)

      if AbsP > AbsPMax
        AbsPMax = AbsP
      end

      if AbsPMax > 1.0
        break
      end
    end

    if AbsPMax > 1.0
      dtMax = dt
    else
      dtMin = dt
    end

    println("Current dt: ", dt)
    println("Current AbsPMax: ", AbsPMax, "\n")
  end

  return dt
end

mutable struct SSPRKS2
  const NumStages::Int
  
  function SSPRKS2(NumStages_::Int)

    newSSPRKS2 = new(NumStages_)

    return newSSPRKS2
  end
end # struct SSPRKS2


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct SSPRKS2_IntegratorOptions{Callback}
  callback::Callback # callbacks; used in Trixi
  adaptive::Bool # whether the algorithm is adaptive; ignored
  dtmax::Float64 # ignored
  maxiters::Int # maximal numer of time steps
  tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function SSPRKS2_IntegratorOptions(callback, tspan; maxiters=typemax(Int), kwargs...)
  SSPRKS2_IntegratorOptions{typeof(callback)}(callback, false, Inf, maxiters, [last(tspan)])
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct SSPRKS2_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, SSPRKS2_IntegratorOptions}
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
  opts::SSPRKS2_IntegratorOptions
  finalstep::Bool # added for convenience
end

# Forward integrator.destats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::SSPRKS2_Integrator, field::Symbol)
  if field === :destats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::SSPRKS2;
                dt, callback=nothing, kwargs...)

  u0    = copy(ode.u0)
  du    = similar(u0)
  u_tmp = similar(u0)

  t0 = first(ode.tspan)
  iter = 0


  integrator = SSPRKS2_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                (prob=ode,), ode.f, alg,
                SSPRKS2_IntegratorOptions(callback, ode.tspan; kwargs...), false)
            
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

function solve!(integrator::SSPRKS2_Integrator)
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

    @trixi_timeit timer() "SSPRKS2 ODE integration step" begin

      @threaded for j in eachindex(integrator.u)
        integrator.u_tmp[j] = integrator.u[j] # Used for incremental stage update
      end
      
      for stage = 1:alg.NumStages

        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t)

        @threaded for i in eachindex(integrator.du)
          integrator.u_tmp[i] += integrator.dt/(alg.NumStages - 1) * integrator.du[i]
        end
      end

      @threaded for i in eachindex(integrator.u)
        integrator.u[i] *= 1/alg.NumStages
        integrator.u[i] += (alg.NumStages - 1)/alg.NumStages * integrator.u_tmp[i]
      end
    end # SSPRKS2 step

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
  end # "main loop" timer
  
  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u),
                                integrator.sol.prob)
end

# get a cache where the RHS can be stored
get_du(integrator::SSPRKS2_Integrator) = integrator.du
get_tmp_cache(integrator::SSPRKS2_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SSPRKS2_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::SSPRKS2_Integrator, dt)
  integrator.dt = dt
end

# stop the time integration
function terminate!(integrator::SSPRKS2_Integrator)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::SSPRKS2_Integrator, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.u_tmp, new_size)

  resize!(integrator.k1, new_size)
  resize!(integrator.k_higher, new_size)
end

end # @muladd