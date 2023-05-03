# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

"""
    Heun()

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

mutable struct Heun
end # struct Heun

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct Heun_IntegratorOptions{Callback}
  callback::Callback # callbacks; used in Trixi
  adaptive::Bool # whether the algorithm is adaptive; ignored
  dtmax::Float64 # ignored
  maxiters::Int # maximal numer of time steps
  tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function Heun_IntegratorOptions(callback, tspan; maxiters=typemax(Int), kwargs...)
  Heun_IntegratorOptions{typeof(callback)}(callback, false, Inf, maxiters, [last(tspan)])
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct Heun_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, Heun_IntegratorOptions}
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
  opts::Heun_IntegratorOptions
  finalstep::Bool # added for convenience
end

# Forward integrator.destats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::Heun_Integrator, field::Symbol)
  if field === :destats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::Heun;
                dt, callback=nothing, kwargs...)

  u0    = copy(ode.u0)
  du    = similar(u0)
  u_tmp = similar(u0)

  t0 = first(ode.tspan)
  iter = 0

  integrator = Heun_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                (prob=ode,), ode.f, alg,
                Heun_IntegratorOptions(callback, ode.tspan; kwargs...), false)
            
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

function solve!(integrator::Heun_Integrator)
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

    @trixi_timeit timer() "Heun ODE integration step" begin

      @threaded for j in eachindex(integrator.u)
        integrator.u_tmp[j] = integrator.u[j] # Used for incremental stage update
      end
      
      # du = k1
      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t)

      # Add already to next state
      @threaded for i in eachindex(integrator.du)
        integrator.u_tmp[i] += 0.5 * integrator.du[i] * integrator.dt
      end

      # Compute k2:
      integrator.f(integrator.du, integrator.u + integrator.dt * integrator.du, prob.p, integrator.t + integrator.dt)
      # Add to next state
      @threaded for i in eachindex(integrator.du)
        integrator.u_tmp[i] += 0.5 * integrator.du[i] * integrator.dt
      end

      @threaded for i in eachindex(integrator.u)
        integrator.u[i] = integrator.u_tmp[i]
      end
    end # Heun step

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
get_du(integrator::Heun_Integrator) = integrator.du
get_tmp_cache(integrator::Heun_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::Heun_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::Heun_Integrator, dt)
  integrator.dt = dt
end

# stop the time integration
function terminate!(integrator::Heun_Integrator)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::Heun_Integrator, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.u_tmp, new_size)

  resize!(integrator.k1, new_size)
  resize!(integrator.k_higher, new_size)
end

end # @muladd