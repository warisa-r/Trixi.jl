# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

struct Limiterp1Callback{Indicator}
  indicator::Indicator
  # TODO: Different limiters (minmod, monotonized central, superbee, van-Leer, ...)
end

function Limiterp1Callback(indicator)
  limiterp1_callback = Limiterp1Callback{typeof(indicator)}(indicator)

  condition = (u, t, integrator) -> true # Called every timestep TODO: Maybe invoke indicator already here?
  DiscreteCallback(condition, limiterp1_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:Limiterp1Callback})
  print(io, "Limiterp1Callback")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:Limiterp1Callback})
  @nospecialize cb # reduce precompilation time

  if get(io, :compact, false)
    show(io, cb)
  else
    #limiterp1_callback = cb.affect!

    setup = [
             "Limiter" => "MinMod", # limiterp1_callback.limiter
            ]
    summary_box(io, "Limiterp1Callback", setup)
  end
end

function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:Limiterp1Callback}
  # TODO: Limiting of initial condition desired?
  cb.affect!(integrator)
  return nothing
end

function (limiterp1_callback::Limiterp1Callback)(integrator; kwargs...)
  u_ode = integrator.u
  semi = integrator.p

  @trixi_timeit timer() "Limiter_p1" begin
    limiterp1_callback(u_ode, semi,
                       integrator.t, integrator.iter; kwargs...)
  end
end

@inline function (limiterp1_callback::Limiterp1Callback)(u_ode::AbstractVector,
                                                         semi::SemidiscretizationHyperbolic,
                                                         t, iter;
                                                         kwargs...)
  # Note that we don't `wrap_array` the vector `u_ode` to be able to `resize!`
  # it when doing AMR while still dispatching on the `mesh` etc.
  limiterp1_callback(u_ode, mesh_equations_solver_cache(semi)..., semi, t, iter; kwargs...)
end

function (limiterp1_callback::Limiterp1Callback)(u_ode::AbstractVector, mesh::TreeMesh,
                                                 equations, dg::DG, cache, semi,
                                                 t, iter;
                                                 only_refine=false, only_coarsen=false,
                                                 passive_args=())

  @unpack indicator = limiterp1_callback

  u = wrap_array(u_ode, mesh, equations, dg, cache)
  alpha = @trixi_timeit timer() "indicator" indicator(u, mesh, equations, dg, cache,
                                                      t=t, iter=iter)
  writedlm("alpha.txt", alpha) 
end

end # @muladd