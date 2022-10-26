# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
  
function ComputeFE2S_Coefficients(Stages::Int, PathPseudoExtrema::AbstractString,
                                  NumTrueComplex_::Int)

  c  = zeros(Stages)                                  
  a  = zeros(NumTrueComplex_)
  b1 = zeros(NumTrueComplex_)
  b2 = zeros(NumTrueComplex_)

  ### Set RKM / Butcher Tableau parameters corresponding to Base-Case (Minimal number stages) ### 

  PathPureReal = PathPseudoExtrema * "PureReal" * string(Stages) * ".txt"
  NumPureReal, PureReal = read_file(PathPureReal, Float64)

  @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
  @assert PureReal[1] <= -1.0 "Assume that pure-real pseudo-extremum is smaller then 1.0"
  ForwardEulerWeight = -1.0 / PureReal[1]
  # Set timestep to previous forward euler step
  c[2] = ForwardEulerWeight

  PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(Stages) * ".txt"
  NumTrueComplex, TrueComplex = read_file(PathTrueComplex, ComplexF64)
  @assert NumTrueComplex == NumTrueComplex_ "Assume that all but one pseudo-extremum are complex"

  # Sort ascending => ascending timesteps (real part is always negative)
  perm = sortperm(real.(TrueComplex))
  TrueComplex = TrueComplex[perm]

  
  for i = 1:NumTrueComplex
    AbsSquared = abs(TrueComplex[i]) * abs(TrueComplex[i])

    # Relevant Coefficients of (1 - 1/(Re + i * Im)) * (1 - 1/(Re - i * Im))
    a_1 = -2.0 * real(TrueComplex[i]) / AbsSquared
    a_2 = 1.0 / AbsSquared


    # Set a, b1, b2 such that they are all evaluated at the same timestep.
    a[i] = a_1

    b2[i] = a_2 / a[i]
    b1[i] = a_1 - b2[i]

    c[2*i + 1] = c[2*i] + a_1
    c[2*i + 2] = c[2*i + 1]
  end
  

  println("ForwardEulerWeight:\n"); display(ForwardEulerWeight); println("\n")
  println("a:\n"); display(a); println("\n")
  println("b1:\n"); display(b1); println("\n")
  println("b2:\n"); display(b2); println("\n")

  println("c:\n"); display(c); println("\n")

  return ForwardEulerWeight, a, b1, b2, c, TrueComplex
end

function BuildButcherTableau(ForwardEulerWeight::Float64, a::Vector{Float64}, 
                             b1::Vector{Float64}, b2::Vector{Float64},
                             Stages::Int, NumTrueComplex::Int)

  A = zeros(Stages, Stages)

  # Forward Euler contribution
  A[2:end, 1] .= ForwardEulerWeight

  # Intermediate two-step submethods
  for j = 1:NumTrueComplex
    A[2*j + 1, 2*j] = a[j]

    A[(2 + 2 * j):end, 2*j]     .= b1[j]
    A[(2 + 2 * j):end, 2*j + 1] .= b2[j]
  end

  println("Butcher Tableau matrix A:\n"); display(A); println()

  c = sum(A, dims = 2)
  println("Timesteps set as c_i = sum_j a_{ij}:\n"); display(c); println()

  return A
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
  a::Vector{Float64}
  b1::Vector{Float64}
  b2::Vector{Float64}
  c::Vector{Float64}
  TrueComplex::Vector{ComplexF64}

  A::Matrix{Float64} # Butcher Tableau (Not used)

  # Constructor for previously computed A Coeffs
  function FE2S(Stages_::Int, PathPseudoExtrema_::AbstractString)
    @assert Stages_ % 2 == 0 "Support only even number of stages for the moment"
    newFE2S = new(Stages_, Int(Stages_ / 2 - 1))

    newFE2S.ForwardEulerWeight, newFE2S.a, newFE2S.b1, newFE2S.b2, newFE2S.c, newFE2S.TrueComplex = 
      ComputeFE2S_Coefficients(Stages_, PathPseudoExtrema_, newFE2S.NumTrueComplex)

    newFE2S.A = BuildButcherTableau(newFE2S.ForwardEulerWeight, newFE2S.a, newFE2S.b1, newFE2S.b2, 
                                    Stages_, newFE2S.NumTrueComplex)

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

  u0    = copy(ode.u0) # Initial value
  du    = similar(u0)
  u_tmp = similar(u0)

  t0 = first(ode.tspan)
  iter = 0

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
      
      # First Forward Euler step
      integrator.f(integrator.du, integrator.u, prob.p, integrator.t) # du = k1
      integrator.u_tmp = integrator.u + alg.ForwardEulerWeight * integrator.dt * integrator.du
      t_stage = integrator.t + alg.c[2] * integrator.dt

      # Intermediate "two-step" sub methods
      for i in eachindex(alg.a)
        #=
        # Low-storage implementation (only one k = du):
        integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stage) # du = k1

        integrator.u_tmp += integrator.dt * alg.b1[i] * integrator.du

        t_stage = integrator.t + alg.c[2*i + 1] * integrator.dt
        integrator.f(integrator.du, integrator.u_tmp + integrator.dt *(alg.a[i] - alg.b1[i]) * integrator.du, 
                     prob.p, t_stage) # du = k2
        integrator.u_tmp += integrator.dt * alg.b2[i] * integrator.du
        =#
        
        # Another version, "directly read-off"
        # TODO: Do timesteps c for this!
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t)
        k1 = integrator.dt * integrator.du

        AbsValSqaured = abs(alg.TrueComplex[i]) * abs(alg.TrueComplex[i])
        integrator.f(integrator.du, integrator.u_tmp * (-2 * real(alg.TrueComplex[i]) / AbsValSqaured) + 
                                    k1 / AbsValSqaured, prob.p, integrator.t)

        integrator.u_tmp += integrator.du * integrator.dt
      end

      t_stage = integrator.t + alg.c[end] * integrator.dt
      # Final Euler step with step length of dt (Due to form of stability polynomial)
      integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stage) # k1
      integrator.u += integrator.dt * integrator.du
      

      #=
      k = zeros(length(integrator.u), alg.Stages)      
      # k1
      integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
      k[:, 1] = integrator.du * integrator.dt

      for i = 2:alg.Stages
        tmp = copy(integrator.u)

        # Partitioned Runge-Kutta approach: One state that contains updates from all levels
        for j = 1:i-1
          tmp += alg.A[i, j] * k[:, j]
        end

        integrator.f(integrator.du, tmp, prob.p, integrator.t)

        k[:, i] = integrator.du * integrator.dt
      end
      integrator.u += k[:, end]
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
