using DelimitedFiles

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
  
  # Sort in DESCENDING manner
  #perm = sortperm(TimeSteps, rev=true)
  TimeSteps = TimeSteps[perm]

  # Find position of ForwardEulerWeight after sorting, required to do steps in correct order
  IndexForwardEuler = findfirst(x -> x==ForwardEulerWeight, TimeSteps)

  InvAbsValsSquared     = InvAbsValsSquared[perm]
  TwoRealOverAbsSquared = TwoRealOverAbsSquared[perm]

  println("InvAbsValsSquared:\n"); display(InvAbsValsSquared); println("\n")
  println("TwoRealOverAbsSquared:\n"); display(TwoRealOverAbsSquared); println("\n")

  println("TimeSteps:\n"); display(TimeSteps); println("\n")
  println("Sum of Timesteps:\n\n");  println(sum(TimeSteps))

  return ForwardEulerWeight, InvAbsValsSquared, TwoRealOverAbsSquared, TimeSteps, IndexForwardEuler, perm
end

function ComputeFE2S_Coefficients_RealRK(Stages::Int, PathPseudoExtrema::AbstractString,
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

  # Find first element where a timestep would be greater 1 => Special treatment
  IndGreater1 = findfirst(x->real(x) > -1.0, TrueComplex)
  println(IndGreater1)

  if IndGreater1 === nothing
    # Easy: All can use negated inverse root as timestep
    IndGreater1 = NumTrueComplex + 1
  end

  a  = zeros(NumTrueComplex_)
  b1 = zeros(NumTrueComplex_)
  b2 = zeros(NumTrueComplex_)

  for i = 1:IndGreater1 - 1
    a[i]  = -1.0 / real(TrueComplex[i])
    # Ansatz: Use same weight for both b1, b2.
    b1[i] = -real(TrueComplex[i]) / (abs(TrueComplex[i]) .* abs.(TrueComplex[i]))
    b2[i] = b1[i]
  end

  for i = IndGreater1:NumTrueComplex
    # To avoid timesteps > 1: set a[i] to 1.
    a[i]  = 1.0

    #b2[i] = 1.0 / (abs(TrueComplex[i]) * abs(TrueComplex[i]) * a[i])
    b2[i] = 1.0 / (abs(TrueComplex[i]) * abs(TrueComplex[i]))
    
    b1[i] = -2.0 * real(TrueComplex[i]) / (abs(TrueComplex[i]) * abs(TrueComplex[i])) - b2[i]
  end

  println("a:\n"); display(a); println()
  println("b1:\n"); display(b1); println()
  println("b2:\n"); display(b2); println()

  println("b1 + b2 (Timesteps):\n"); display(b1 .+ b2); println()

  return a, b1, b2
end



function read_eta_opt(Stages::Int, BasePath::AbstractString, NumTrueComplex_::Int, TimeStepSort::Vector{Int64})
  Path = BasePath * "eta" * string(Stages) * ".txt"
  eta_opt = readdlm(Path, ',')
  @assert size(eta_opt, 1) == NumTrueComplex_ "Provided eta values do not match stage count!"

  # Add dummy row to eta_opt resembling forward euler
  eta_opt = vcat([42.0 42.0 42.0], eta_opt)

  # Sort with ascending timestep
  eta_opt[:, 1] = permute!(eta_opt[:, 1], TimeStepSort)
  eta_opt[:, 2] = permute!(eta_opt[:, 2], TimeStepSort)
  eta_opt[:, 3] = permute!(eta_opt[:, 3], TimeStepSort)

  return eta_opt
end


function read_ShuOsherCoeffs(BasePath::AbstractString, Stages::Int)
  alpha = readdlm(BasePath * "alpha.txt", ',')
  #display(alpha)
  @assert size(alpha, 1) == Stages - 1 "Provided alpha.txt coefficients do not match stage count!"

  beta = readdlm(BasePath * "beta.txt", ',')
  #display(beta)
  @assert size(beta, 1) == Stages - 1 "Provided beta.txt coefficients do not match stage count!"

  return alpha, beta
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
  TimeStepSort::Vector{Int64}
  IndexForwardEuler::Int

  eta_opt::Matrix{Float64}

  a::Vector{Float64}
  b1::Vector{Float64}
  b2::Vector{Float64}

  alpha::Matrix{Float64}
  beta::Matrix{Float64}

  # Constructor for previously computed A Coeffs
  function FE2S(Stages_::Int, PathPseudoExtrema_::AbstractString)
    if Stages_ % 2 == 0 
      NumTrueComplex_ = Int(Stages_ / 2 - 1)
    else
      NumTrueComplex_ = Int((Stages_ - 1) / 2)
    end
    newFE2S = new(Stages_, NumTrueComplex_)

    newFE2S.alpha, newFE2S.beta = read_ShuOsherCoeffs(PathPseudoExtrema_, Stages_)

    
    newFE2S.ForwardEulerWeight, newFE2S.InvAbsValsSquared, newFE2S.TwoRealOverAbsSquared, 
    newFE2S.TimeSteps, newFE2S.IndexForwardEuler, newFE2S.TimeStepSort = 
      ComputeFE2S_Coefficients(Stages_, PathPseudoExtrema_, newFE2S.NumTrueComplex)

    newFE2S.a, newFE2S.b1, newFE2S.b2 =
      ComputeFE2S_Coefficients_RealRK(Stages_, PathPseudoExtrema_, newFE2S.NumTrueComplex)

    #newFE2S.eta_opt = read_eta_opt(Stages_, PathPseudoExtrema_, newFE2S.NumTrueComplex, newFE2S.TimeStepSort)
    

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
  u_1::uType
  t_stage::RealT
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

  # TODO: Introduce u1, u2, k1, k2!
  du    = similar(u0)
  u_tmp = similar(u0)

  k1 = similar(u0)
  u_1 = similar(u0)

  t0 = first(ode.tspan)
  iter = 0

  integrator = FE2S_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, 
                 ode.p, # the semidiscretization
                 (prob=ode,), # Not really sure whats going on here
                 ode.f, # the right-hand-side of the ODE u' = f(u, p, t)
                 alg, # The ODE integration algorithm/method
                 FE2S_IntegratorOptions(callback, ode.tspan; kwargs...), false, k1, u_1, t0)

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

    @trixi_timeit timer() "Forward Euler Two Stage ODE integration step" begin
      @threaded for j in eachindex(integrator.u)
        integrator.u_tmp[j] = integrator.u[j] # Used for incremental stage update
        # For Shu-Osher form
        integrator.u_1[j] = integrator.u[j] 
      end
      
      
      ### Successive Intermediate Stages implementation ###
      integrator.t_stage = integrator.t

      # Two-stage substeps with smaller timestep than ForwardEuler
      for i = 1:alg.IndexForwardEuler-1
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

        @threaded for j in eachindex(integrator.du)
          #=
          # k = 0
          integrator.k1[j] = integrator.u_tmp[j] * alg.TwoRealOverAbsSquared[i] + 
                            integrator.dt * integrator.du[j] * alg.InvAbsValsSquared[i]
          
          # k = 1
          integrator.k1[j] = integrator.u_tmp[j] * alg.TwoRealOverAbsSquared[i] / sqrt(alg.InvAbsValsSquared[i]) + 
                            integrator.dt * integrator.du[j] * sqrt(alg.InvAbsValsSquared[i])                       
          
          # k = 2
          integrator.k1[j] = integrator.u_tmp[j] * alg.TwoRealOverAbsSquared[i] / alg.InvAbsValsSquared[i] + 
                            integrator.dt * integrator.du[j]   
          
          # Optimized eta
          
          integrator.k1[j] = integrator.u_tmp[j] * alg.eta_opt[i, 1] + 
                            integrator.dt * integrator.du[j] * alg.eta_opt[i, 2]  
          
          =#                    
          # eta_1 = 1 (Somewhat RK like)
          
          integrator.k1[j] = integrator.u_tmp[j] + 
                            integrator.dt * integrator.du[j] / alg.TwoRealOverAbsSquared[i] * alg.InvAbsValsSquared[i]          
          
          
          # "Lebedev-way" (See https://infoscience.epfl.ch/record/182180/files/abd_cheb_springer.pdf Eq. (17).)
          # u_tmp = g_i
          #=
          integrator.u_tmp[j] = integrator.u_tmp[j] + 
                                integrator.dt * integrator.du[j] * 0.5 * alg.TwoRealOverAbsSquared[i]
          =#                                
        end
        # For eta_1 = 1 (Somewhat RK like)
        integrator.f(integrator.du, integrator.k1, prob.p, integrator.t_stage + integrator.dt / alg.TwoRealOverAbsSquared[i] * alg.InvAbsValsSquared[i])
        # Instead for Lebedev way:
        
        integrator.t_stage += integrator.dt * 0.5 * alg.TwoRealOverAbsSquared[i]
        #integrator.f(integrator.k1, integrator.u_tmp, prob.p, integrator.t_stage)
        

        @threaded for j in eachindex(integrator.du)
          # k = 0
          #integrator.u_tmp[j] += integrator.dt * integrator.du[j]

          # k = 1
          #integrator.u_tmp[j] += integrator.dt * integrator.du[j] * sqrt(alg.InvAbsValsSquared[i])

          # k = 2
          #integrator.u_tmp[j] += integrator.dt * integrator.du[j] * alg.InvAbsValsSquared[i]

          # Optimized eta
          #integrator.u_tmp[j] += integrator.dt * integrator.du[j] * alg.eta_opt[i, 3]

          # eta_1 = 1 (Somewhat RK like)
          integrator.u_tmp[j] += integrator.dt * integrator.du[j] * alg.TwoRealOverAbsSquared[i]

          # "Lebedev-way" (See https://infoscience.epfl.ch/record/182180/files/abd_cheb_springer.pdf Eq. (17).)
          #=
          integrator.u_tmp[j] += integrator.dt * (0.5 * alg.TwoRealOverAbsSquared[i] * integrator.du[j] + 
                                alg.InvAbsValsSquared[i] / (0.5 * alg.TwoRealOverAbsSquared[i]) * 
                                (integrator.k1[j] - integrator.du[j]))
          =#                              
        end
        # For eta_1 = 1 (Somewhat RK like)
        #integrator.t_stage += integrator.dt * alg.TwoRealOverAbsSquared[i]
        # Instead for Lebedev way:
        #integrator.t_stage += integrator.dt * 0.5 * alg.TwoRealOverAbsSquared[i]
      end

      # Forward Euler step
      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage) # du = k1
      integrator.t_stage += integrator.dt * alg.ForwardEulerWeight

      @threaded for j in eachindex(integrator.du)
        integrator.u_tmp[j] += alg.ForwardEulerWeight * integrator.dt * integrator.du[j]
      end

      # Two-stage substeps with bigger timestep than ForwardEuler
      for i = alg.IndexForwardEuler+1:length(alg.InvAbsValsSquared)
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

        @threaded for j in eachindex(integrator.du)
          #= 
          # k = 0
          integrator.k1[j] = integrator.u_tmp[j] * alg.TwoRealOverAbsSquared[i] + 
                            integrator.dt * integrator.du[j] * alg.InvAbsValsSquared[i]
          
          # k = 1
          integrator.k1[j] = integrator.u_tmp[j] * alg.TwoRealOverAbsSquared[i] / sqrt(alg.InvAbsValsSquared[i]) + 
                            integrator.dt * integrator.du[j] * sqrt(alg.InvAbsValsSquared[i])                       
          
          # k = 2
          integrator.k1[j] = integrator.u_tmp[j] * alg.TwoRealOverAbsSquared[i] / alg.InvAbsValsSquared[i] + 
                            integrator.dt * integrator.du[j]

          
          # Optimized eta             
          integrator.k1[j] = integrator.u_tmp[j] * alg.eta_opt[i, 1] + 
                            integrator.dt * integrator.du[j] * alg.eta_opt[i, 2]  
          
          =#                
          # eta_1 = 1 (Somewhat RK like)
          
          integrator.k1[j] = integrator.u_tmp[j] + 
                            integrator.dt * integrator.du[j] / alg.TwoRealOverAbsSquared[i] * alg.InvAbsValsSquared[i]          
                                 
          
          # "Lebedev-way" (See https://infoscience.epfl.ch/record/182180/files/abd_cheb_springer.pdf Eq. (17).)
          # u_tmp = g_i
          #=
          integrator.u_tmp[j] = integrator.u_tmp[j] + 
                                integrator.dt * integrator.du[j] * 0.5 * alg.TwoRealOverAbsSquared[i]
          =#                                
        end

        integrator.f(integrator.du, integrator.k1, prob.p, integrator.t_stage + integrator.dt / alg.TwoRealOverAbsSquared[i] * alg.InvAbsValsSquared[i])
        # Instead for Lebedev way:
        
        #integrator.t_stage += integrator.dt * 0.5 * alg.TwoRealOverAbsSquared[i]
        #integrator.f(integrator.k1, integrator.u_tmp, prob.p, integrator.t_stage)
        

        @threaded for j in eachindex(integrator.du)                                  
          # k = 0
          #integrator.u_tmp[j] += integrator.dt * integrator.du[j]

          # k = 1
          #integrator.u_tmp[j] += integrator.dt * integrator.du[j] * sqrt(alg.InvAbsValsSquared[i])

          # k = 2
          #integrator.u_tmp[j] += integrator.dt * integrator.du[j] * alg.InvAbsValsSquared[i]

          # Optimized eta
          #integrator.u_tmp[j] += integrator.dt * integrator.du[j] * alg.eta_opt[i, 3]

          # eta_1 = 1 (Somewhat RK like)
          integrator.u_tmp[j] += integrator.dt * integrator.du[j] * alg.TwoRealOverAbsSquared[i]

          # "Lebedev-way" (See https://infoscience.epfl.ch/record/182180/files/abd_cheb_springer.pdf Eq. (17).)
          #=
          integrator.u_tmp[j] += integrator.dt * (0.5 * alg.TwoRealOverAbsSquared[i] * integrator.du[j] + 
                                alg.InvAbsValsSquared[i] / (0.5 * alg.TwoRealOverAbsSquared[i]) * 
                                (integrator.k1[j] - integrator.du[j]))
          =#                            
        end
        # For eta_1 = 1 (Somewhat RK like)
        integrator.t_stage += integrator.dt * alg.TwoRealOverAbsSquared[i]
        # Instead for Lebedev way:
        #integrator.t_stage += integrator.dt * 0.5 * alg.TwoRealOverAbsSquared[i]
      end
      

      #=
      ### Classic RK version ###
      # Forward Euler step
      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t) # du = k1
      
      @threaded for j in eachindex(integrator.du)
        integrator.u_tmp[j] += alg.ForwardEulerWeight * integrator.dt * integrator.du[j]
      end
      integrator.t_stage = integrator.t + integrator.dt * alg.ForwardEulerWeight

      # Intermediate "two-step" sub methods
      for i in eachindex(alg.a)
        # Low-storage implementation (only one k = du):
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)
        #integrator.t_stage = integrator.t - integrator.dt * (alg.b1[i-1] + alg.b2[i-1])
  
        # First RK update
        @threaded for j in eachindex(integrator.du)
          integrator.k1[j]     = integrator.u_tmp[j] + integrator.dt * alg.a[i] * integrator.du[j]
          integrator.u_tmp[j] += integrator.dt * alg.b1[i] * integrator.du[j]
        end
        
        integrator.f(integrator.du, integrator.k1, prob.p, integrator.t_stage + integrator.dt * alg.a[i])

        # Second RK update
        @threaded for j in eachindex(integrator.du)
          integrator.u_tmp[j] += integrator.dt .* alg.b2[i] .* integrator.du[j]
        end
        integrator.t_stage += integrator.dt * (alg.b1[i] + alg.b2[i])
      end
      =#

      #=
      ### Shu-Osher Form with two substages ###
      # NOTE: No efficient implementation at this stage!
      for i = 1:alg.Stages - 1
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)
        integrator.f(integrator.k1, integrator.u_1, prob.p, integrator.t_stage)

        @threaded for j in eachindex(integrator.u_tmp)
          integrator.u_tmp[j] = alg.alpha[i, 1] * integrator.u_tmp[j] + alg.alpha[i, 2] * integrator.u_1[j] + 
                                  integrator.dt * (alg.beta[i, 1] * integrator.du[j] + 
                                                   alg.beta[i, 2] * integrator.k1[j])
        end

        # Switch u_tmp & u_1
        @threaded for j in eachindex(integrator.u_tmp)
          integrator.k1[j] = integrator.u_1[j]
          integrator.u_1[j] = integrator.u_tmp[j]
          integrator.u_tmp[j] = integrator.k1[j]
        end
      end

      # Switch back for last step
      @threaded for j in eachindex(integrator.u_tmp)
        integrator.u_tmp[j] = integrator.u_1[j]
      end
      =#

      # Final Euler step with step length of dt (Due to form of stability polynomial)
      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)
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
