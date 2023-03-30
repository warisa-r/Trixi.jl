using DelimitedFiles

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

function Process_PE(Stages::Int, PathPseudoExtrema::AbstractString,
                    NumTrueComplex_::Int)

  PathPureReal = PathPseudoExtrema * "PureReal" * string(Stages) * ".txt"
  NumPureReal, PureReal = read_file(PathPureReal, Float64)

  @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
  ForwardEulerWeight = -1.0 / PureReal[1]

  PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(Stages) * ".txt"
  NumTrueComplex, TrueComplex = read_file(PathTrueComplex, ComplexF64)
  @assert NumTrueComplex == NumTrueComplex_ "Assume that all but one pseudo-extremum are complex"

  # Sort ascending => ascending timesteps (real part is always negative)
  perm = sortperm(real.(TrueComplex))
  TrueComplex = TrueComplex[perm]

  InvAbsValsSquared     = 1.0./abs.(TrueComplex.^2)
  TwoRealOverAbsSquared = -2.0 * real.(TrueComplex) .* InvAbsValsSquared

  return TrueComplex, ForwardEulerWeight, InvAbsValsSquared, TwoRealOverAbsSquared           
end
  

function ComputeTimeSteps(Stages::Int, NumTrueComplex::Int, 
                          ForwardEulerWeight::Float64, alpha::Matrix{Float64}, beta::Matrix{Float64})
  # Timestep computation: c = (I - alpha)^(-1) * beta * Vector{1}

  # Compute (I - alpha)^(-1) = sum_{k=0}^{S-1} alpha^k 
  InvIMinusAlpha = Matrix(1.0I, Stages, Stages)

  alphaMat = zeros(Stages, Stages) # = alpha_{1:S}
  betaMat  = zeros(Stages, Stages)

  # Include Forward Euler at first position
  alphaMat[2, 1] = 1.0
  betaMat[2, 1]  = ForwardEulerWeight

  for i = 3:Stages
    alphaMat[i, i-2] = alpha[i-1, 1]
    alphaMat[i, i-1] = alpha[i-1, 2]

    betaMat[i, i-2] = beta[i-1, 1]
    betaMat[i, i-1] = beta[i-1, 2]
  end

  alphaPower = alphaMat
  for i = 1:Stages-1
    InvIMinusAlpha += alphaPower
    alphaPower *= alphaMat
  end
  c = InvIMinusAlpha * betaMat * ones(Stages)
  #display(c)

  c_combined = zeros(NumTrueComplex + 1)
  c_combined[1] = c[2]
  for i = 2:NumTrueComplex + 1
    c_combined[i] = c[2*i - 1] + c[2*i]
  end
  #display(c_combined)

  # TODO: Implement sorting of two stage steps based on combined timesteps!

  return c
end

function ComputeTimeSteps(Stages::Int, NumTrueComplex::Int, 
                          alpha::Matrix{Float64}, beta::Matrix{Float64})
  # Timestep computation: c = (I - alpha)^(-1) * beta * Vector{1}

  # Compute (I - alpha)^(-1) = sum_{k=0}^{S-1} alpha^k 
  InvIMinusAlpha = Matrix(1.0I, Stages, Stages)

  alphaMat = zeros(Stages, Stages) # = alpha_{1:S}
  betaMat  = zeros(Stages, Stages)

  # Include Forward Euler at first position
  alphaMat[2, 1] = 1.0
  betaMat[2, 1]  = beta[1, 2]

  for i = 3:Stages
    alphaMat[i, i-2] = alpha[i-1, 1]
    alphaMat[i, i-1] = alpha[i-1, 2]

    betaMat[i, i-2] = beta[i-1, 1]
    betaMat[i, i-1] = beta[i-1, 2]
  end

  alphaPower = alphaMat
  for i = 1:Stages-1
    InvIMinusAlpha += alphaPower
    alphaPower *= alphaMat
  end
  c = InvIMinusAlpha * betaMat * ones(Stages)
  display(c)

  c_combined = zeros(NumTrueComplex + 1)
  c_combined[1] = c[2]
  for i = 2:NumTrueComplex + 1
    c_combined[i] = c[2*i - 1] + c[2*i]
  end
  #display(c_combined)

  # TODO: Implement sorting of two stage steps based on combined timesteps!

  return c
end


function FE2S_Coeffs_CaseDep(Stages::Int, NumTrueComplex::Int, 
                             TrueComplex::Vector{<:ComplexF64},
                             ForwardEulerWeight::Float64,
                             InvAbsValsSquared::Vector{Float64},
                             TwoRealOverAbsSquared::Vector{Float64})

  # Find first element where a timestep would be greater 1 => Special treatment
  IndGreater1 = findfirst(x->real(x) > -1.0, TrueComplex)

  if IndGreater1 === nothing
    # Easy: No "troubled roots" leading to Euler steps with size > 1
    IndGreater1 = NumTrueComplex + 1
  end

  # Parameters of Shu-Osher form
  alpha = zeros(Stages-1, 2) # Weights in combination
  beta  = zeros(Stages-1, 2) # Timesteps

  # Forward Euler step
  alpha[1, 2] = 1.0
  beta[1, 2]  = ForwardEulerWeight

  # Two-stage submethods
  for i = 1:IndGreater1 - 1
    # First substage
    alpha[2*i, 2] = 1.0
    beta[2*i, 2]  = -1.0 / real(TrueComplex[i])

    # Second substage
    alpha[2*i+1, 1] = 1.0
    # Ansatz: Use same weight for both beta_{j+1, j-1} = beta_{j+1, j}
    beta[2*i+1, 1] = -real(TrueComplex[i]) / (abs(TrueComplex[i]) .* abs.(TrueComplex[i]))
    beta[2*i+1, 2] = beta[2*i+1, 1]
  end

  for i = IndGreater1:NumTrueComplex
    # First substage
    alpha[2*i, 2] = 1.0
    # To avoid timesteps > 1: set beta_{j, j-1} to 1.
    beta[2*i, 2]  = 1.0

    # Second Substage
    alpha[2*i+1, 1] = 1.0

    beta[2*i+1, 1] = TwoRealOverAbsSquared[i] - InvAbsValsSquared[i]
    beta[2*i+1, 2] = InvAbsValsSquared[i]
  end

  c = ComputeTimeSteps(Stages, NumTrueComplex, ForwardEulerWeight, alpha, beta)

  return alpha, beta, c
end

function FE2S_Coeffs_Consecutive(Stages::Int, NumTrueComplex::Int, 
                                 TrueComplex::Vector{<:ComplexF64},
                                 ForwardEulerWeight::Float64,
                                 InvAbsValsSquared::Vector{Float64},
                                 TwoRealOverAbsSquared::Vector{Float64})

  # Parameters of Shu-Osher form
  alpha = zeros(Stages-1, 2) # Weights in combination
  beta  = zeros(Stages-1, 2) # Timesteps

  # Forward Euler step
  alpha[1, 2] = 1.0
  beta[1, 2]  = ForwardEulerWeight

  # Two-stage submethods
  for i = 1:NumTrueComplex
    # First substage
    alpha[2*i, 2] = 1.0
    beta[2*i, 2]  = -1.0 / (2.0 * real(TrueComplex[i]))

    # Second substage
    alpha[2*i+1, 1] = 1.0

    beta[2*i+1, 1]  = 0.0
    beta[2*i+1, 2]  = TwoRealOverAbsSquared[i]
  end

  c = ComputeTimeSteps(Stages, NumTrueComplex, ForwardEulerWeight, alpha, beta)

  return alpha, beta, c
end

function FE2S_Coeffs_NegBeta(Stages::Int, NumTrueComplex::Int, 
                             TrueComplex::Vector{<:ComplexF64},
                             ForwardEulerWeight::Float64,
                             InvAbsValsSquared::Vector{Float64},
                             TwoRealOverAbsSquared::Vector{Float64})

  # Parameters of Shu-Osher form
  alpha = zeros(Stages-1, 2) # Weights in combination
  beta  = zeros(Stages-1, 2) # Timesteps

  # Forward Euler step
  alpha[1, 2] = 1.0
  beta[1, 2]  = ForwardEulerWeight

  # Two-stage submethods
  for i = 1:NumTrueComplex
    # First substage
    alpha[2*i, 2] = 1.0
    beta[2*i, 2]  = -1.0 / (2.0 * real(TrueComplex[i]))

    # Second substage
    alpha[2*i+1, 2] = 1.0

    beta[2*i+1, 1]  = 0.5 * TwoRealOverAbsSquared[i] + 1.0 / (real(TrueComplex[i]))
    beta[2*i+1, 2]  = -1.0 / (real(TrueComplex[i]))
  end

  c = ComputeTimeSteps(Stages, NumTrueComplex, ForwardEulerWeight, alpha, beta)

  return alpha, beta, c                                 
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

# The maximum internal amplification factor as defined in (2.18) in https://doi.org/10.1137/130936245
function MaxInternalAmpFactor(Stages::Int, alpha::Matrix{Float64}, beta::Matrix{Float64}, EigValsScaled::Vector{<:Complex})
  M = 0.0

  # Parameters of last Forward Euler step with weight 1
  alpha_Splus    = zeros(1, Stages)
  alpha_Splus[1] = 1.0

  beta_Splus = zeros(ComplexF64, 1, Stages)

  alphaMat = zeros(Stages, Stages) # = alpha_{1:S}
  betaMat  = zeros(ComplexF64, Stages, Stages)
  betaMatEV = similar(betaMat)

  # CARE: Assumes that Forward Euler step is always executed first!
  # Include Forward Euler at first position
  alphaMat[2, 1] = 1.0
  betaMat[2, 1]  = beta[1, 2]

  for i = 3:Stages
    alphaMat[i, i-2] = alpha[i-1, 1]
    alphaMat[i, i-1] = alpha[i-1, 2]

    betaMat[i, i-2] = beta[i-1, 1]
    betaMat[i, i-1] = beta[i-1, 2]
  end

  for i in eachindex(EigValsScaled)
    beta_Splus[Stages] = EigValsScaled[i]

    # Compute (I - alpha_1:S - z * beta_1:S)^(-1)
    Inv = I # corresponds to i = 0

    # Multiply beta with eigenvalue
    betaMatEV = betaMat * EigValsScaled[i]

    Power = alphaMat + betaMatEV
    for i = 1:Stages-1
      Inv += Power
      Power *= alphaMat + betaMatEV
    end

    # Compute maximum of Q_j of this eigenvalue
    QMax = maximum(abs.((alpha_Splus + beta_Splus) * Inv))

    if QMax > M
      M = QMax
    end
  end

  return M
end

# More precise variant of the maximum internal aplification factor from https://doi.org/10.1137/130936245
function InternalAmpFactor(Stages::Int, alpha::Matrix{Float64}, beta::Matrix{Float64}, EigValsScaled::Vector{<:Complex})
  M = 0.0

  # Parameters of last Forward Euler step with weight 1
  alpha_Splus    = zeros(1, Stages)
  alpha_Splus[1] = 1.0

  beta_Splus = zeros(ComplexF64, 1, Stages)

  alphaMat = zeros(Stages, Stages) # = alpha_{1:S}
  betaMat  = zeros(ComplexF64, Stages, Stages)
  betaMatEV = similar(betaMat)

  # CARE: Assumes that Forward Euler step is always executed first!
  # Include Forward Euler at first position
  alphaMat[2, 1] = 1.0
  betaMat[2, 1]  = beta[1, 2]

  for i = 3:Stages
    alphaMat[i, i-2] = alpha[i-1, 1]
    alphaMat[i, i-1] = alpha[i-1, 2]

    betaMat[i, i-2] = beta[i-1, 1]
    betaMat[i, i-1] = beta[i-1, 2]
  end

  for i in eachindex(EigValsScaled)
    beta_Splus[Stages] = EigValsScaled[i]

    # Compute (I - alpha_1:S - z * beta_1:S)^(-1)
    Inv = I # corresponds to i = 0

    # Multiply beta with eigenvalue
    betaMatEV = betaMat * EigValsScaled[i]

    Power = alphaMat + betaMatEV
    for i = 1:Stages-1
      Inv += Power
      Power *= alphaMat + betaMatEV
    end

    # Compute sum of Q_j of this eigenvalue
    QSum = sum(abs.((alpha_Splus + beta_Splus) * Inv))

    if QSum > M
      M = QSum
    end
  end

  return M
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

  alpha::Matrix{Float64}
  beta::Matrix{Float64}
  c::Vector{Float64}

  IndexForwardEuler::Int

  # Constructor for previously computed A Coeffs
  function FE2S(Stages_::Int, PathPseudoExtrema_::AbstractString)

    if Stages_ % 2 == 0 
      NumTrueComplex_ = Int(Stages_ / 2 - 1)
    else
      NumTrueComplex_ = Int((Stages_ - 1) / 2)
    end

    newFE2S = new(Stages_, NumTrueComplex_)

    newFE2S.alpha, newFE2S.beta = read_ShuOsherCoeffs(PathPseudoExtrema_, Stages_)
    newFE2S.c = ComputeTimeSteps(Stages_, NumTrueComplex_, newFE2S.alpha, newFE2S.beta)
    # TODO: Compute Timesteps if Forward Euler is already included!

    #=
    TrueComplex_, newFE2S.ForwardEulerWeight, newFE2S.InvAbsValsSquared, newFE2S.TwoRealOverAbsSquared = 
      Process_PE(Stages_, PathPseudoExtrema_, newFE2S.NumTrueComplex)

    
    newFE2S.alpha, newFE2S.beta, newFE2S.c = FE2S_Coeffs_CaseDep(Stages_, NumTrueComplex_, TrueComplex_, 
                                              newFE2S.ForwardEulerWeight, 
                                              newFE2S.InvAbsValsSquared, newFE2S.TwoRealOverAbsSquared)
    
                                                  
    newFE2S.alpha, newFE2S.beta, newFE2S.c = FE2S_Coeffs_NegBeta(Stages_, NumTrueComplex_, TrueComplex_, 
                                              newFE2S.ForwardEulerWeight, 
                                              newFE2S.InvAbsValsSquared, newFE2S.TwoRealOverAbsSquared)                                              
    
    
    newFE2S.alpha, newFE2S.beta, newFE2S.c = FE2S_Coeffs_Consecutive(Stages_, NumTrueComplex_, TrueComplex_, 
                                              newFE2S.ForwardEulerWeight, 
                                              newFE2S.InvAbsValsSquared, newFE2S.TwoRealOverAbsSquared)
    =#
    #display(newFE2S.alpha)
    #display(newFE2S.beta)

    return newFE2S
  end
end # struct FE2S

function StabPolyFE2S(NumTrueComplex::Int, z::Complex, alpha::Matrix{Float64}, beta::Matrix{Float64})

  # Forward Euler step
  P = 1 + z * beta[1, 2]

  # Two-stage submethods
  for i = 1:NumTrueComplex
    k1 = 1 + z * beta[2*i, 2]
    P *= alpha[2*i + 1, 1] + alpha[2*i + 1, 2] * k1 + beta[2*i + 1, 1] * z + beta[2*i + 1, 2] * z * k1
  end

  # Final Forward Euler step
  P = 1 + z * P

  return P
end

# Examine stability of certain parametrization - not really useful
function MaxTimeStep(dtMax::Float64, EigVals::Vector{<:ComplexF64}, alg::FE2S)
  dtEps = 1e-9
  dt    = -1.0
  dtMin = 0.0

  while dtMax - dtMin > dtEps
    dt = 0.5 * (dtMax + dtMin)

    AbsPMax = 0.0
    for i in eachindex(EigVals)
      AbsP = abs(StabPolyFE2S(alg.NumTrueComplex, EigVals[i] * dt, alg.alpha, alg.beta))

      if AbsP > AbsPMax
        AbsPMax = AbsP
      end

      if AbsPMax > 1.0
        break
      end
    end

    if AbsPMax < 1.0
      dtMin = dt
    else
      dtMax = dt
    end

    println("Current dt: ", dt)
    println("Current AbsPMax: ", AbsPMax, "\n")
  end

  return dt
end


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

      ### Shu-Osher Form with two substages ###
      # TODO: Correct timestep!
      for i = 1:alg.Stages - 1
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage + integrator.dt * alg.c[i])
        integrator.f(integrator.k1, integrator.u_1, prob.p, integrator.t_stage + integrator.dt * alg.c[i])

        @threaded for j in eachindex(integrator.u_tmp)
          integrator.u_tmp[j] = alg.alpha[i, 1] * integrator.u_tmp[j] + alg.alpha[i, 2] * integrator.u_1[j] + 
                                  integrator.dt * (alg.beta[i, 1] * integrator.du[j] + 
                                                   alg.beta[i, 2] * integrator.k1[j])
        end

        # Switch u_tmp & u_1
        integrator.u_tmp, integrator.u_1 = integrator.u_1, integrator.u_tmp
      end

      # Final Euler step with step length of dt (Due to form of stability polynomial)
      integrator.f(integrator.du, integrator.u_1, prob.p, integrator.t_stage + integrator.dt * alg.c[alg.Stages])
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
