# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
using DelimitedFiles: readdlm
using LinearAlgebra: eigvals

@muladd begin
#! format: noindent

function compute_PERK2_b_embedded_coeffs(num_stage_evals, num_stages,
                                         embedded_monomial_coeffs, a_unknown, c)
    b_embedded = zeros(num_stage_evals - 1)

    # Solve for b_embedded in a matrix-free manner, using a loop-based serial approach
    # We go in reverse order since we have to solve b_embedded last entry from the highest degree first.
    for i in (num_stage_evals - 1):-1:3 # i here represents the degree of stability polynomial we are going through
        # Initialize b_embedded[i]
        b_embedded[i] = embedded_monomial_coeffs[i - 2]

        # Subtract the contributions of the upper triangular part
        for j in (i + 1):(num_stage_evals - 1)
            # Compute the equivalent of A[i, j] without creating the matrix
            aij = c[num_stages - num_stage_evals + j - 1]
            for k in 1:(i - 2) # This loops become inactive for i = 1 and i = 2 since there is no a_unknown contribution there.
                aij *= a_unknown[j - k] # i-2 times multiplications of a_unknown. The first one is already accounted for by c-coeff.
            end

            # Update b_embedded[i] with the computed value
            b_embedded[i] -= aij * b_embedded[j]
        end

        # Retrieve the value of b_embedded by dividing all the a_unknown and c values associated with it
        b_embedded[i] /= c[num_stages - num_stage_evals + i - 1]
        for k in 1:(i - 2)
            b_embedded[i] /= a_unknown[i - k]
        end
    end

    # The second order constraint or i = 2
    b_embedded[2] = 1 / 2
    for j in 3:(num_stage_evals - 1)
        b_embedded[2] -= c[num_stages - num_stage_evals + j] * b_embedded[j]
    end
    b_embedded[2] /= c[num_stages - num_stage_evals + 2]

    b_embedded[1] = 1
    # The first order constraint or i = 1
    for j in 2:(num_stage_evals - 1)
        b_embedded[1] -= b_embedded[j]
    end

    return b_embedded
end

# Compute the Butcher tableau for a paired explicit Runge-Kutta method order 2
# using a list of eigenvalues
function compute_EmbeddedPairedExplicitRK2_butcher_tableau(num_stages, eig_vals, tspan,
                                                           bS, cS; verbose = false)
    # c Vector from Butcher Tableau (defines timestep per stage)
    c = zeros(num_stages)
    for k in 2:num_stages
        c[k] = cS * (k - 1) / (num_stages - 1)
    end
    stage_scaling_factors = bS * reverse(c[2:(end - 1)])

    # - 2 Since first entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    coeffs_max = num_stages - 2

    a_matrix = zeros(coeffs_max, 2)
    a_matrix[:, 1] = c[3:end]

    consistency_order = 2
    b_embedded = zeros(num_stages - 1)

    dtmax = tspan[2] - tspan[1]
    dteps = 1e-9 # Hyperparameter of the optimization, might be too large for systems requiring very small timesteps

    num_eig_vals, eig_vals = filter_eig_vals(eig_vals; verbose)

    monomial_coeffs, dt_opt = bisect_stability_polynomial(consistency_order,
                                                          num_eig_vals, num_stages,
                                                          dtmax,
                                                          dteps,
                                                          eig_vals; verbose)

    monomial_coeffs = undo_normalization!(monomial_coeffs, consistency_order,
                                          num_stages)

    monomial_coeffs_embedded, dt_opt_embedded = bisect_stability_polynomial(consistency_order -
                                                                            1,
                                                                            num_eig_vals,
                                                                            num_stages -
                                                                            1,
                                                                            dtmax,
                                                                            dteps,
                                                                            eig_vals;
                                                                            verbose)
    monomial_coeffs_embedded = undo_normalization!(monomial_coeffs_embedded,
                                                   consistency_order - 1,
                                                   num_stages - 1)

    num_monomial_coeffs = length(monomial_coeffs)
    @assert num_monomial_coeffs == coeffs_max
    A = compute_a_coeffs(num_stages, stage_scaling_factors, monomial_coeffs)

    # TODO: Write an updated version of this since there is not that many constraint for this
    # Compute b_embedded
    b_embedded = compute_PERK2_b_embedded_coeffs(num_stages, num_stages,
                                                 monomial_coeffs_embedded, A, c) # THIS CANNOT BE REUSED FOR 2ND ORDER STUFF

    b_full = construct_b_vector(b_embedded, num_stages - 1, num_stages - 1)

    println("mon_embedded ", monomial_coeffs_embedded)

    println("Sum of b_full: ", sum(b_full))
    println("Dot product of b_full and c: ", dot(b_full, c))

    # Calculate and print the percentage of dt_opt_embedded / dt_opt
    percentage_dt_opt = (dt_opt_embedded / dt_opt) * 100
    println("Percentage of dt_opt_embedded / dt_opt: ", percentage_dt_opt, "%")

    a_matrix[:, 1] -= A
    a_matrix[:, 2] = A

    return a_matrix, b_embedded, c, dt_opt
end

# Compute the Butcher tableau for a paired explicit Runge-Kutta method order 2
# using provided monomial coefficients file
function compute_EmbeddedPairedExplicitRK2_butcher_tableau(num_stages,
                                                           base_path_monomial_coeffs::AbstractString,
                                                           bS, cS)
    # c Vector form Butcher Tableau (defines timestep per stage)
    c = zeros(num_stages)
    for k in 2:num_stages
        c[k] = cS * (k - 1) / (num_stages - 1)
    end
    stage_scaling_factors = bS * reverse(c[2:(end - 1)])

    # - 2 Since first entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    coeffs_max = num_stages - 2

    a_matrix = zeros(coeffs_max, 2)
    a_matrix[:, 1] = c[3:end]

    path_monomial_coeffs = joinpath(base_path_monomial_coeffs,
                                    "gamma_" * string(num_stages) * ".txt")

    @assert isfile(path_monomial_coeffs) "Couldn't find file"
    monomial_coeffs = readdlm(path_monomial_coeffs, Float64)
    num_monomial_coeffs = size(monomial_coeffs, 1)

    @assert num_monomial_coeffs == coeffs_max
    A = compute_a_coeffs(num_stages, stage_scaling_factors, monomial_coeffs)

    a_matrix[:, 1] -= A
    a_matrix[:, 2] = A

    return a_matrix, c
end

@doc raw"""
    EmbeddedPairedExplicitRK2(num_stages, base_path_monomial_coeffs::AbstractString, dt_opt,
                      bS = 1.0, cS = 0.5)
    EmbeddedPairedExplicitRK2(num_stages, tspan, semi::AbstractSemidiscretization;
                      verbose = false, bS = 1.0, cS = 0.5)
    EmbeddedPairedExplicitRK2(num_stages, tspan, eig_vals::Vector{ComplexF64};
                      verbose = false, bS = 1.0, cS = 0.5)
    Parameters:
    - `num_stages` (`Int`): Number of stages in the PERK method.
    - `base_path_monomial_coeffs` (`AbstractString`): Path to a file containing 
      monomial coefficients of the stability polynomial of PERK method.
      The coefficients should be stored in a text file at `joinpath(base_path_monomial_coeffs, "gamma_$(num_stages).txt")` and separated by line breaks.
    - `dt_opt` (`Float64`): Optimal time step size for the simulation setup.
    - `tspan`: Time span of the simulation.
    - `semi` (`AbstractSemidiscretization`): Semidiscretization setup.
    -  `eig_vals` (`Vector{ComplexF64}`): Eigenvalues of the Jacobian of the right-hand side (rhs) of the ODEProblem after the
      equation has been semidiscretized.
    - `verbose` (`Bool`, optional): Verbosity flag, default is false.
    - `bS` (`Float64`, optional): Value of b in the Butcher tableau at b_s, when 
      s is the number of stages, default is 1.0.
    - `cS` (`Float64`, optional): Value of c in the Butcher tableau at c_s, when
      s is the number of stages, default is 0.5.

The following structures and methods provide a minimal implementation of
the second-order paired explicit Runge-Kutta (PERK) method
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).

- Brian Vermeire (2019).
  Paired explicit Runge-Kutta schemes for stiff systems of equations
  [DOI: 10.1016/j.jcp.2019.05.014](https://doi.org/10.1016/j.jcp.2019.05.014)
"""
mutable struct EmbeddedPairedExplicitRK2 <: AbstractPairedExplicitRKSingle
    const num_stages::Int

    a_matrix::Matrix{Float64}
    b_embedded::Vector{Float64}
    c::Vector{Float64}
    b1::Float64
    bS::Float64
    cS::Float64
    dt_opt::Float64
end # struct EmbeddedPairedExplicitRK2

# Constructor that reads the coefficients from a file
function EmbeddedPairedExplicitRK2(num_stages,
                                   base_path_monomial_coeffs::AbstractString,
                                   dt_opt,
                                   bS = 1.0, cS = 0.5)
    # If the user has the monomial coefficients, they also must have the optimal time step
    a_matrix, c = compute_EmbeddedPairedExplicitRK2_butcher_tableau(num_stages,
                                                                    base_path_monomial_coeffs,
                                                                    bS, cS)

    return EmbeddedPairedExplicitRK2(num_stages, a_matrix, c, 1 - bS, bS, cS, dt_opt)
end

# Constructor that calculates the coefficients with polynomial optimizer from a
# semidiscretization
function EmbeddedPairedExplicitRK2(num_stages, tspan, semi::AbstractSemidiscretization;
                                   verbose = false,
                                   bS = 1.0, cS = 0.5)
    eig_vals = eigvals(jacobian_ad_forward(semi))

    return EmbeddedPairedExplicitRK2(num_stages, tspan, eig_vals; verbose, bS, cS)
end

# Constructor that calculates the coefficients with polynomial optimizer from a
# list of eigenvalues
function EmbeddedPairedExplicitRK2(num_stages, tspan, eig_vals::Vector{ComplexF64};
                                   verbose = false,
                                   bS = 1.0, cS = 0.5)
    a_matrix, b_embedded, c, dt_opt = compute_EmbeddedPairedExplicitRK2_butcher_tableau(num_stages,
                                                                                        eig_vals,
                                                                                        tspan,
                                                                                        bS,
                                                                                        cS;
                                                                                        verbose)
    println("b_embedded: ", b_embedded)

    return EmbeddedPairedExplicitRK2(num_stages, a_matrix, b_embedded, c, 1 - bS, bS,
                                     cS, dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct EmbeddedPairedExplicitRK2Integrator{RealT <: Real, uType, Params, Sol, F,
                                                   Alg,
                                                   PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSingleIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    tdir::RealT
    dt::RealT # current time step
    dtcache::RealT # manually set time step
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::Alg # This is our own class written above; Abbreviation for ALGorithm
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # EmbeddedPairedExplicitRK2 stages:
    k1::uType
    k_higher::uType
    # Extra register for saving u
    u_old::uType
    # Storage for embedded method's u
    u_e::uType
    nreject::Int # This is probably what we want to know as a criteria of choosing stepsize controllers -> Should actually be in .stats like OrDiffEq
end

function init(ode::ODEProblem, alg::EmbeddedPairedExplicitRK2;
              dt, callback::Union{CallbackSet, Nothing} = nothing, controller, abstol,
              reltol, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)
    u_e = zero(u0)

    # EmbeddedPairedExplicitRK2 stages
    k1 = zero(u0)
    k_higher = zero(u0)
    # Required for embedded, i.e., populated PERK method
    u_old = zero(u0)

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    integrator = EmbeddedPairedExplicitRK2Integrator(u0, du, u_tmp, t0, tdir, dt, dt,
                                                     0.0, iter,
                                                     ode.p,
                                                     (prob = ode,), ode.f, alg,
                                                     EmbeddedPairedExplicitRKOptions(callback,
                                                                                     ode.tspan,
                                                                                     controller,
                                                                                     abstol,
                                                                                     reltol,
                                                                                     ;
                                                                                     kwargs...),
                                                     false, true, EEst, 1.0, false,
                                                     k1, k_higher,
                                                     u_old, u_e, 0)
    # initialize callbacks
    if callback isa CallbackSet
        for cb in callback.continuous_callbacks
            throw(ArgumentError("Continuous callbacks are unsupported with paired explicit Runge-Kutta methods."))
        end
        for cb in callback.discrete_callbacks
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::EmbeddedPairedExplicitRK2;
               dt, callback = nothing, controller, abstol = 1e-4, reltol = 1e-4,
               kwargs...)
    integrator = init(ode, alg, dt = alg.dt_opt, callback = callback,
                      controller = controller, abstol = abstol, reltol = reltol;
                      kwargs...)

    # Start actual solve
    solve!(integrator)
end

function solve!(integrator::EmbeddedPairedExplicitRK2Integrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        step!(integrator)
    end # "main loop" timer

    println("Stats: ", integrator.stats) # TODO: Do we want this to be showed the way naccept is showed? This means we have to do sth with alive.jl
    # that can be generalized to all the other integrators

    rhs_eval = integrator.stats.naccept + integrator.stats.nreject

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

function step!(integrator::EmbeddedPairedExplicitRK2Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    modify_dt_for_tstops!(integrator)

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        # Save the value of the current u right now in u_old
        @threaded for i in eachindex(integrator.du)
            integrator.u_old[i] = integrator.u[i]
        end

        # For the case of num_stages = num_stage_evals, the vector b_embedded will be full
        # k1
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        @threaded for i in eachindex(integrator.du)
            integrator.k1[i] = integrator.du[i] * integrator.dt
            # Add the contribution of the first stage (b_1 in general non-zero)
            integrator.u_e[i] = integrator.u_old[i] +
                                alg.b_embedded[1] * integrator.k1[i]
        end

        # Construct current state
        @threaded for i in eachindex(integrator.u)
            integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
        end
        # k2
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[2] * integrator.dt)

        @threaded for i in eachindex(integrator.du)
            integrator.k_higher[i] = integrator.du[i] * integrator.dt
            integrator.u_e[i] = integrator.u_old[i] +
                                alg.b_embedded[1] * integrator.k_higher[i]
        end

        # Higher stages
        for stage in 3:(alg.num_stages)
            # Construct current state
            @threaded for i in eachindex(integrator.u)
                integrator.u_tmp[i] = integrator.u[i] +
                                      alg.a_matrix[stage - 2, 1] *
                                      integrator.k1[i] +
                                      alg.a_matrix[stage - 2, 2] *
                                      integrator.k_higher[i]
            end

            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)

            @threaded for i in eachindex(integrator.du)
                integrator.k_higher[i] = integrator.du[i] * integrator.dt
                integrator.u_e[i] = integrator.u_old[i] +
                                    alg.b_embedded[1] * integrator.k_higher[i]
            end
        end

        @threaded for i in eachindex(integrator.u)
            integrator.u[i] += alg.b1 * integrator.k1[i] +
                               alg.bS * integrator.k_higher[i]
        end
    end # EmbeddedPairedExplicitRK2 step

    # Compute the estimated local truncation error
    integrator.EEst = norm((integrator.u - integrator.u_e) ./
                           (integrator.opts.abstol .+
                            integrator.opts.reltol .*
                            max.(abs.(integrator.u),
                                 abs.(integrator.u_e))), 2) # Use this norm according to PID controller from OrdinaryDiffEq.jl

    dt_factor = stepsize_controller!(integrator, controller, alg) # Then no need for dt_factor then! Since q_old is already set to dt_factor

    if accept_step_controller(integrator, controller)
        integrator.t += integrator.dt # The logic and the function to increment the accepted time step has to be called here.
        integrator.iter += 1
        dt_new = step_accept_controller!(integrator, controller, alg, dt_factor)
        set_proposed_dt!(integrator, dt_new)

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
    else
        @threaded for i in eachindex(integrator.du) # Reset u from step! to u_old
            integrator.u[i] = integrator.u_old[i]
        end
        integrator.nreject += 1 # Increment nreject
        step_reject_controller!(integrator, controller, alg, integrator.u_old)
    end
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::EmbeddedPairedExplicitRK2Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)
end
end # @muladd
