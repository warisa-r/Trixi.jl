# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
using DelimitedFiles: readdlm

@muladd begin
#! format: noindent
function compute_b_embedded_coeffs(num_stage_evals, num_stages,
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

# Some function defined so that I can check if the second order condition is met. This will be removed later.
function construct_b_vector(b_unknown, num_stages_embedded, num_stage_evals_embedded)
    # Construct the b_embedded vector
    b_embedded = [
        b_unknown[1],
        zeros(Float64, num_stages_embedded - num_stage_evals_embedded)...,
        b_unknown[2:end]...,
        0
    ]
    return b_embedded
end

# Compute the Butcher tableau for a paired explicit Runge-Kutta method order 3
# using a list of eigenvalues
function compute_EmbeddedPairedRK3_butcher_tableau(num_stages, num_stage_evals, tspan,
                                                   eig_vals::Vector{ComplexF64};
                                                   verbose = false, cS2)
    # Initialize array of c
    c = compute_c_coeffs(num_stages, cS2)

    # Initialize the array of our solution
    a_unknown = zeros(num_stage_evals - 2)
    b_embedded = zeros(num_stage_evals - 1)

    # Special case of e = 3
    if num_stage_evals == 3
        a_unknown = [0.25] # Use classic SSPRK33 (Shu-Osher) Butcher Tableau
    else
        # Calculate coefficients of the stability polynomial in monomial form
        consistency_order = 3
        dtmax = tspan[2] - tspan[1]
        dteps = 1.0f-9

        num_eig_vals, eig_vals = filter_eig_vals(eig_vals; verbose)

        monomial_coeffs, dt_opt = bisect_stability_polynomial(consistency_order,
                                                                num_eig_vals,
                                                                num_stage_evals,
                                                                dtmax, dteps,
                                                                eig_vals; verbose)
        monomial_coeffs = undo_normalization!(monomial_coeffs, consistency_order,
                                              num_stage_evals)

        # Solve the nonlinear system of equations from monomial coefficient and
        # Butcher array abscissae c to find Butcher matrix A
        # This function is extended in TrixiNLsolveExt.jl
        a_unknown = solve_a_butcher_coeffs_unknown!(a_unknown, num_stages,
                                                    num_stage_evals,
                                                    monomial_coeffs, cS2, c;
                                                    verbose)

        monomial_coeffs_embedded, dt_opt_embedded = bisect_stability_polynomial(consistency_order -
                                                                         1,
                                                                         num_eig_vals,
                                                                         num_stage_evals -
                                                                         1,
                                                                         dtmax, dteps,
                                                                         eig_vals;
                                                                         verbose)
        monomial_coeffs_embedded = undo_normalization!(monomial_coeffs_embedded,
                                                       consistency_order - 1,
                                                       num_stage_evals - 1)

        b_embedded = compute_b_embedded_coeffs(num_stage_evals, num_stages,
                                      monomial_coeffs_embedded, a_unknown, c)

        b_full = construct_b_vector(b_embedded, num_stages - 1, num_stage_evals - 1)

        println("Sum of b_full: ", sum(b_full))
        println("Dot product of b_full and c: ", dot(b_full, c))

        # Calculate and print the percentage of dt_opt_embedded / dt_opt
        percentage_dt_opt = (dt_opt_embedded / dt_opt) * 100
        println("Percentage of dt_opt_embedded / dt_opt: ", percentage_dt_opt, "%")
    end

    # Fill A-matrix in P-ERK style
    a_matrix = zeros(num_stage_evals - 2, 2)
    a_matrix[:, 1] = c[(num_stages - num_stage_evals + 3):end]
    a_matrix[:, 1] -= a_unknown
    a_matrix[:, 2] = a_unknown

    return a_matrix, b_embedded, c, dt_opt, dt_opt_embedded
end

# Compute the Butcher tableau for a paired explicit Runge-Kutta method order 3
# using provided values of coefficients a in A-matrix of Butcher tableau
function compute_EmbeddedPairedRK3_butcher_tableau(num_stages, num_stage_evals,
                                                   base_path_coeffs::AbstractString;
                                                   cS2)

    # Initialize array of c
    c = compute_c_coeffs(num_stages, cS2)

    # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    coeffs_max = num_stage_evals - 2

    a_matrix = zeros(coeffs_max, 2)
    a_matrix[:, 1] = c[3:end]

    b_embedded = zeros(coeffs_max)

    path_a_coeffs = joinpath(base_path_coeffs,
                             "a_" * string(num_stages) * "_" * string(num_stage_evals) *
                             ".txt")

    path_b_coeffs = joinpath(base_path_coeffs,
                             "b_" * string(num_stages) * "_" * string(num_stage_evals) *
                             ".txt")

    @assert isfile(path_a_coeffs) "Couldn't find file $path_a_coeffs"
    a_coeffs = readdlm(path_a_coeffs, Float64)
    num_a_coeffs = size(a_coeffs, 1)

    @assert num_a_coeffs == coeffs_max
    # Fill A-matrix in P-ERK style
    a_matrix[:, 1] -= a_coeffs
    a_matrix[:, 2] = a_coeffs

    @assert isfile(path_b_coeffs) "Couldn't find file $path_b_coeffs"
    b_embedded = readdlm(path_b_coeffs, Float64)
    num_b_coeffs = size(b_embedded, 1)
    @assert num_b_coeffs == coeffs_max

    return a_matrix, b_embedded, c
end

@doc raw"""
    EmbeddedPairedRK3(num_stages, base_path_coeffs::AbstractString, dt_opt;
                      cS2 = 1.0f0)
    EmbeddedPairedRK3(num_stages, tspan, semi::AbstractSemidiscretization;
                      verbose = false, cS2 = 1.0f0)
    EmbeddedPairedRK3(num_stages, tspan, eig_vals::Vector{ComplexF64};
                      verbose = false, cS2 = 1.0f0)

    Parameters:
    - `num_stages` (`Int`): Number of stages in the paired explicit Runge-Kutta (P-ERK) method.
    - `base_path_coeffs` (`AbstractString`): Path to a file containing some coefficients in the A-matrix and a file containing 
      in some coefficients in the b_embedded vector of the Butcher tableau of the Runge Kutta method.
      The matrix should be stored in a text file at `joinpath(base_path_a_coeffs, "a_$(num_stages).txt")` and separated by line breaks.
    - `dt_opt` (`Float64`): Optimal time step size for the simulation setup.
    - `tspan`: Time span of the simulation.
    - `semi` (`AbstractSemidiscretization`): Semidiscretization setup.
    -  `eig_vals` (`Vector{ComplexF64}`): Eigenvalues of the Jacobian of the right-hand side (rhs) of the ODEProblem after the
      equation has been semidiscretized.
    - `verbose` (`Bool`, optional): Verbosity flag, default is false.
    - `cS2` (`Float64`, optional): Value of c in the Butcher tableau at c_{s-2}, when
      s is the number of stages, default is 1.0f0.

The following structures and methods provide an implementation of
the third-order paired explicit Runge-Kutta (P-ERK) method
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).
The original paper is
- Nasab, Vermeire (2022)
Third-order Paired Explicit Runge-Kutta schemes for stiff systems of equations
[DOI: 10.1016/j.jcp.2022.111470](https://doi.org/10.1016/j.jcp.2022.111470)
While the changes to SSPRK33 base-scheme are described in 
- Doehring, Schlottke-Lakemper, Gassner, Torrilhon (2024)
Multirate Time-Integration based on Dynamic ODE Partitioning through Adaptively Refined Meshes for Compressible Fluid Dynamics
[DOI: 10.1016/j.jcp.2024.113223](https://doi.org/10.1016/j.jcp.2024.113223)
"""
mutable struct EmbeddedPairedRK3 <: AbstractPairedExplicitRKSingle
    const num_stages::Int # S
    const num_stage_evals::Int # e

    a_matrix::Matrix{Float64}
    b_embedded::Vector{Float64}
    c::Vector{Float64}
    dt_opt::Float64
    dt_opt_embedded::Float64
    abs_tol::Float64 # User-specified absolute tolerance # Should be in integrator.opts instead......... ahhhhhhhhhhhhhhhhhhhhhh
    rel_tol::Float64 # User-specified relative tolerance
end # struct EmbeddedPairedRK3

# Constructor for previously computed A Coeffs
function EmbeddedPairedRK3(num_stages, num_stage_evals,
                           base_path_coeffs::AbstractString, dt_opt, dt_opt_embedded;
                           cS2 = 1.0f0, abs_tol = 1e-4, rel_tol = 1e-4)
    a_matrix, b_embedded, c = compute_EmbeddedPairedRK3_butcher_tableau(num_stages,
                                                               num_stage_evals,
                                                               base_path_coeffs;
                                                               cS2)

    return EmbeddedPairedRK3(num_stages, num_stage_evals, a_matrix, b_embedded, c, dt_opt,
                             dt_opt_embedded, abs_tol, rel_tol)
end

# Constructor that computes Butcher matrix A coefficients from a semidiscretization
function EmbeddedPairedRK3(num_stages, num_stage_evals, tspan,
                           semi::AbstractSemidiscretization;
                           verbose = false, cS2 = 1.0f0, abs_tol = 1e-4, rel_tol = 1e-4)
    eig_vals = eigvals(jacobian_ad_forward(semi))

    return EmbeddedPairedRK3(num_stages, num_stage_evals, tspan, eig_vals; verbose, cS2, abs_tol, rel_tol)
end

# Constructor that calculates the coefficients with polynomial optimizer from a list of eigenvalues
function EmbeddedPairedRK3(num_stages, num_stage_evals, tspan,
                           eig_vals::Vector{ComplexF64};
                           verbose = false, cS2 = 1.0f0, abs_tol = 1e-4, rel_tol = 1e-4)
    a_matrix, b_embedded, c, dt_opt, dt_opt_embedded = compute_EmbeddedPairedRK3_butcher_tableau(num_stages,
                                                                                   num_stage_evals,
                                                                                   tspan,
                                                                                   eig_vals;
                                                                                   verbose,
                                                                                   cS2)
    return EmbeddedPairedRK3(num_stages, num_stage_evals, a_matrix, b_embedded, c, dt_opt,
                             dt_opt_embedded, abs_tol, rel_tol)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct EmbeddedPairedExplicitRKOptions{Callback, TStops}
    callback::Callback # callbacks; used in Trixi
    adaptive::Bool # whether the algorithm is adaptive
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    #qsteady_max::Float64
    #qsteady_min::Float64
    controller::AbstractController # When we find out what kind of controller is the best, make it an option
    abstol::Float64 # User-specified absolute tolerance
    reltol::Float64 # User-specified relative tolerance
    tstops::TStops # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function EmbeddedPairedExplicitRKOptions(callback, tspan , controller, abstol, reltol; maxiters = typemax(Int), kwargs...)
    tstops_internal = BinaryHeap{eltype(tspan)}(FasterForward())
    # We add last(tspan) to make sure that the time integration stops at the end time
    push!(tstops_internal, last(tspan))
    # We add 2 * last(tspan) because add_tstop!(integrator, t) is only called by DiffEqCallbacks.jl if tstops contains a time that is larger than t
    # (https://github.com/SciML/DiffEqCallbacks.jl/blob/025dfe99029bd0f30a2e027582744528eb92cd24/src/iterative_and_periodic.jl#L92)
    push!(tstops_internal, 2 * last(tspan))
    EmbeddedPairedExplicitRKOptions{typeof(callback), typeof(tstops_internal)}(callback,
                                                                       true, Inf,
                                                                       maxiters, controller, abstol, reltol,
                                                                       tstops_internal)
end


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct EmbeddedPairedRK3Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
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
    EEst::Float64 # The estimated local truncation error
    qold::Float64 # Temporary time step factor
    force_stepfail::Bool
    # PairedExplicitRK stages:
    k1::uType
    k_higher::uType
    # Extra register for saving u
    u_old::uType
    # Storage for embedded method's u
    u_e::uType
    nreject::Int # This is probably what we want to know as a criteria of choosing stepsize controllers -> Should actually be in .stats like OrDiffEq
end

function init(ode::ODEProblem, alg::EmbeddedPairedRK3;
              dt, callback::Union{CallbackSet, Nothing} = nothing, controller, abstol, reltol, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)
    u_e = zero(u0)

    # PairedExplicitRK stages
    k1 = zero(u0)
    k_higher = zero(u0)

    # Required for embedded, i.e., populated PERK method
    u_old = zero(u0)

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0
    EEst = 0.0

    integrator = EmbeddedPairedRK3Integrator(u0, du, u_tmp, t0, tdir, dt, dt, iter,
                                             ode.p,
                                             (prob = ode,), ode.f, alg,
                                             EmbeddedPairedExplicitRKOptions(callback,
                                                                     ode.tspan, controller, abstol, reltol,;
                                                                     kwargs...),
                                             false, true, EEst, 1.0, false,
                                             k1, k_higher,
                                             u_old, u_e, 0)

    # initialize callbacks
    if callback isa CallbackSet
        for cb in callback.continuous_callbacks
            error("Continuous callbacks are unsupported with paired explicit Runge-Kutta methods.")
        end
        for cb in callback.discrete_callbacks
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    elseif !isnothing(callback)
        error("unsupported")
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::EmbeddedPairedRK3;
               dt, callback = nothing, controller, abstol = 1e-4, reltol = 1e-4, kwargs...)
    # TODO: If the algorithm to determine the stepsize is error-basesd then we should throw an error when user input StepsizeCallback?
    integrator = init(ode, alg, dt = alg.dt_opt, callback = callback, controller= controller, abstol = abstol, reltol = reltol; kwargs...)

    # Start actual solve
    solve!(integrator)
end

function solve!(integrator::EmbeddedPairedRK3Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    @unpack controller = integrator.opts

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        # Look at https://github.com/SciML/OrdinaryDiffEq.jl/blob/d76335281c540ee5a6d1bd8bb634713e004f62ee/src/integrators/controllers.jl#L174
        # for the original implementation of the adaptive time stepping and time step controller
        # TODO: adapt the logic here https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqCore/src/integrators/integrator_utils.jl#L209
        # Maybe do sth like step! first. Call the controller. Check if the step is accepted. If not, reset the value back to u_old
        t_old = integrator.t
        # Save the value of the current u right now in u_old
        @threaded for i in eachindex(integrator.du)
            integrator.u_old[i] = integrator.u[i]
        end

        step!(integrator)

        dt_factor = stepsize_controller!(integrator, controller, alg) # Then no need for dt_factor then! Since q_old is already set to dt_factor

        if accept_step_controller(integrator, controller)
            dt_new = step_accept_controller!(integrator, controller, alg, dt_factor)
            set_proposed_dt!(integrator, dt_new)
        else
            step_reject_controller!(integrator, controller, alg, integrator.u_old)
        end
    end # "main loop" timer

    println("Number of rejected steps: ", integrator.nreject) # TODO: Maybe this should be include somewhere else. Maybe in .stats

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

function step!(integrator::EmbeddedPairedRK3Integrator)
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

    # TODO: This function should probably be moved to another function called `proposed_dt` or similar
    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        # Set `u_old` to incoming `u`
        #=    
        @threaded for i in eachindex(integrator.du)
            integrator.u_old[i] = integrator.u[i]
        end
        =#

        # k1 is in general required, as we use b_1 to satisfy the first-order cons. cond.
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        @threaded for i in eachindex(integrator.du)
            integrator.k1[i] = integrator.du[i] * integrator.dt
            # Add the contribution of the first stage (b_1 in general non-zero)
            integrator.u_e[i] = integrator.u_old[i] + alg.b_embedded[1] * integrator.k1[i]
        end

        # Higher order stages
        for stage in 2:(alg.num_stages - alg.num_stage_evals + 1)
            # Construct current state
            @threaded for i in eachindex(integrator.du)
                integrator.u_tmp[i] = integrator.u_old[i] +
                                      alg.c[stage] * integrator.k1[i]
            end

            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)

            @threaded for i in eachindex(integrator.du)
                integrator.k_higher[i] = integrator.du[i] * integrator.dt
            end
        end

        # This is the first stage after stage 1 for which we need to evaluate `k_higher`
        stage = alg.num_stages - alg.num_stage_evals + 2
        # Construct current state
        @threaded for i in eachindex(integrator.du)
            integrator.u_tmp[i] = integrator.u_old[i] + alg.c[stage] * integrator.k1[i]
        end

        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[stage] * integrator.dt)

        @threaded for i in eachindex(integrator.du)
            integrator.k_higher[i] = integrator.du[i] * integrator.dt
            integrator.u_e[i] += alg.b_embedded[2] * integrator.k_higher[i]
        end

        # Non-reducible stages
        for stage in (alg.num_stages - alg.num_stage_evals + 3):(alg.num_stages - 1)
            # Construct current state
            @threaded for i in eachindex(integrator.du)
                integrator.u_tmp[i] = integrator.u_old[i] +
                                      alg.a_matrix[stage - alg.num_stages + alg.num_stage_evals - 2,
                                                   1] *
                                      integrator.k1[i] +
                                      alg.a_matrix[stage - alg.num_stages + alg.num_stage_evals - 2,
                                                   2] *
                                      integrator.k_higher[i]
            end

            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)

            @threaded for i in eachindex(integrator.du)
                integrator.k_higher[i] = integrator.du[i] * integrator.dt
                integrator.u_e[i] += alg.b_embedded[stage - alg.num_stages + alg.num_stage_evals] *
                                   integrator.k_higher[i]
            end

            # Last stage
            @threaded for i in eachindex(integrator.du)
                integrator.u_tmp[i] = integrator.u_old[i] +
                                    alg.a_matrix[alg.num_stage_evals - 2, 1] *
                                    integrator.k1[i] +
                                    alg.a_matrix[alg.num_stage_evals - 2, 2] *
                                    integrator.k_higher[i]
            end

            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                        integrator.t + alg.c[alg.num_stages] * integrator.dt)

            @threaded for i in eachindex(integrator.u)
                # Note that 'k_higher' carries the values of K_{S-1}
                # and that we construct 'K_S' "in-place" from 'integrator.du'
                integrator.u[i] = integrator.u_old[i] + (integrator.k1[i] + integrator.k_higher[i] +
                4.0 * integrator.du[i] * integrator.dt) / 6.0
            end
        end
    end # PairedExplicitRK step timer
    
    # Compute the estimated local truncation error
    integrator.EEst = norm((integrator.u - integrator.u_e) ./ (alg.abs_tol .+ alg.rel_tol .* max.(
        abs.(integrator.u), 
        abs.(integrator.u_e))), 2) # Use this norm according to PID controller from OrdinaryDiffEq.jl

    integrator.iter += 1
    integrator.t += integrator.dt # The logic and the function to increment the accepted time step has to be called here.

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

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::EmbeddedPairedRK3Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)
end
end # @muladd
