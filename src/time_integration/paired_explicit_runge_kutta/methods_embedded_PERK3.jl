# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
using DelimitedFiles: readdlm

@muladd begin
#! format: noindent

function solve_b_embedded end

#=
function compute_b_embedded_coeffs(num_stage_evals, num_stages, embedded_monomial_coeffs, a_unknown, c)

    A = zeros(num_stage_evals - 1, num_stage_evals - 1)
    b_embedded = zeros(num_stage_evals - 1)
    rhs = [1, 1/2, embedded_monomial_coeffs...]

    # sum(b) = 1
    A[1, :] .= 1

    # The second order constraint: dot(b,c) = 1/2
    for i in 2:num_stage_evals - 1
        A[2, i] = c[num_stages - num_stage_evals + i]
    end

    # Fill the A matrix
    for i in 3:(num_stage_evals - 1)
        # z^i
        for j in i: (num_stage_evals - 1)
            println("i = ", i, ", j = ", j)
            println("[num_stages - num_stage_evals + j - 1] = ", num_stages - num_stage_evals + j - 1)
            A[i,j] = c[num_stages - num_stage_evals + j - 1]
            # number of times a_unknown should be multiplied in each power of z
            for k in 1: i-2
                # so we want to get from a[k] * ... i-2 times (1 time is already accounted for by c-coeff)
                # j-1 - k + 1 = j - k
                println("a_unknown at index: ", j - k)
                A[i, j] *= a_unknown[j - k] # a[k] is in the same stage as b[k-1] -> since we also store b_1
            end
        end
        #rhs[i] /= factorial(i)
    end

    display(A)

    b_embedded = A \ rhs
    return b_embedded
end
=#

# Some function defined so that I can check if the second order condition is met. This will be removed later.
function construct_b_vector(b_unknown, num_stages_embedded, num_stage_evals_embedded)
    # Construct the b vector
    b = [
        b_unknown[1],
        zeros(Float64, num_stages_embedded - num_stage_evals_embedded-1)...,
        b_unknown[2:end]...,
        0
    ]
    return b
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
    b = zeros(num_stage_evals - 1)

    # Special case of e = 3
    if num_stage_evals == 3
        a_unknown = [0.25] # Use classic SSPRK33 (Shu-Osher) Butcher Tableau
    else
        # Calculate coefficients of the stability polynomial in monomial form
        consistency_order = 3
        dtmax = tspan[2] - tspan[1]
        dteps = 1.0f-9

        num_eig_vals, eig_vals = filter_eig_vals(eig_vals; verbose)

        monomial_coeffs, dt_opt_a = bisect_stability_polynomial(consistency_order,
                                                              num_eig_vals, num_stage_evals,
                                                              dtmax, dteps,
                                                              eig_vals; verbose)
        monomial_coeffs = undo_normalization!(monomial_coeffs, consistency_order,
                                              num_stage_evals)

        # Solve the nonlinear system of equations from monomial coefficient and
        # Butcher array abscissae c to find Butcher matrix A
        # This function is extended in TrixiNLsolveExt.jl
        #a_unknown = solve_a_butcher_coeffs_unknown!(a_unknown, num_stages,
        #                                            num_stage_evals,
        #                                            monomial_coeffs, cS2, c;
        #                                            verbose)

        b, dt_opt_b, a_unknown, c = solve_b_embedded(consistency_order, num_eig_vals, num_stage_evals,
        num_stages,
        dtmax, dteps, eig_vals, monomial_coeffs;
        verbose = true)

        println("dt_opt_b = ", dt_opt_b)
        println("dt_opt_a = ", dt_opt_a)

        # Calculate and print the percentage difference
        percentage_difference = (dt_opt_b / dt_opt_a) * 100
        println("Percentage difference (dt_opt_b / dt_opt_a * 100) = ", percentage_difference)

        b_full = construct_b_vector(b, num_stages - 1, num_stage_evals - 1)

        println("dot(b, c) = ", dot(b_full, c))
        println("sum(b) = ", sum(b_full))
        println("b: ", b)
        println("a_unknown: ", a_unknown)
        println("c: ", c)
        

        error("b found.")
    end
    # Fill A-matrix in P-ERK style
    a_matrix = zeros(num_stage_evals - 2, 2)
    a_matrix[:, 1] = c[(num_stages - num_stage_evals + 3):end]
    a_matrix[:, 1] -= a_unknown
    a_matrix[:, 2] = a_unknown

    return a_matrix, b, c, dt_opt_a, dt_opt_b # Return the optimal time step from the b coefficients for testing purposes
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

    b = zeros(coeffs_max)

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
    b = readdlm(path_b_coeffs, Float64)
    num_b_coeffs = size(b, 1)
    @assert num_b_coeffs == coeffs_max

    return a_matrix, b, c
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
      in some coefficients in the b vector of the Butcher tableau of the Runge Kutta method.
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
    b::Vector{Float64}
    c::Vector{Float64}
    dt_opt_a::Float64
    dt_opt_b::Float64
end # struct EmbeddedPairedRK3

# Constructor for previously computed A Coeffs
function EmbeddedPairedRK3(num_stages, num_stage_evals,
                           base_path_coeffs::AbstractString, dt_opt_a, dt_opt_b;
                           cS2 = 1.0f0)
    a_matrix, b, c = compute_EmbeddedPairedRK3_butcher_tableau(num_stages,
                                                               num_stage_evals,
                                                               base_path_coeffs;
                                                               cS2)

    return EmbeddedPairedRK3(num_stages, num_stage_evals, a_matrix, b, c, dt_opt_a, dt_opt_b)
end

# Constructor that computes Butcher matrix A coefficients from a semidiscretization
function EmbeddedPairedRK3(num_stages, num_stage_evals, tspan,
                           semi::AbstractSemidiscretization;
                           verbose = false, cS2 = 1.0f0)
    eig_vals = eigvals(jacobian_ad_forward(semi))

    return EmbeddedPairedRK3(num_stages, num_stage_evals, tspan, eig_vals; verbose, cS2)
end

# Constructor that calculates the coefficients with polynomial optimizer from a list of eigenvalues
function EmbeddedPairedRK3(num_stages, num_stage_evals, tspan,
                           eig_vals::Vector{ComplexF64};
                           verbose = false, cS2 = 1.0f0)
    a_matrix, b, c, dt_opt_a, dt_opt_b = compute_EmbeddedPairedRK3_butcher_tableau(num_stages,
                                                                       num_stage_evals,
                                                                       tspan,
                                                                       eig_vals;
                                                                       verbose, cS2)
    return EmbeddedPairedRK3(num_stages, num_stage_evals, a_matrix, b, c, dt_opt_a, dt_opt_b)
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
    force_stepfail::Bool
    # PairedExplicitRK stages:
    k1::uType
    k_higher::uType
end

function init(ode::ODEProblem, alg::EmbeddedPairedRK3;
              dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # PairedExplicitRK stages
    k1 = zero(u0)
    k_higher = zero(u0)

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    integrator = EmbeddedPairedRK3Integrator(u0, du, u_tmp, t0, tdir, dt, dt, iter,
                                             ode.p,
                                             (prob = ode,), ode.f, alg,
                                             PairedExplicitRKOptions(callback,
                                                                     ode.tspan;
                                                                     kwargs...),
                                             false, true, false,
                                             k1, k_higher)

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
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve!(integrator)
end

function solve!(integrator::EmbeddedPairedRK3Integrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        step!(integrator)
    end # "main loop" timer

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

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        # k1
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        @threaded for i in eachindex(integrator.du)
            integrator.k1[i] = integrator.du[i] * integrator.dt
        end

        @threaded for i in eachindex(integrator.u)
            # add the contribution of the first stage
            integrator.u[i] += alg.b[1] * integrator.k1[i]
        end

        # Higher stages where the weight of b in the butcher tableau is zero
        for stage in 2:(alg.num_stages - alg.num_stage_evals)
            # Construct current state
            @threaded for i in eachindex(integrator.du)
                integrator.u_tmp[i] = integrator.u[i] + alg.c[stage] * integrator.k1[i]
            end

            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)

            @threaded for i in eachindex(integrator.du)
                integrator.k_higher[i] = integrator.du[i] * integrator.dt
            end
        end

        # #k_(s-e+1) and k_(s-e+2)
        # Construct current state

        for j in 1:2
            @threaded for i in eachindex(integrator.du)
                integrator.u_tmp[i] = integrator.u[i] +
                                    alg.c[alg.num_stages - alg.num_stage_evals + 2] *
                                    integrator.k1[i]
            end

            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                        integrator.t +
                        alg.c[alg.num_stages - alg.num_stage_evals + j] * integrator.dt)

            @threaded for i in eachindex(integrator.du)
                integrator.k_higher[i] = integrator.du[i] * integrator.dt
            end

            @threaded for i in eachindex(integrator.u)
                integrator.u[i] += integrator.k_higher[i] *
                                alg.b[j+1]
            end
        end

        # Higher stages after num_stage_evals where b is non-zero
        for stage in (alg.num_stages - alg.num_stage_evals + 3):(alg.num_stages - 1)
            # Construct current state
            @threaded for i in eachindex(integrator.du)
                integrator.u_tmp[i] = integrator.u[i] +
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
            end

            @threaded for i in eachindex(integrator.u)
                integrator.u[i] += integrator.k_higher[i] *
                                   alg.b[stage - alg.num_stages + alg.num_stage_evals+1]
            end
        end

        
        # Last stage
        @threaded for i in eachindex(integrator.du)
            integrator.u_tmp[i] = integrator.u[i] +
                                  alg.a_matrix[alg.num_stage_evals - 2, 1] *
                                  integrator.k1[i] +
                                  alg.a_matrix[alg.num_stage_evals - 2, 2] *
                                  integrator.k_higher[i]
        end
        

        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[alg.num_stages] * integrator.dt)
    end # PairedExplicitRK step timer

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

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::EmbeddedPairedRK3Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)
end
end # @muladd
