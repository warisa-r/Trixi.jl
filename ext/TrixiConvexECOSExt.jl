# Package extension for adding Convex-based features to Trixi.jl
module TrixiConvexECOSExt

# Required for coefficient optimization in P-ERK scheme integrators
if isdefined(Base, :get_extension)
    using Convex: MOI, solve!, Variable, minimize, evaluate, sumsquares
    using ECOS: Optimizer
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..Convex: MOI, solve!, Variable, minimize, evaluate, sumsquares
    using ..ECOS: Optimizer
end

# Use other necessary libraries
using LinearAlgebra: eigvals

# Use functions that are to be extended and additional symbols that are not exported
using Trixi: Trixi, undo_normalization!, bisect_stability_polynomial, solve_b_embedded, solve_a_butcher_coeffs_unknown!, @muladd

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Undo normalization of stability polynomial coefficients by index factorial
# relative to consistency order.
function Trixi.undo_normalization!(gamma_opt, consistency_order, num_stage_evals)
    for k in (consistency_order + 1):num_stage_evals
        gamma_opt[k - consistency_order] = gamma_opt[k - consistency_order] /
                                           factorial(k)
    end
    return gamma_opt
end

# Compute stability polynomials for paired explicit Runge-Kutta up to specified consistency
# order, including contributions from free coefficients for higher orders, and
# return the maximum absolute value
function stability_polynomials!(pnoms, consistency_order, num_stage_evals,
                                normalized_powered_eigvals_scaled,
                                gamma)
    num_eig_vals = length(pnoms)

    # Initialize with zero'th order (z^0) coefficient
    for i in 1:num_eig_vals
        pnoms[i] = 1.0
    end

    # First `consistency_order` terms of the exponential
    for k in 1:consistency_order
        for i in 1:num_eig_vals
            pnoms[i] += normalized_powered_eigvals_scaled[i, k]
        end
    end

    # Contribution from free coefficients
    for k in (consistency_order + 1):num_stage_evals
        pnoms += gamma[k - consistency_order] * normalized_powered_eigvals_scaled[:, k]
    end

    # For optimization only the maximum is relevant
    return maximum(abs(pnoms))
end

#=
The following structures and methods provide a simplified implementation to 
discover optimal stability polynomial for a given set of `eig_vals`
These are designed for the one-step (i.e., Runge-Kutta methods) integration of initial value ordinary 
and partial differential equations.

- Ketcheson and Ahmadia (2012).
Optimal stability polynomials for numerical integration of initial value problems
[DOI: 10.2140/camcos.2012.7.247](https://doi.org/10.2140/camcos.2012.7.247)
=#

# Perform bisection to optimize timestep for stability of the polynomial
function Trixi.bisect_stability_polynomial(consistency_order, num_eig_vals,
                                           num_stage_evals,
                                           dtmax, dteps, eig_vals;
                                           verbose = false)
    dtmin = 0.0
    dt = -1.0
    abs_p = -1.0

    # Construct stability polynomial for each eigenvalue
    pnoms = ones(Complex{Float64}, num_eig_vals, 1)

    # Init datastructure for monomial coefficients
    gamma = Variable(num_stage_evals - consistency_order)

    normalized_powered_eigvals = zeros(Complex{Float64}, num_eig_vals, num_stage_evals)

    for j in 1:num_stage_evals
        fac_j = factorial(j)
        for i in 1:num_eig_vals
            normalized_powered_eigvals[i, j] = eig_vals[i]^j / fac_j
        end
    end

    normalized_powered_eigvals_scaled = similar(normalized_powered_eigvals)

    if verbose
        println("Start optimization of stability polynomial \n")
    end

    # Bisection on timestep
    while dtmax - dtmin > dteps
        dt = 0.5 * (dtmax + dtmin)

        # Compute stability polynomial for current timestep
        for k in 1:num_stage_evals
            dt_k = dt^k
            for i in 1:num_eig_vals
                normalized_powered_eigvals_scaled[i, k] = dt_k *
                                                          normalized_powered_eigvals[i,
                                                                                     k]
            end
        end

        # Use last optimal values for gamma in (potentially) next iteration
        problem = minimize(stability_polynomials!(pnoms, consistency_order,
                                                  num_stage_evals,
                                                  normalized_powered_eigvals_scaled,
                                                  gamma))

        solve!(problem,
               # Parameters taken from default values for EiCOS
               MOI.OptimizerWithAttributes(Optimizer, "gamma" => 0.99,
                                           "delta" => 2e-7,
                                           "feastol" => 1e-9,
                                           "abstol" => 1e-9,
                                           "reltol" => 1e-9,
                                           "feastol_inacc" => 1e-4,
                                           "abstol_inacc" => 5e-5,
                                           "reltol_inacc" => 5e-5,
                                           "nitref" => 9,
                                           "maxit" => 100,
                                           "verbose" => 3); silent_solver = true)

        abs_p = problem.optval

        if abs_p < 1
            dtmin = dt
        else
            dtmax = dt
        end
    end

    if verbose
        println("Concluded stability polynomial optimization \n")
    end

    gamma_opt = evaluate(gamma)

    # Catch case S = 3 (only one opt. variable)
    if isa(gamma_opt, Number)
        gamma_opt = [gamma_opt]
    end

    return gamma_opt, dt
end

function compute_b_embedded_coeffs(num_stage_evals, num_stages, monomial_coeffs, gamma, cS2;verbose)
    A = zeros(num_stage_evals - 1, num_stage_evals - 1)
    b_embedded = zeros(num_stage_evals - 1)
    rhs = [1, 1 / 2, gamma...]  # last element of embedded_monomial_coeffs is actually c_{S-2}
    
    c_temp = []
    # Linear increasing timestep for remainder
    for i in 2:(num_stages - 2)
        push!(c_temp, cS2 * (i - 1) / (num_stages - 3))
    end

    # Last timesteps as for SSPRK33, see motivation in Section 3.3 of
    # https://doi.org/10.1016/j.jcp.2024.113223
    c= [c_temp..., 1.0f0, 0.5f0]

    a_unknown = Variable(num_stages - 2)

    a_unknown = solve_a_butcher_coeffs_unknown!(a_unknown, num_stages,
                                                    num_stage_evals,
                                                    monomial_coeffs, cS2, c; verbose)

    # sum(b) = 1
    A[1, :] .= 1

    # The second order constraint: dot(b,c) = 1/2
    for i in 2:(num_stage_evals - 1)
        A[2, i] = c[num_stages - num_stage_evals + i]
    end

    # Fill the A matrix
    for i in 3:(num_stage_evals - 1)
        # z^i
        for j in i:(num_stage_evals - 1)
            println("i = ", i, ", j = ", j)
            println("[num_stages - num_stage_evals + j - 1] = ",
                    num_stages - num_stage_evals + j - 1)
            A[i, j] = c[num_stages - num_stage_evals + j - 1]
            # number of times a_unknown should be multiplied in each power of z
            for k in 1:(i - 2)
                # so we want to get from a[k] * ... i-2 times (1 time is already accounted for by c-coeff)
                # j-1 - k + 1 = j - k
                println("a_unknown at index: ", j - k)
                A[i, j] *= a_unknown[j - k] # a[k] is in the same stage as b[k-1] -> since we also store b_1
            end
        end
        rhs[i] /= factorial(i)
    end

    A_inv = inv(A)

    b_embedded = A_inv * rhs

    return b_embedded
end

#TODO: Add an optimization of the embedded scheme that subject the b solved from the set of gamme to be +
#      the same as the b from the original scheme. This will be done by using the same optimization as the normal scheme but with cosntraint and the function
#      that will be used to construct the b vector from the gamma vector.
function Trixi.solve_b_embedded(consistency_order, num_eig_vals, num_stage_evals,
                                num_stages,
                                dtmax, dteps, eig_vals, monomial_coeffs;
                                verbose = false)
    dtmin = 0.0
    dt = -1.0
    abs_p = -1.0
    consistency_order_embedded = consistency_order - 1

    num_stage_evals_embedded = num_stage_evals - 1

    # Construct stability polynomial for each eigenvalue
    pnoms = ones(Complex{Float64}, num_eig_vals, 1)

    # Init datastructure for monomial coefficients
    gamma = Variable(num_stage_evals_embedded - consistency_order_embedded)
    cS2 = Variable(1) # c also need to be optimized

    normalized_powered_eigvals = zeros(Complex{Float64}, num_eig_vals,
                                       num_stage_evals_embedded)

    for j in 1:num_stage_evals_embedded
        fac_j = factorial(j)
        for i in 1:num_eig_vals
            normalized_powered_eigvals[i, j] = eig_vals[i]^j / fac_j
        end
    end

    normalized_powered_eigvals_scaled = similar(normalized_powered_eigvals)

    if verbose
        println("Start optimization of stability polynomial \n")
    end

    # Bisection on timestep
    while dtmax - dtmin > dteps
        dt = 0.5 * (dtmax + dtmin)

        # Compute stability polynomial for current timestep
        for k in 1:num_stage_evals_embedded
            dt_k = dt^k
            for i in 1:num_eig_vals
                normalized_powered_eigvals_scaled[i, k] = dt_k *
                                                          normalized_powered_eigvals[i,
                                                                                     k]
            end
        end

        # all b and c of the embedded scheme must be positive values. Additionally, c_{S-2} <=  1
        constraint = [
            compute_b_embedded_coeffs(num_stage_evals, num_stages, monomial_coeffs, gamma, cS2;verbose) .>= -1e-5,
            cS2 > 0,
            cS2 <= 1
        ]

        # Use last optimal values for gamma in (potentially) next iteration
        problem = minimize(stability_polynomials!(pnoms, consistency_order_embedded,
                                                  num_stage_evals_embedded,
                                                  normalized_powered_eigvals_scaled,
                                                  gamma), constraint)

        solve!(problem,
               # Parameters taken from default values for EiCOS
               MOI.OptimizerWithAttributes(Optimizer, "gamma" => 0.99,
                                           "delta" => 2e-7,
                                           "feastol" => 1e-9,
                                           "abstol" => 1e-9,
                                           "reltol" => 1e-9,
                                           "feastol_inacc" => 1e-4,
                                           "abstol_inacc" => 5e-5,
                                           "reltol_inacc" => 5e-5,
                                           "nitref" => 9,
                                           "maxit" => 100,
                                           "verbose" => 3); silent_solver = true)

        abs_p = problem.optval

        if abs_p < 1
            dtmin = dt
        else
            dtmax = dt
        end
    end

    if verbose
        println("Concluded stability polynomial optimization \n")
    end

    gamma_opt = evaluate(gamma)

    # Catch case S = 3 (only one opt. variable)
    if isa(gamma_opt, Number)
        gamma_opt = [gamma_opt]
    end

    c = compute_c_coeffs(num_stages, gamma_opt[num_stage_evals_embedded - consistency_order_embedded + 1])

    a_unknown = solve_a_butcher_coeffs_unknown!(a_unknown, num_stages,
                                                num_stage_evals,
                                                monomial_coeffs, gamma_opt[num_stage_evals_embedded - consistency_order_embedded + 1], c;
                                                verbose)

    b_embedded = compute_b_embedded_coeffs(num_stage_evals, num_stages, gamma_opt,
                                           a_unknown, c)

    return b_embedded, dt, a_unknown, c
end
end # @muladd

end # module TrixiConvexECOSExt
