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
using Trixi: Trixi, undo_normalization!, bisect_stability_polynomial, solve_b_embedded, @muladd

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

#=
function compute_b_embedded_coeffs(num_stage_evals, num_stages, embedded_monomial_coeffs, a_unknown, c)

    A = zeros(num_stage_evals - 1, num_stage_evals - 1)
    b_embedded = zeros(num_stage_evals - 1)
    rhs = [1, 1/2, embedded_monomial_coeffs...]  # Cast embedded_monomial_coeffs to Float64 if needed

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
        rhs[i] /= factorial(i)
    end

    A_inv = inv(A)

    b_embedded = A_inv * rhs
    
    return b_embedded
end
=#

#TODO: Add an optimization of the embedded scheme that subject the b solved from the set of gamme to be +
#      the same as the b from the original scheme. This will be done by using the same optimization as the normal scheme but with cosntraint and the function
#      that will be used to construct the b vector from the gamma vector.
function Trixi.solve_b_embedded(consistency_order, num_eig_vals, num_stage_evals, num_stages,
                                           dtmax, dteps, eig_vals, a_unknown, c;
                                           verbose = false)
    dtmin = 0.0
    dt = -1.0
    abs_p = -1.0
    consistency_order_embedded = consistency_order - 1

    num_stage_evals_embedded = num_stage_evals -1

    # Construct stability polynomial for each eigenvalue
    pnoms = ones(Complex{Float64}, num_eig_vals, 1)

    # Init datastructure for monomial coefficients
    gamma = Variable(num_stage_evals_embedded - consistency_order_embedded)

    normalized_powered_eigvals = zeros(Complex{Float64}, num_eig_vals, num_stage_evals_embedded)

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

        #constraint = compute_b_embedded_coeffs(num_stage_evals, num_stages, gamma, a_unknown, c).>= -1e-5

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

    b_embedded = compute_b_embedded_coeffs(num_stage_evals, num_stages, gamma_opt, a_unknown, c)

    return b_embedded, dt
end

function Trixi.compute_b_embedded_coeffs(num_stage_evals, num_stages, embedded_monomial_coeffs, a_unknown, c)

    A = zeros(num_stage_evals-1, num_stage_evals)
    rhs = [1, 1/2, embedded_monomial_coeffs...]

    # Define the variables
    b_embedded = Variable(num_stage_evals)  # the unknown coefficients we want to solve for
    
    # Initialize A matrix for the constraints
    A[1, :] .= 1  # sum(b) = 1 constraint row
    for i in 1:num_stage_evals-1
        A[2, i+1] = c[num_stages - num_stage_evals + i]  # second order constraint: dot(b, c) = 1/2
    end

    # Fill the A matrix with other constraints based on monomial coefficients
    for i in 3:(num_stage_evals-1)
        for j in i+1:num_stage_evals
            A[i, j] = c[num_stages - num_stage_evals + j - 2]
            for k in 1:i-2
                A[i, j] *= a_unknown[j - 1 - k]  # recursive dependence on `a_unknown`
            end
        end
    end

    println("A matrix of finding b")
    display(A)

    # Set up weights to prioritize the first two constraints
    weights = [2, 2, ones(num_stage_evals-3)...]  # Heavier weight for the first two rows
    weighted_residual = weights .* (A * b_embedded - rhs)  # Element-wise multiplication for weighting

    # Set up the objective to minimize the weighted norm of the difference
    problem = minimize(sumsquares(weighted_residual), [b_embedded >= 0])

    solve!(problem, # Parameters taken from default values for EiCOS
                                MOI.OptimizerWithAttributes(Optimizer, "gamma" => 0.99,
                                "delta" => 2e-7,
                                "feastol" => 1e-9,
                                "abstol" => 1e-9,
                                "reltol" => 1e-9,
                                "feastol_inacc" => 1e-7,
                                "abstol_inacc" => 5e-6,
                                "reltol_inacc" => 5e-7,
                                "nitref" => 9,
                                "maxit" => 1000000,
                                "verbose" => 3); silent_solver = true)
    

    ot = problem.optval
    println("Optimal value of the objective function: ", ot)
    return evaluate(b_embedded)
end
end # @muladd

end # module TrixiConvexECOSExt
