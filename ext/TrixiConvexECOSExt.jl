# Package extension for adding Convex-based features to Trixi.jl
module TrixiConvexECOSExt

# Required for coefficient optimization in P-ERK scheme integrators
if isdefined(Base, :get_extension)
    using Convex: MOI, solve!, Variable, minimize, evaluate, sumsquares, dot
    using ECOS: Optimizer
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..Convex: MOI, solve!, Variable, minimize, evaluate, sumsquares, dot
    using ..ECOS: Optimizer
end

# Use other necessary libraries
using LinearAlgebra: eigvals

# Use functions that are to be extended and additional symbols that are not exported
using Trixi: Trixi, undo_normalization!, bisect_stability_polynomial, @muladd

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

function construct_b_vector(b_unknown, num_stages_embedded, num_stage_evals_embedded)
    # Construct the b vector
    b = [
        b_unknown[1];
        fill(0.0, num_stages_embedded - num_stage_evals_embedded);
        b_unknown[2:end];
        0.0
    ]
    return b
end


# Compute stability polynomials for paired explicit Runge-Kutta up to specified consistency
# order(p = 2), including contributions from free coefficients for higher orders, and
# return the maximum absolute value
function stability_polynomials!(pnoms, 
                                num_stages_embedded, num_stage_evals_embedded,
                                normalized_powered_eigvals_scaled,
                                a_unknown, b_unknown, c)
    num_stages = num_stages_embedded + 1
    num_stage_evals = num_stage_evals_embedded + 1
    b = construct_b_vector(b_unknown, num_stages_embedded, num_stage_evals_embedded)
    a = zeros(num_stages)
    a[2] = c[2]

    num_unknown = length(a_unknown)
    for i = 1:num_unknown
        a[num_stages - i + 1] = a_unknown[num_unknown - i + 1]
    end

    num_eig_vals = length(pnoms)

    # Initialize with 1 + z
    for i in 1:num_eig_vals
        pnoms[i] = 1.0 + normalized_powered_eigvals_scaled[i, 1]
    end
    # z^2: b^T * c
    #pnoms += dot(b, c) * normalized_powered_eigvals_scaled[:, 2]
    pnoms += 0.5 * normalized_powered_eigvals_scaled[:, 2]

    # Contribution from free coefficients
    for i in 3:num_stage_evals_embedded
        sum = 0.0
        for j = i + num_stages_embedded - num_stage_evals_embedded:num_stages_embedded
            prod = 1.0
            for k = 3 + j - i:j
                prod *= a[k]
            end
            sum += prod * b[j] * c[j - i + 2]
        end
        pnoms += sum * normalized_powered_eigvals_scaled[:, i] 
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
function Trixi.bisect_stability_polynomial(num_eig_vals, eig_vals,
                                     num_stages, num_stage_evals,
                                     num_stages_embedded, num_stage_evals_embedded,
                                     a, c,
                                     dtmax, dteps)
    dtmin = 0.0
    dt = -1.0
    abs_p = -1.0

    # Construct stability polynomial for each eigenvalue
    pnoms = ones(Complex{Float64}, num_eig_vals, 1)

    # There are e - 2 free variables for the stability polynomial of the embedded scheme
    b_unknown = Variable(num_stage_evals - 1)

    normalized_powered_eigvals = zeros(Complex{Float64}, num_eig_vals, num_stage_evals)

    for j in 1:num_stage_evals
        #fac_j = factorial(j)
        for i in 1:num_eig_vals
            #normalized_powered_eigvals[i, j] = eig_vals[i]^j / fac_j
            # Try first without factorial normalization
            normalized_powered_eigvals[i, j] = eig_vals[i]^j
        end
    end

    normalized_powered_eigvals_scaled = similar(normalized_powered_eigvals)

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

        constraints = []
        for i in 1:num_stage_evals -1
            push!(constraints, b_unknown[i] >=-1e-6)
        end
        push!(constraints, sum(b_unknown) == 1.0)
        # Second-order constraint
        # Since c[1] is always 0 we can ignore the contribution of b[1] and only account for the ones from b_unknown
        push!(constraints, 2 * dot(b_unknown[2:end], c[num_stages - num_stage_evals + 2:num_stages - 1]) == 1.0) # Since c[1] = 0.0
        

        # Use last optimal values for b in (potentially) next iteration
        problem = minimize(stability_polynomials!(pnoms,
                                                  num_stages_embedded, num_stage_evals_embedded,
                                                  normalized_powered_eigvals_scaled,
                                                  a, b_unknown, c),
                                                  constraints
                                                  )
                                                  
                                                          
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

    b_unknown_opt = evaluate(b_unknown)

    # Catch case S = 3 (only one opt. variable)
    if isa(b_unknown_opt, Number)
        b_unknown_opt = [b_unknown_opt]
    end

    return b_unknown_opt, dt
end
end # @muladd

end # module TrixiConvexECOSExt
