# Package extension for adding NLsolve-based features to Trixi.jl
module TrixiJuMPIpoptExt

# Required for finding coefficients in Butcher tableau in the third order of 
# P-ERK scheme integrators
if isdefined(Base, :get_extension)
    using JuMP
    using Ipopt
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..JuMP
    using ..Ipopt
end

# Use other necessary libraries for the dot product in finding b_unknown
using LinearAlgebra: dot

# Use functions that are to be extended and additional symbols that are not exported
using Trixi: Trixi, @muladd

# We use a random initialization of the nonlinear solver.
# To make the tests reproducible, we need to seed the RNG
using StableRNGs: StableRNG, rand

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent
function Trixi.solve_a_butcher_coeffs_with_JuMP(num_stages, num_stage_evals,
                                                monomial_coeffs, c;
                                                verbose::Bool = false,
                                                max_iter::Int = 100000)
    model = Model(Ipopt.Optimizer)

    if !verbose
        # Set Ipopt options to make it silent
        set_optimizer_attribute(model, "print_level", 0)
        set_optimizer_attribute(model, "sb", "yes")
    end

    # Add variables for the unknowns (a_unknown coefficients) with non-negativity constraint
    a_unknown = @variable(model, [i in 1:(num_stage_evals - 2)], lower_bound=0.0)

    # Dummy function for JuMP to optimize
    @NLobjective(model, Min, 1.0)

    # Start defining the non-linear constraints after our non-linear equations

    # For explicit methods, a_{1,1} = 0 and a_{2,1} = c_2 (Butcher's condition)
    a_coeff = [0, c[2], a_unknown...]

    # Equality constraint array that ensures that the stability polynomial computed from 
    # the to-be-constructed Butcher-Tableau matches the monomial coefficients of the 
    # optimized stability polynomial.
    for i in 1:(num_stage_evals - 4)
        term1 = a_coeff[num_stage_evals - 1]
        term2 = a_coeff[num_stage_evals]
        for j in 1:i
            term1 *= a_coeff[num_stage_evals - 1 - j]
            term2 *= a_coeff[num_stage_evals - j]
        end
        term1 *= c[num_stages - 2 - i] * 1 / 6 # 1 / 6 = b_{S-1}
        term2 *= c[num_stages - 1 - i] * 2 / 3 # 2 / 3 = b_S

        @NLconstraint(model, monomial_coeffs[i] - (term1 + term2)==0.0)
    end

    # Highest coefficient: Only one term present
    i = num_stage_evals - 3
    term2 = a_coeff[num_stage_evals]
    for j in 1:i
        term2 *= a_coeff[num_stage_evals - j]
    end
    term2 *= c[num_stages - 1 - i] * 2 / 3 # 2 / 3 = b_S

    @NLconstraint(model, monomial_coeffs[i] - term2==0.0)

    # Third-order consistency condition
    term1 = 4 * a_coeff[num_stage_evals] # Have to seperate this into two terms. Otherwise, JuMP throws an error with muladd.
    term2 = a_coeff[num_stage_evals - 1]
    @NLconstraint(model, 1 - term1 - term2==0.0)

    # Add nonlinear constraints ensuring that every a_unknown is non-negative
    for i in 1:(num_stage_evals - 2)
        @NLconstraint(model, a_unknown[i]>=0.0)
    end

    # Adding condition involving c and a_unknown (ensuring c[i] - a_unknown_value >= 0.0)
    for i in (num_stages - num_stage_evals + 3):(num_stage_evals - 2)
        @NLconstraint(model,
                      c[i] - a_unknown[i - (num_stage_evals - num_stages + 2)]>=0.0)
    end

    # Iterative attempts with different initial guesses
    rng = StableRNG(555)  # Use Stable RNG for reproducibility
    for attempt in 1:max_iter
        if verbose
            println("Attempt $attempt with a new initial guess")
        end

        # Set random initial guess
        for i in 1:length(a_unknown)
            set_start_value(a_unknown[i], rand(rng) * 0.1)
        end

        # Solve the model
        optimize!(model)

        status = termination_status(model)
        if status == LOCALLY_SOLVED || status == OPTIMAL
            if verbose
                println("Solution found after $attempt attempts.")
            end
            return value.(a_unknown), attempt
        end
    end

    return fill(-1, N), max_iter # Instead of popping an error, gives this as a return value
end

function compute_b_embedded_coeffs(model, b_embedded, num_stage_evals, num_stages,
                                   monomial_coeffs_embedded,
                                   a_unknown::Vector{T}, c::Vector{U}) where {T, U}

    #A = Matrix{T}(undef, num_stage_evals - 1, num_stage_evals - 1)  # Use undefined elements for flexibility
    A = Matrix{Any}(undef, num_stage_evals - 1, num_stage_evals - 1)  # Define an untyped matrix
    fill!(A, 0)  # Fill the matrix with zeros
    rhs = [1, 1 / 2, monomial_coeffs_embedded...]

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
            #println("i = ", i, ", j = ", j)
            #println("[num_stages - num_stage_evals + j - 1] = ", num_stages - num_stage_evals + j - 1)
            A[i, j] = c[num_stages - num_stage_evals + j - 1]
            # number of times a_unknown should be multiplied in each power of z
            for k in 1:(i - 2)
                # so we want to get from a[k] * ... i-2 times (1 time is already accounted for by c-coeff)
                # j-1 - k + 1 = j - k
                #println("a_unknown at index: ", j - k)
                A[i, j] *= a_unknown[j - k] # a[k] is in the same stage as b[k-1] -> since we also store b_1
            end
        end
        #rhs[i] /= factorial(i) # The monomial coefficient has already been normalized.
    end

    #display(A)

    # NLconstraint to solve the system of equations
    for i in 1:(num_stage_evals - 1)
        row_term = 0.0
        for j in 1:(num_stage_evals - 1)
            row_term += A[i, j] * b_embedded[j]  # Explicit summation of products
        end
        @NLconstraint(model, row_term - rhs[i]==0.0)    
    end
end

function Trixi.optimize_c_embedded_scheme(num_stages, num_stage_evals, monomial_coeffs,
                                          monomial_coeffs_embedded; verbose = false, max_iter = 50000) # The monomial coefficients must be normalized before passing to this function.
    # Define the model
    model = Model(Ipopt.Optimizer)

    verbose = true # See what happens first. Remove this laters

    if !verbose
        # Set Ipopt options to make it silent
        set_optimizer_attribute(model, "print_level", 0)
        set_optimizer_attribute(model, "sb", "yes")
    end

    # Set free c as the variable to be optimized. There are s-2 free cs as the two last entries of the coefficients
    # are fixed.
    c_free = @variable(model, [i in 1:(num_stages - 2)], lower_bound=0.0,
                       upper_bound=1.0)

    # Add variables for the unknowns (a_unknown coefficients) with non-negativity constraint
    a_unknown = @variable(model, [i in 1:(num_stage_evals - 2)], lower_bound=0.0)

    # Add variables for the unknowns (b_unknown coefficients) with non-negativity constraint
    b_embedded = @variable(model, [i in 1:(num_stage_evals - 1)], lower_bound=0.0)

    # Dummy objective (Min, 1) to make the model work. I find that using something else than 1 cause things to go haywire.
    @NLobjective(model, Min, 1.0)

    # Define full c vector
    c = [c_free..., 1.0f0, 0.5f0]

    # Set the NLconstraint that c must be between 0 and 1.
    for i in 1:(num_stage_evals - 2)
        @NLconstraint(model, 0.0<=c_free[i]<=1.0)
    end

    # For explicit methods, a_{1,1} = 0 and a_{2,1} = c_2 (Butcher's condition)
    a_coeff = [0, c[2], a_unknown...]

    # Equality constraint array that ensures that the stability polynomial computed from 
    # the to-be-constructed Butcher-Tableau matches the monomial coefficients of the 
    # optimized stability polynomial.
    for i in 1:(num_stage_evals - 4)
        term1 = a_coeff[num_stage_evals - 1]
        term2 = a_coeff[num_stage_evals]
        for j in 1:i
            term1 *= a_coeff[num_stage_evals - 1 - j]
            term2 *= a_coeff[num_stage_evals - j]
        end
        term1 *= c[num_stages - 2 - i] * 1 / 6 # 1 / 6 = b_{S-1}
        term2 *= c[num_stages - 1 - i] * 2 / 3 # 2 / 3 = b_S

        @NLconstraint(model, monomial_coeffs[i] - (term1 + term2)==0.0)
    end

    # Highest coefficient: Only one term present
    i = num_stage_evals - 3
    term2 = a_coeff[num_stage_evals]
    for j in 1:i
        term2 *= a_coeff[num_stage_evals - j]
    end
    term2 *= c[num_stages - 1 - i] * 2 / 3 # 2 / 3 = b_S

    @NLconstraint(model, monomial_coeffs[i] - term2==0.0)

    # Third-order consistency condition
    term1 = 4 * a_coeff[num_stage_evals] # Have to seperate this into two terms. Otherwise, JuMP throws an error with muladd.
    term2 = a_coeff[num_stage_evals - 1]
    @NLconstraint(model, 1 - term1 - term2==0.0)

    # Add nonlinear constraints ensuring that every a_unknown is non-negative
    for i in 1:(num_stage_evals - 2)
        @NLconstraint(model, a_unknown[i]>=0.0)
    end

    # Adding condition involving c and a_unknown (ensuring c[i] - a_unknown_value >= 0.0)
    for i in (num_stages - num_stage_evals + 3):(num_stage_evals - 2)
        @NLconstraint(model,
                      c[i] - a_unknown[i - (num_stage_evals - num_stages + 2)]>=0.0)
    end

    compute_b_embedded_coeffs(model, b_embedded, num_stage_evals, num_stages,
                              monomial_coeffs_embedded, a_unknown, c)

    for i in 1:(num_stage_evals - 1)
        @NLconstraint(model, b_embedded[i]>=0.0)
    end

    rng = StableRNG(555)  # Use Stable RNG for reproducibility

    # Optimize and find c
    for attempt in 1:max_iter
        if verbose
            println("Attempt $attempt with a new initial guess")
        end

        # Set random initial guess
        for i in 1:length(c_free)
            set_start_value(c_free[i], rand(rng) * 0.1)
        end

        for i in 1:length(a_unknown)
            set_start_value(a_unknown[i], rand(rng) * 0.1)
        end

        for i in 1:length(b_embedded)
            set_start_value(b_embedded[i], rand(rng) * 0.1)
        end

        # Solve the model
        optimize!(model)

        status = termination_status(model)
        if status == LOCALLY_SOLVED || status == OPTIMAL
            if verbose
                println("Solution found after $attempt attempts.")
            end
            c_values = [value.(c_free)..., 1.0f0, 0.5f0]
            a_unknown_values = value.(a_unknown)
            b_embedded_values = value.(b_embedded)

            return a_unknown_values, b_embedded_values, c_values
        end
    end

    error("Tough luck. No c. Oops")
end
end
end
