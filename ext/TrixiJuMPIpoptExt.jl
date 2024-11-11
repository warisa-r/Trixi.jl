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

        @NLconstraint(model, monomial_coeffs[i] - (term1 + term2) == 0.0)
    end

    # Highest coefficient: Only one term present
    i = num_stage_evals - 3
    term2 = a_coeff[num_stage_evals]
    for j in 1:i
        term2 *= a_coeff[num_stage_evals - j]
    end
    term2 *= c[num_stages - 1 - i] * 2 / 3 # 2 / 3 = b_S

    @NLconstraint(model, monomial_coeffs[i] - term2 == 0.0)

    # Third-order consistency condition
    term1 = 4 * a_coeff[num_stage_evals] # Have to seperate this into two terms. Otherwise, JuMP throws an error with muladd.
    term2 = a_coeff[num_stage_evals - 1]
    @NLconstraint(model, 1 - term1 - term2 == 0.0)

    # Add nonlinear constraints ensuring that every a_unknown is non-negative
    for i in 1:(num_stage_evals - 2)
        @NLconstraint(model, a_unknown[i] >= 0.0)
    end

    # Adding condition involving c and a_unknown (ensuring c[i] - a_unknown_value >= 0.0)
    for i in (num_stages - num_stage_evals + 3):(num_stage_evals - 2)
        @NLconstraint(model, c[i] - a_unknown[i - num_stages + num_stage_evals - 2] >= 0.0)
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
            return value.(a_unknown)
        end
    end

    error("Unable to find a valid solution after $max_iter attempts.")
end
end
end
