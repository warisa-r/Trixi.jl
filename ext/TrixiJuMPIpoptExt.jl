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
function PairedExplicitRK3_butcher_tableau_residual(a_unknown, num_stages, num_stage_evals, monomial_coeffs, c)
    c_ts = c # ts = timestep
    # For explicit methods, a_{1,1} = 0 and a_{2,1} = c_2 (Butcher's condition)
    a_coeff = [0, c_ts[2], a_unknown...]

    residual = 0.0
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
        term1 *= c_ts[num_stages - 2 - i] * 1 / 6 # 1 / 6 = b_{S-1}
        term2 *= c_ts[num_stages - 1 - i] * 2 / 3 # 2 / 3 = b_S

        residual += (monomial_coeffs[i] - (term1 + term2))^2
    end

    # Highest coefficient: Only one term present
    i = num_stage_evals - 3
    term2 = a_coeff[num_stage_evals]
    for j in 1:i
        term2 *= a_coeff[num_stage_evals - j]
    end
    term2 *= c_ts[num_stages - 1 - i] * 2 / 3 # 2 / 3 = b_S
    residual += (monomial_coeffs[i] - term2)^2

    # Third-order consistency condition
    residual += (1 - 4 * a_coeff[num_stage_evals] - a_coeff[num_stage_evals - 1])^2
end

function Trixi.solve_a_butcher_coeffs_with_JuMP(num_stages::Int,
                                                monomial_coeffs::Vector{T},
                                                c::Vector{T};
                                                verbose::Bool = false,
                                                max_iter::Int = 100000) where {T <:
                                                                               Real}
    # JuMP model setup
    model = Model(Ipopt.Optimizer)

    num_stage_evals = num_stages

    # Initial guess for a_unknown
    rng = StableRNG(555)
    x0 = [0.1 * rand(rng) for _ in 1:(num_stage_evals - 2)]

    # Add variables for the unknowns (a_unknown coefficients) with non-negativity constraint
    a_unknown = @variable(model, [i in 1:(num_stage_evals - 2)], lower_bound=0.0)

    # Define the objective function directly using the residual function
    @objective(model, Min, PairedExplicitRK3_butcher_tableau_residual(a_unknown, num_stages, num_stage_evals, monomial_coeffs, c))

    # Add nonlinear constraints
    for i in 1:(num_stage_evals - 2)
        @constraint(model, a_unknown[i] >= 0.0)  # Ensure a_unknown is non-negative
    end

    # Adding condition involving c and a_unknown (ensuring c[i] - a_unknown_value >= 0.0)
    for i in (num_stage_evals - num_stages + 3):(num_stage_evals - 2)
        @constraint(model, c[i] - a_unknown[i - (num_stage_evals - num_stages + 2)] >= 0.0)
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

        # Check if the solution meets the criteria
        a_unknown_values = value.(a_unknown)

        #Turn aUnknown into butcher tableau
        A = zeros(Float64, num_stages, num_stages)
        A[:, 1] .= c
        k = 1
        for i in (num_stages-num_stage_evals+3):num_stages
            A[i, 1] -= a_unknown_values[k]
            A[i, i-1] = a_unknown_values[k]
            k += 1
        end

        b1 = 1/6
        bS = 4/6
        bS1 = 1/6
        b = zeros(Float64, num_stages)
        b[1] = b1
        b[num_stages - 1] = bS1
        b[num_stages] = bS

        println(sum(b))
        println(dot(b,c))
        println(dot(b , c.^2))
        println(dot(b ,A * c))

        if termination_status(model) == OPTIMAL || termination_status(model) == ALMOST_OPTIMAL
            if verbose
                println("Valid solution found on attempt $attempt.")
            end
            return a_unknown_values
        else
            println("Solution invalid on attempt $attempt. Retrying...")
        end
    end

    error("Unable to find a valid solution after $max_iter attempts.")

    #a_unknown_values = value.(a_unknown)
    #return a_unknown_values
end
end
end
