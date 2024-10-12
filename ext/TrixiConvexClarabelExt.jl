# Package extension for adding Convex-based features to Trixi.jl
module TrixiConvexClarabelExt

# Required for coefficient optimization in P-ERK scheme integrators
if isdefined(Base, :get_extension)
    using Convex: MOI, solve!, Variable, minimize, evaluate
    using Clarabel: Optimizer
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..Convex: MOI, solve!, Variable, minimize, evaluate
    using ..Clarabel: Optimizer
end

# Use functions that are to be extended and additional symbols that are not exported
using Trixi: Trixi, @muladd

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# New version of stability polynomial of the embedded scheme
function stability_polynomials()
    #TODO: Implement the stability polynomial of the embedded scheme
end

function bisect_stability_polynomial()
    #TODO: Implement the bisect_stability_polynomial function. This should either be called by the function below or be implemented there(better).
    # the names of the functions here just correspond with the original code better
end

function Trixi.solve_b_butcher_coeffs_unknown(num_stages, a_matrix, c, dt_opt,eig_vals; verbose)
    #TODO: Implement this
end
end # @muladd

end # module TrixiConvexClarabelExt
