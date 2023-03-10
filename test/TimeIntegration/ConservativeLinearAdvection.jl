module ConservativeLinearAdvection

using Trixi
using Trixi: AbstractEquations, get_node_vars
import Trixi: varnames, default_analysis_integrals, flux, max_abs_speed_naive,
              have_nonconservative_terms

# Since there is not yet native support for variable coefficients, we use two
# variables: one for the basic unknown `u` and another one for the coefficient `a`
struct ConservativeLinearAdvectionEquation <: AbstractEquations{1 #= spatial dimension =#,
                                                                2 #= two variables (u,a) =#}
end

varnames(::typeof(cons2cons), ::ConservativeLinearAdvectionEquation) = ("scalar", "advection_velocity")

default_analysis_integrals(::ConservativeLinearAdvectionEquation) = ()


function flux(u, orientation, equation::ConservativeLinearAdvectionEquation) 
    return SVector(u[2] * u[1], 0) 
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, ::ConservativeLinearAdvectionEquation)
    _, advection_velocity_ll = u_ll
    _, advection_velocity_rr = u_rr

    return max(abs(advection_velocity_ll), abs(advection_velocity_rr))
end

have_nonconservative_terms(::ConservativeLinearAdvectionEquation) = Val(false)

end # module



# Create a simulation setup
import .ConservativeLinearAdvection
using Trixi
using OrdinaryDiffEq, Plots

equation = ConservativeLinearAdvection.ConservativeLinearAdvectionEquation()

function initial_condition_const(x, t, equation::ConservativeLinearAdvection.ConservativeLinearAdvectionEquation)
  scalar = 1
  advection_velocity = 2 + cos(x[1])
  SVector(scalar, advection_velocity)
end

# You can derive the exact solution for this setup using the method of characteristics
function initial_condition_sine(x, t, equation::ConservativeLinearAdvection.ConservativeLinearAdvectionEquation)
    x0 = -2 * atan(sqrt(3) * tan(sqrt(3) / 2 * t - atan(tan(x[1] / 2) / sqrt(3))))
    scalar = sin(x0)
    advection_velocity = 2 + cos(x[1])
    SVector(scalar, advection_velocity)
end

# Create a uniform mesh in 1D in the interval [-π, π] with periodic boundaries
mesh = TreeMesh(-Float64(π), Float64(π), # min/max coordinates
                initial_refinement_level=7, n_cells_max=10^4)

# Create a DGSEM solver with polynomials of degree `polydeg`
# Remember to pass a tuple of the form `(conservative_flux, nonconservative_flux)`
# as `surface_flux` and `volume_flux` when working with nonconservative terms

surface_flux = flux_lax_friedrichs
solver = DGSEM(polydeg=5, surface_flux=surface_flux)

# Setup the spatial semidiscretization containing all ingredients
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_const, solver)

# Create an ODE problem with given time span
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

# Set up some standard callbacks summarizing the simulation setup and computing
# errors of the numerical solution
summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval=1000)
callbacks = CallbackSet(summary_callback, analysis_callback);


# OrdinaryDiffEq's `solve` method evolves the solution in time and executes
# the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = 3e-3 / 2, save_everystep=false, callback = callbacks)

plot(sol)