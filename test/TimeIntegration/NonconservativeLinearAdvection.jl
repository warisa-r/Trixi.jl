module NonconservativeLinearAdvection

using Trixi
using Trixi: AbstractEquations, get_node_vars
import Trixi: varnames, default_analysis_integrals, flux, max_abs_speed_naive,
              have_nonconservative_terms

# Since there is not yet native support for variable coefficients, we use two
# variables: one for the basic unknown `u` and another one for the coefficient `a`
struct NonconservativeLinearAdvectionEquation <: AbstractEquations{1 #= spatial dimension =#,
                                                                   2 #= two variables (u,a) =#}
end

varnames(::typeof(cons2cons), ::NonconservativeLinearAdvectionEquation) = ("scalar", "advection_velocity")

default_analysis_integrals(::NonconservativeLinearAdvectionEquation) = ()


# The conservative part of the flux is zero
flux(u, orientation, equation::NonconservativeLinearAdvectionEquation) = zero(u)

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, ::NonconservativeLinearAdvectionEquation)
    _, advection_velocity_ll = u_ll
    _, advection_velocity_rr = u_rr

    return max(abs(advection_velocity_ll), abs(advection_velocity_rr))
end


# We use nonconservative terms
have_nonconservative_terms(::NonconservativeLinearAdvectionEquation) = Val(true)

# This "nonconservative numerical flux" implements the nonconservative terms.
# In general, nonconservative terms can be written in the form
#   g(u) ∂ₓ h(u)
# Thus, a discrete difference approximation of this nonconservative term needs
# - `u mine`:  the value of `u` at the current position (for g(u))
# - `u_other`: the values of `u` in a neighborhood of the current position (for ∂ₓ h(u))
function flux_nonconservative(u_mine, u_other, orientation,
                              equations::NonconservativeLinearAdvectionEquation)
    _, advection_velocity = u_mine
    scalar, _            = u_other

    return SVector(advection_velocity * scalar, zero(scalar))
end

end # module



# Create a simulation setup
import .NonconservativeLinearAdvection
using Trixi
using OrdinaryDiffEq

equation = NonconservativeLinearAdvection.NonconservativeLinearAdvectionEquation()

# You can derive the exact solution for this setup using the method of
# characteristics
function initial_condition_sine(x, t, equation::NonconservativeLinearAdvection.NonconservativeLinearAdvectionEquation)
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
vol_flux = flux_central # Usually symmetric fluxes required
#vol_flux = flux_lax_friedrichs
volume_flux  = (vol_flux, NonconservativeLinearAdvection.flux_nonconservative)
surface_flux = (flux_lax_friedrichs, NonconservativeLinearAdvection.flux_nonconservative)
solver = DGSEM(polydeg=5, surface_flux=surface_flux,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

# Setup the spatial semidiscretization containing all ingredients
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_sine, solver)

# Create an ODE problem with given time span
tspan = (0.0, 200.0)
ode = semidiscretize(semi, tspan);

# Set up some standard callbacks summarizing the simulation setup and computing
# errors of the numerical solution
summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval=1000)
callbacks = CallbackSet(summary_callback, analysis_callback);


# OrdinaryDiffEq's `solve` method evolves the solution in time and executes
# the passed callbacks
#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = 3e-3 / 2, save_everystep=false, callback = callbacks)
=#

NumStages = 120

NumStageRef = 16
dtRef = 0.0171722409308131328

# Classic RK
CFL = 0.97 # Edge case for 30 stages
CFL = 0.81 # Edge case for 60 stages
CFL = 0.46 # Edge case for 120 stages

# FE2S with k = 0
CFL = 0.01 # Edge case for 30 stages
CFL = 0.004 # Edge case for 60 stages
CFL = 0.002 # Edge case for 120 stages

# FE2S with k = 1
#CFL = 0.12 # Edge case for 30 stages
#CFL = 0.06 # Edge case for 60 stages
#CFL = 0.03 # Edge case for 120 stages

# FE2S with k = 2
#CFL = 0.003 # Edge case for 30 stages
#CFL = 0.06 # Edge case for 60 stages
#CFL = 0.03 # Edge case for 120 stages

# FE2S with opt eta
#CFL = 0.01 # Edge case for 30 stages
#CFL = 0.02 # Edge case for 60 stages
#CFL = 0.004 # Edge case for 120 stages


# FE2S with RK-inspired substages
CFL = 0.99 # Edge case for 30 stages
#CFL = 0.86  # Edge case for 60 stages
#CFL = 0.35  # Edge case for 120 stages


# Lebedev way
CFL = 0.93 # Edge case for 30 stages
CFL = 0.72  # Edge case for 60 stages
CFL = 0.46  # Edge case for 120 stages



dtOptMin = dtRef * (NumStages / NumStageRef) * CFL

# 4 stages
dtOptMin = 0.00383180375676602133

# 30 stages
CFL = 0.95
dtOptMin = 0.031436587387667768 * CFL

# 60 stages
CFL = 0.75
dtOptMin = 0.066430373464476819 * CFL

# 120 stages
CFL = 0.62
dtOptMin = 0.128857 * CFL

ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Adv_NonConstSpeed/7/120/")
#ode_algorithm = PERK(4, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Adv_NonConstSpeed/7/")

#=
ode_algorithm = PERK_Multi(NumStages, 0,
                           "/home/daniel/Desktop/git/MA/EigenspectraGeneration/Spectra/1D_Burgers_smooth_SourceTerms/")
=#

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks)

# Plot the numerical solution at the final time
using Plots: plot
plot(sol)