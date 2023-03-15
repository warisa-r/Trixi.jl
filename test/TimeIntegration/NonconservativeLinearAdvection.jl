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
using OrdinaryDiffEq, LinearAlgebra, Plots

equation = NonconservativeLinearAdvection.NonconservativeLinearAdvectionEquation()

# You can derive the exact solution for this setup using the method of characteristics
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
solver = DGSEM(polydeg=0, surface_flux=surface_flux,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

# Setup the spatial semidiscretization containing all ingredients
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_sine, solver)

A = jacobian_ad_forward(semi)
Eigenvalues = eigvals(A)

# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

EigValsReal = real(Eigenvalues)
EigValsImag = imag(Eigenvalues)

plotdata = scatter(EigValsReal, EigValsImag, label = "Spectrum Start")

# Create an ODE problem with given time span
tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan);

# Set up some standard callbacks summarizing the simulation setup and computing
# errors of the numerical solution
summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval=10)
callbacks = CallbackSet(summary_callback, analysis_callback);


# OrdinaryDiffEq's `solve` method evolves the solution in time and executes
# the passed callbacks
#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt = 3e-3 / 2, save_everystep=false, callback = callbacks)
=#

NumStages = 4

NumStageRef = 4
dtRef = 0.0711939144157440751

#=
NumStageRef = 16
dtRef = 0.40301178302433982
=#

CFL = 0.9
dtOptMin = dtRef * (NumStages / NumStageRef) * CFL


#ode_algorithm = FE2S(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Adv_NonConstSpeed/120/")
ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/1D_Adv_NonConstSpeed/")

#=
ode_algorithm = PERK_Multi(NumStages, 0,
                           "/home/daniel/Desktop/git/MA/EigenspectraGeneration/Spectra/1D_Burgers_smooth_SourceTerms/")
=#

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dtOptMin,
                  save_everystep=false, callback=callbacks)

# Plot the numerical solution at the final time
#plot(sol)

#pd = PlotData1D(sol)
#plot!(getmesh(pd))

A = jacobian_ad_forward(semi, tspan[end], sol.u[end])
Eigenvalues = eigvals(A)

# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

EigValsReal = real(Eigenvalues)
EigValsImag = imag(Eigenvalues)

plotdata = scatter!(EigValsReal, EigValsImag, label = "Spectrum Final")
display(plotdata)