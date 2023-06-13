
using OrdinaryDiffEq, Plots, LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

PolyDegree = 3


surface_flux = flux_lax_friedrichs
volume_flux  = flux_central
basis = LobattoLegendreBasis(PolyDegree)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=false,
                                         variable=Trixi.scalar)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)


solver = DGSEM(polydeg=PolyDegree, surface_flux=flux_lax_friedrichs)

coordinates_min = -5.0 # minimum coordinate
coordinates_max =  5.0 # maximum coordinate

InitialRefinement = 6
# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                # Start from one cell => Results in 1 + 2 + 4 + 8 + 16 = 2^5 - 1 = 31 cells
                initial_refinement_level=InitialRefinement,
                n_cells_max=30_000) # set maximum capacity of tree data structure

initial_condition = initial_condition_gauss

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

#=
A = jacobian_ad_forward(semi)

Eigenvalues = eigvals(A)

# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

# Sometimes due to numerical issues some eigenvalues have positive real part, which is erronous (for hyperbolic eqs)
Eigenvalues = Eigenvalues[real(Eigenvalues) .< 0]

EigValsReal = real(Eigenvalues)
EigValsImag = imag(Eigenvalues)

plotdata = nothing
plotdata = scatter(EigValsReal, EigValsImag, label = "Spectrum")
display(plotdata)

EigValFile = "EigenvalueList.txt"
ofstream = open(EigValFile, "w")
for i in eachindex(Eigenvalues)
  realstring = string(EigValsReal[i])
  write(ofstream, realstring)

  write(ofstream, "+")

  imstring = string(EigValsImag[i])
  write(ofstream, imstring)
  write(ofstream, "i") # Cpp uses "I" for the imaginary unit
  if i != length(Eigenvalues)
    write(ofstream, "\n")
  end
end
close(ofstream)
=#

###############################################################################
# ODE solvers, callbacks etc.

StartTime = 0.0
EndTime = 10


# Create ODEProblem
ode = semidiscretize(semi, (StartTime, EndTime));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

Interval = 500
# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=Interval, extra_analysis_errors=(:conservation_error,))

# 4 Levels

amr_controller = ControllerThreeLevel(semi, 
                                      IndicatorMax(semi, variable=first),
                                      base_level=InitialRefinement,
                                      med_level=InitialRefinement+1, med_threshold=0.1, #0.1
                                      max_level=InitialRefinement+3, max_threshold=0.6) #0.6


# 3 Levels

amr_controller = ControllerThreeLevel(semi, 
                                      IndicatorMax(semi, variable=first),
                                      base_level=InitialRefinement,
                                      med_level=InitialRefinement+1, med_threshold=0.1, #0.1
                                      max_level=InitialRefinement+2, max_threshold=0.6) #0.6


amr_callback = AMRCallback(semi, amr_controller,
                           interval=Interval,
                           adapt_initial_condition=false) # Adaption of initial condition not yet supported

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, 
                        analysis_callback,
                        amr_callback)

#callbacks = CallbackSet(summary_callback, analysis_callback)

stepsize_callback = StepsizeCallback(cfl=0.5)                    
callbacksSSPRK22 = CallbackSet(summary_callback,
                               amr_callback,
                               analysis_callback,
                               stepsize_callback)         

###############################################################################
# run the simulation

# S_base = 4, Shock-Capturing
dtOptMin = 0.0842359527669032104 / (2.0^(InitialRefinement - 6))
CFL = 0.5

# S_base = 4, optimized for NO Shock-Capturing 
dtOptMin = 0.06823193651780457 / (2.0^(InitialRefinement - 6))
#CFL = 0.8
CFL = 1.0 # No Shock capturing turned on

ode_algorithm = PERK_Multi(4, 2, "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv_3rd/", 
                           1.0, 0.5)

sol = Trixi.solve(ode, ode_algorithm, dt = dtOptMin * CFL, save_everystep=false, callback=callbacks)

#=
sol = solve(ode, SSPRK22(),
            dt = 42.0,
            save_everystep=false, callback=callbacksSSPRK22);
=#

# Print the timer summary
summary_callback()

plot(sol, plot_title = EndTime)

pd = PlotData1D(sol)
plot!(getmesh(pd))