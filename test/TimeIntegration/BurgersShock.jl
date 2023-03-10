
using OrdinaryDiffEq, Plots, LinearAlgebra
using LazySets # For convex hull
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = InviscidBurgersEquation1D()

function initial_condition_sin(x, t, equation::InviscidBurgersEquation1D)
  return SVector(sin(x[1]))
end

initial_condition = initial_condition_sin

PolyDegree = 0
numerical_flux = flux_lax_friedrichs
solver = DGSEM(polydeg=PolyDegree, surface_flux=numerical_flux)

coordinates_min = 0.0
coordinates_max = 2 * pi
RefinementLevel = 9
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=RefinementLevel,
                n_cells_max=10_000)

# No source terms: Blow-up @ t=1
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.9)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors=(:l2_error_primitive,
                                                            :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution)
                        #stepsize_callback)


###############################################################################
# run the simulation


### NON-SMOOTH CASE (No sources) ###


dtRef = 0.0507316850264032865
NumStagesRef = 4


dtRef = 0.335539181598142041
NumStagesRef = 16


NumStages = 16

CFL = 0.75
dt = dtRef * NumStages / NumStagesRef * CFL

ode_algorithm = PERK(NumStages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/Burgers/")

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks);


#=                  
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=42, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
=#

summary_callback() # print the timer summary

plot(sol)
pd = PlotData1D(sol)
plot!(getmesh(pd))


A = jacobian_ad_forward(semi, tspan[1], sol.u[1])
Eigenvalues = eigvals(A)

# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

# Sometimes due to numerical issues some eigenvalues have positive real part, which is erronous (for hyperbolic eqs)
Eigenvalues = Eigenvalues[real(Eigenvalues) .< 0]

EigValsReal = real(Eigenvalues)
EigValsImag = imag(Eigenvalues)

for i = 2:length(sol.u)
  A = jacobian_ad_forward(semi, tspan[i], sol.u[i])
  Eigenvalues = eigvals(A)

  # Complex conjugate eigenvalues have same modulus
  Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

  # Sometimes due to numerical issues some eigenvalues have positive real part, which is erronous (for hyperbolic eqs)
  Eigenvalues = Eigenvalues[real(Eigenvalues) .< 0]

  append!(EigValsReal, real(Eigenvalues))
  append!(EigValsImag, imag(Eigenvalues))
end

plotdata = scatter(EigValsReal, EigValsImag, label = "Spectrum")
display(plotdata)

EigValFile = "EigenvalueList_Refined" * string(RefinementLevel) * ".txt"
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

### Convex Hull Section ###

# Add origin (0,0) to Eigenvalues to have a "nice" convex hull (no funny business around origin)
# CARE: Do not do this for hyperbolic diffusion!
append!(EigValsReal, 0.0)
append!(EigValsImag, 0.0)

# Tweak data into right format for convex hull
points = hcat(EigValsReal, EigValsImag) # dims: 2 x NumEigVals
points_t = permutedims(points, (2, 1)) # dims: NumEigVals x 2
points_t_vv = [points_t[:,i] for i in 1:size(points_t, 2)]

# Compute convex hull
ConvexHull = convex_hull(points_t_vv)
#plot!(plotdata, VPolygon(ConvexHull), alpha = 0.2, label = "Convex hull / polygon", legend = :outertop)

ConvexHull_real = []
ConvexHull_imag = []

for i in eachindex(ConvexHull)
  append!(ConvexHull_real, ConvexHull[i][1])
  append!(ConvexHull_imag, ConvexHull[i][2])
end


# Get rid of possible double zeros (closure of convex hull along real axis)
indicesZero = findall(x->x==0, ConvexHull_imag)
MinRealZeroImag = minimum(ConvexHull_real[indicesZero])
MaxRealZeroImag = maximum(ConvexHull_real[indicesZero])

indicesPlus = findall(x->x>0, ConvexHull_imag)
ConvexHull_real = ConvexHull_real[indicesPlus]
ConvexHull_imag = ConvexHull_imag[indicesPlus]

# Add zero endpoints (previously removed)
# Left end
append!(ConvexHull_real, MinRealZeroImag)
append!(ConvexHull_imag, 0.0)

# Right end
append!(ConvexHull_real, MaxRealZeroImag)
append!(ConvexHull_imag, 0.0)

# Sort vector in ascending manner
perm = sortperm(ConvexHull_real)
ConvexHull_real = ConvexHull_real[perm]
ConvexHull_imag = ConvexHull_imag[perm]

#ConvexHull_real_File = "Hull_" * string(NumCells) * "_real.txt"
ConvexHull_real_File = "Hull_" * string(RefinementLevel) * "_real.txt"
ofstream = open(ConvexHull_real_File, "w")
for i in eachindex(ConvexHull_real)
  str = string(ConvexHull_real[i])
  write(ofstream, str)
  if i != length(ConvexHull_real)
    write(ofstream, "\n")
  end
end
close(ofstream)

#ConvexHull_imag_File = "Hull_" * string(NumCells) * "_imag.txt"
ConvexHull_imag_File = "Hull_" * string(RefinementLevel) * "_imag.txt"
ofstream = open(ConvexHull_imag_File, "w")
for i in eachindex(ConvexHull_imag)
  str = string(ConvexHull_imag[i])
  write(ofstream, str)
  if i != length(ConvexHull_imag)
    write(ofstream, "\n")
  end
end
close(ofstream)


# Tweak convex hull into dataformat allowing plotting
ConvexHull_manip = hcat(ConvexHull_real, ConvexHull_imag)
ConvexHull_manip_t = permutedims(ConvexHull_manip, (2, 1)) # dims: NumEigVals x 2
ConvexHull_manip_t_vv = [ConvexHull_manip_t[:,i] for i in 1:size(ConvexHull_manip_t, 2)]

plot!(plotdata, VPolygon(ConvexHull_manip_t_vv), alpha=0.2) # If origin not added to spectrum
display(plotdata)

# Plot convex hull boundary points
plotdata = scatter!(ConvexHull_real, ConvexHull_imag, label = "Convex hull boundary points")
display(plotdata)