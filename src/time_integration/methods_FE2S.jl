# TODO: Currently hard-coded to second order accurate methods!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

function ReadInFile(FilePath::AbstractString, DataType::Type)
  @assert isfile(FilePath)
  Data = zeros(DataType, 0)
  open(FilePath, "r") do File
    while !eof(File)     
      LineContent = readline(File)     
      append!(Data, parse(DataType, LineContent))
    end
  end
  NumLines = length(Data)

  return NumLines, Data
end

function ComputeFE2S_Coefficients(StagesMin::Int, NumDoublings::Int, PathPseudoExtrema::AbstractString)

  ForwardEulerWeights = zeros(NumDoublings+1)
  StagesMax = -1
  if NumDoublings == 0
    @assert StagesMin % 2 == 0 "Expect even number of Runge-Kutta steges!"
    StagesMax = Int(StagesMin / 2 - 1)
  else
    StagesMax = StagesMin * 2^(NumDoublings-1) -1
  end

  a  = zeros(StagesMax, NumDoublings+1)
  b1 = zeros(StagesMax, NumDoublings+1)
  b2 = zeros(StagesMax, NumDoublings+1)

  ### Set RKM / Butcher Tableau parameters corresponding to Base-Case (Minimal number stages) ### 

  PathPureReal = PathPseudoExtrema * "PureReal" * string(StagesMin) * ".txt"
  NumPureReal, PureReal = ReadInFile(PathPureReal, Float64)

  @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
  @assert PureReal[1] <= -1.0 "Assume that pure-real pseudo-extremum is smaller then 1.0"
  ForwardEulerWeights[1] = -1.0 / PureReal[1]

  PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(StagesMin) * ".txt"
  NumTrueComplex, TrueComplex = ReadInFile(PathTrueComplex, ComplexF64)
  @assert NumTrueComplex == StagesMin / 2 - 1 "Assume that all but one pseudo-extremum are complex"

  # Sort ascending => ascending timesteps (real part is always negative)
  perm = sortperm(real.(TrueComplex))
  TrueComplex = TrueComplex[perm]

  # Find first element where a timestep would be greater 1 => Special treatment
  IndGreater1 = findfirst(x->real(x) > -1.0, TrueComplex)

  if IndGreater1 == nothing
    # Easy: All can use negated inverse root as timestep
    IndGreater1 = NumTrueComplex + 1
  end

  for i = 1:IndGreater1 - 1
    a[i, 1]  = -1.0 / real(TrueComplex[i])
    b1[i, 1] = -real(TrueComplex[i]) / (abs(TrueComplex[i]) .* abs.(TrueComplex[i]))
    b2[i, 1] = b1[i, 1]
  end

  # To avoid timesteps > 1, compute difference between current maximum timestep = a[IndGreater1 - 1] +  and 1.
  dtGreater1 = (1.0 - a[IndGreater1 - 1, 1]) / (NumTrueComplex - IndGreater1 + 1)

  # TODO: Fill only up to 0.5 for coupled schemes
  for i = IndGreater1:NumTrueComplex
    # Fill gap a[IndGreater1 - 1] to 1 equidistantly
    a[i, 1]  = a[IndGreater1 - 1, 1] + dtGreater1 * (i - IndGreater1 + 1)

    b2[i, 1] = 1.0 / (abs(TrueComplex[i]) * abs(TrueComplex[i]) * a[i, 1])
    b1[i, 1] = -2.0 * real(TrueComplex[i]) / (abs(TrueComplex[i]) * abs(TrueComplex[i])) - b2[i, 1]
  end

  # Combine distinct (!) timesteps of pure real and true complex roots
  c = vcat(ForwardEulerWeights[1], a[:, 1])

  ### Set RKM / Butcher Tableau parameters corresponding for higher stages ### 

  for i = 1:NumDoublings
    Degree = StagesMin * 2^i

    PathPureReal = PathPseudoExtrema * "PureReal" * string(Degree) * ".txt"
    NumPureReal, PureReal = ReadInFile(PathPureReal, Float64)

    @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
    @assert PureReal[1] <= -1.0 "Assume that pure-real pseudo-extremum is smaller then 1.0"
    ForwardEulerWeights[i+1] = -1.0 / PureReal[1]

    PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(Degree) * ".txt"
    NumTrueComplex, TrueComplex = ReadInFile(PathTrueComplex, Complex{Float64})
    @assert NumTrueComplex == Degree / 2 - 1 "Assume that all but one pseudo-extremum are complex"

    # Sort ascending => ascending timesteps (real part is always negative)
    perm = sortperm(real.(TrueComplex))
    TrueComplex = TrueComplex[perm]

    # Different (!) timesteps of higher degree RKM
    c_higher = zeros(Int(Degree / 2))
    c_higher[1] = ForwardEulerWeights[i+1]
    for j = 2:Int(Degree / 2)
      if j % 2 == 0 # Copy timestep from lower degree
        c_higher[j] = c[Int(j / 2)]
      else # Interpolate timestep
        c_higher[j] = c[Int((j-1)/2)] + 0.5 * (c[Int((j+1)/2)] - c[Int((j-1)/2)])
      end
    end
    c = copy(c_higher)

    for j = 1:length(c_higher) - 1
      a[j, i+1]  = c_higher[j+1]
      b2[j, i+1] = 1.0 / (abs(TrueComplex[j]) * abs(TrueComplex[j]) * a[j, i+1])
      b1[j, i+1] = -2.0 * real(TrueComplex[j]) / (abs(TrueComplex[j]) * abs(TrueComplex[j])) - b2[j, i+1]
    end
  end

  # Re-order columns to comply with the level structure of the mesh quantities
  perm = Vector(NumDoublings+1:-1:1)
  permute!(ForwardEulerWeights, perm)
  Base.permutecols!!(a, perm)

  perm = Vector(NumDoublings+1:-1:1) # 'permutecols!!' "deletes" perm it seems
  Base.permutecols!!(b1, perm)
  
  perm = Vector(NumDoublings+1:-1:1) # 'permutecols!!' "deletes" perm it seems
  Base.permutecols!!(b2, perm)

  println("ForwardEulerWeights:\n"); display(ForwardEulerWeights); println("\n")
  println("a:\n"); display(a); println("\n")
  println("b1:\n"); display(b1); println("\n")
  println("b2:\n"); display(b2); println("\n")
  println("c:\n"); display(c); println("\n")

  return ForwardEulerWeights, a, b1, b2, c
end

# Check stability of individual methods
function CheckStability(ForwardEulerWeight::Float64, a::Vector{Float64}, b1::Vector{Float64}, b2::Vector{Float64},
                        EigVals::Vector{<:Complex{<:Float64}}, dt::Float64)

  @assert length(a)  == length(b1)
  @assert length(b1) == length(b2)

  MaxAbsStabPnom = -1.0

  NumStableEigVals = 0
  for i in eachindex(EigVals)
    z = Complex(EigVals[i] * dt)

    StabPnom = 1.0;
    for j in eachindex(ForwardEulerWeight)
      StabPnom *= 1.0 + z * ForwardEulerWeight[j]
    end

    for j in eachindex(a)
      StabPnom *= 1.0 + z * ((b1[j] + b2[j]) + z * a[j] * b2[j])
    end

    StabPnom *= z
    StabPnom += 1.0

    if abs(StabPnom) - 1.0 >= eps(Float64)
    #if abs(StabPnom) > 1.0
      println(string(i) * "'th Eigenvalue constraint violates stability bound with value " * string(abs(StabPnom)))
      println("This is eigenvalue: ", EigVals[i], "\n")
    else # Stable eigenvalue
      if imag(z) <= 1e-12 # Eigenvalue is assumed real
        NumStableEigVals += 1
      else # Eigenvalue is complex-conjugated
        NumStableEigVals += 2
      end
    end

    if abs(StabPnom) > MaxAbsStabPnom
      MaxAbsStabPnom = abs(StabPnom)
    end
  end

  println("Number of stable Eigenvalues is: ", NumStableEigVals)

  return MaxAbsStabPnom
end

function ComputeAmpMatrix(A::Matrix{Float64}, dt::Float64, DistributionMatrix::Matrix{Int})

  dtApow = dt^2 .* A * A                          
  BasePnom2 = DistributionMatrix * I + dt .* A + 0.5 .* dtApow
  dtApow *= dt .* A
  LowDegreePnom  = BasePnom2 + 0.156149396314828 .* dtApow
  HighDegreePnom = BasePnom2 + 0.164081689164205 .* dtApow

  dtApow *= dt .* A
  LowDegreePnom  += 0.0308768369075866 .* dtApow
  HighDegreePnom += 0.0390225916568780 .* dtApow

  dtApow *= dt .* A
  LowDegreePnom  += 0.00365477959519221 .* dtApow
  HighDegreePnom += 0.00704232209800608 .* dtApow

  dtApow *= dt .* A
  LowDegreePnom  += 0.000201261605214267 .* dtApow
  HighDegreePnom += 0.000984618576386692 .* dtApow

  dtApow *= dt .* A
  HighDegreePnom += 0.000107132166432604 .* dtApow

  dtApow *= dt .* A
  HighDegreePnom += 8.98404284721041e-06 .* dtApow

  dtApow *= dt .* A
  HighDegreePnom += 5.65495237992257e-07 .* dtApow

  dtApow *= dt .* A
  HighDegreePnom += 2.53388343562130e-08 .* dtApow

  dtApow *= dt .* A
  HighDegreePnom += 7.25350370838813e-10 .* dtApow

  dtApow *= dt .* A
  HighDegreePnom += 1.00289907472715e-11 .* dtApow

  return HighDegreePnom, LowDegreePnom
end

function ComputeAmpMatrices(StagesMin::Int, NumDoublings::Int, PathPseudoExtrema::AbstractString, 
                            A::Matrix{Float64}, dt::Float64, DistributionMatrices)

  StagesMax = -1
  if NumDoublings == 0
    @assert StagesMin % 2 == 0 "Expect even number of Runge-Kutta steges!"
    StagesMax = Int(StagesMin / 2)
  else
    StagesMax = StagesMin * 2^(NumDoublings-1)
  end

  PseudoExtrema = zeros(ComplexF64, StagesMax, NumDoublings+1)

  PathPureReal = PathPseudoExtrema * "PureReal" * string(StagesMin) * ".txt"
  NumPureReal, PureReal = ReadInFile(PathPureReal, Float64)
  @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
  PseudoExtrema[1, 1] = PureReal[1]

  PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(StagesMin) * ".txt"
  NumTrueComplex, TrueComplex = ReadInFile(PathTrueComplex, ComplexF64)
  @assert NumTrueComplex == StagesMin / 2 - 1 "Assume that all but one pseudo-extremum are complex"
  PseudoExtrema[2:Int(StagesMin / 2), 1] = TrueComplex

  for i = 1:NumDoublings
    Degree = StagesMin * 2^i

    PathPureReal = PathPseudoExtrema * "PureReal" * string(Degree) * ".txt"
    NumPureReal, PureReal = ReadInFile(PathPureReal, Float64)
    @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
    PseudoExtrema[1, i+1] = PureReal[1]

    PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(Degree) * ".txt"
    NumTrueComplex, TrueComplex = ReadInFile(PathTrueComplex, Complex{Float64})
    @assert NumTrueComplex == Degree / 2 - 1 "Assume that all but one pseudo-extremum are complex"
    PseudoExtrema[2:Int(Degree / 2), i+1] = TrueComplex
  end

  # Keep same order as the datastructures in other files
  perm = Vector(NumDoublings+1:-1:1)
  Base.permutecols!!(PseudoExtrema, perm)
  #display(PseudoExtrema)

  AmpMatrices = zeros(NumDoublings+1, size(A, 1), size(A, 2))
  for i = 1:NumDoublings+1
    AmpMatrices[i, :, :] = dt .* DistributionMatrices[i, :, :] * A

    # Pure real pseudo-extrema
    AmpMatrices[i, :, :] *= (I - dt .* DistributionMatrices[i, :, :] * A ./ PseudoExtrema[1, i])

    # Complex conjugated pseudo-extrema
    for j = 2:Int((StagesMin/2) * 2^(NumDoublings + 1 - i))
      AmpMatrices[i, :, :] *= real((I - dt .* DistributionMatrices[i, :, :] * A ./ PseudoExtrema[j,i]) * 
                                   (I - dt .* DistributionMatrices[i, :, :] * A ./ conj(PseudoExtrema[j,i])))
    end

    AmpMatrices[i, :, :] += I
  end

  return AmpMatrices
end

function ComputeAmpMatrix(StagesMin::Int, NumDoublings::Int, PathPseudoExtrema::AbstractString, 
                          A::Matrix{Float64}, dt::Float64, DistributionMatrices)

  StagesMax = -1
  if NumDoublings == 0
    @assert StagesMin % 2 == 0 "Expect even number of Runge-Kutta steges!"
    StagesMax = Int(StagesMin / 2)
  else
    StagesMax = StagesMin * 2^(NumDoublings-1)
  end

  PseudoExtrema = zeros(ComplexF64, StagesMax, NumDoublings+1)

  PathPureReal = PathPseudoExtrema * "PureReal" * string(StagesMin) * ".txt"
  NumPureReal, PureReal = ReadInFile(PathPureReal, Float64)
  @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
  PseudoExtrema[1, 1] = PureReal[1]

  PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(StagesMin) * ".txt"
  NumTrueComplex, TrueComplex = ReadInFile(PathTrueComplex, ComplexF64)
  @assert NumTrueComplex == StagesMin / 2 - 1 "Assume that all but one pseudo-extremum are complex"
  PseudoExtrema[2:Int(StagesMin / 2), 1] = TrueComplex

  for i = 1:NumDoublings
    Degree = StagesMin * 2^i

    PathPureReal = PathPseudoExtrema * "PureReal" * string(Degree) * ".txt"
    NumPureReal, PureReal = ReadInFile(PathPureReal, Float64)
    @assert NumPureReal == 1 "Assume that there is only one pure real pseudo-extremum"
    PseudoExtrema[1, i+1] = PureReal[1]

    PathTrueComplex = PathPseudoExtrema * "TrueComplex" * string(Degree) * ".txt"
    NumTrueComplex, TrueComplex = ReadInFile(PathTrueComplex, Complex{Float64})
    @assert NumTrueComplex == Degree / 2 - 1 "Assume that all but one pseudo-extremum are complex"
    PseudoExtrema[2:Int(Degree / 2), i+1] = TrueComplex
  end

  # Keep same order as the datastructures in other files
  perm = Vector(NumDoublings+1:-1:1)
  Base.permutecols!!(PseudoExtrema, perm)
  #display(PseudoExtrema)

  # Perform first Forward Euler step on finest grid
  AmpMatrix = I - dt .* DistributionMatrices[1, :, :] * A ./ real(PseudoExtrema[1, 1])

  for step = 1:2^NumDoublings - 1
    # Intermediate "two-step" sub methods on finest level
    AmpMatrix *= real((I - dt .* DistributionMatrices[1, :, :] * A ./ PseudoExtrema[step + 1, 1]) * 
                      (I - dt .* DistributionMatrices[1, :, :] * A ./ conj(PseudoExtrema[step + 1, 1])))
    for coarse_level = 2:NumDoublings+1
      # Check if we can take a timestep on a coarser level
      if step % (2^(coarse_level - 1)) == 1
        if step == 2^(coarse_level - 1) - 1 # Do Forward Euler step for coarser levels
          AmpMatrix *= I - dt .* DistributionMatrices[coarse_level, :, :] * A ./ real(PseudoExtrema[1, coarse_level])
        else # Do intermediate "two-steps" for coarser levels (except for the coarsest, where we do only Forward Euler)
          pos = Int((step - 1) / 2^(coarse_level - 1)) + 1
          AmpMatrix *= real((I - dt .* DistributionMatrices[coarse_level, :, :] * A ./ PseudoExtrema[pos, coarse_level]) * 
                            (I - dt .* DistributionMatrices[coarse_level, :, :] * A ./ conj(PseudoExtrema[pos, coarse_level])))
        end
      end
    end
  end

  # Intermediate "two-step" sub methods
  for step = 2^NumDoublings:length(PseudoExtrema[:, 1]) - 1
    AmpMatrix *= real((I - dt .* DistributionMatrices[1, :, :] * A ./ PseudoExtrema[step + 1, 1]) * 
                      (I - dt .* DistributionMatrices[1, :, :] * A ./ conj(PseudoExtrema[step + 1, 1])))

    for coarse_level = 2:NumDoublings+1
      # Check if we can take a timestep on a coarser level
      if step % (2^(coarse_level - 1)) == 1 # Do only intermediate "two-steps" (no Forward Euler) from here on                      
        pos = Int((step - 1) / 2^(coarse_level - 1)) + 1

        AmpMatrix *= real((I - dt .* DistributionMatrices[coarse_level, :, :] * A ./ PseudoExtrema[pos, coarse_level]) * 
                          (I - dt .* DistributionMatrices[coarse_level, :, :] * A ./ conj(PseudoExtrema[pos, coarse_level])))
      end
    end
  end

  AmpMatrix *= dt .* A
  AmpMatrix += I

  return AmpMatrix
end

function CheckStabilityJointMethod(dtMax_::Float64, A::Matrix{Float64})

  DistributionMatrix = zeros(Int, 96, 96)
  for i = 8 * 4 + 1:96
    DistributionMatrix[i, i] = 1
  end

  dtMax = dtMax_
  dtMin = 0
  dtEps = 1e-9
  while dtMax - dtMin > dtEps
    dt = 0.5 * (dtMax + dtMin)

    #LowDegreePnom, HighDegreePnom = ComputeAmpMatrix(A, dt, I)
    #CombinedAmpMat = DistributionMatrix * HighDegreePnom + (I - DistributionMatrix) * LowDegreePnom

    HighDegreePnom, LowDegreePnom = ComputeAmpMatrix((I - DistributionMatrix) * A, dt, I - DistributionMatrix)
    HighDegreePnom, = ComputeAmpMatrix(DistributionMatrix * A, dt, DistributionMatrix)
    CombinedAmpMat = HighDegreePnom + LowDegreePnom

    eigs = eigvals(CombinedAmpMat)
    #eigs = eigs[real(eigs) .< 0] # Remove positive eigenvalues
    SpectralRadius = maximum(abs.(eigs))

    if SpectralRadius - 1.0 >= eps(Float64)
      dtMax = dt
    else
      dtMin = dt
    end
  end

  return dtMin
end

function CheckStabilityJointMethod(StagesMin::Int, NumDoublings::Int, PathPseudoExtrema::AbstractString, 
                                   A::Matrix{Float64}, dtMax_::Float64)

  DistributionMatrices = zeros(Int, NumDoublings + 1, 96, 96)
  for i = 8 * 4 + 1:96
    DistributionMatrices[1, i, i] = 1
  end
  for i = 1:8*4
    DistributionMatrices[2, i, i] = 1
  end

  dtMax = dtMax_
  #dtMax = 4.2
  dtMin = 0
  dtEps = 1e-9
  while dtMax - dtMin > dtEps
    dt = 0.5 * (dtMax + dtMin)

    #=
    AmpMatrices = ComputeAmpMatrices(StagesMin, NumDoublings, PathPseudoExtrema, A, dt, DistributionMatrices)
    CombinedAmpMat = DistributionMatrices[1, :, :] * AmpMatrices[1, :, :]
    for i = 2:NumDoublings+1
      CombinedAmpMat += DistributionMatrices[i, :, :] * AmpMatrices[i, :, :]
    end
    =#
    CombinedAmpMat = ComputeAmpMatrix(StagesMin, NumDoublings, PathPseudoExtrema, A, dt, DistributionMatrices)

    eigs = eigvals(CombinedAmpMat)
    #eigs = eigs[real(eigs) .< 0] # Remove positive eigenvalues
    SpectralRadius = maximum(abs.(eigs))

    if SpectralRadius - 1.0 >= eps(Float64)
      dtMax = dt
    else
      dtMin = dt
    end
    println(dt)
  end

  return dtMin
end

function FindMaxdtScaling(ForwardEulerWeight::Float64, a::Vector{Float64}, b1::Vector{Float64}, b2::Vector{Float64},
                          EigVals::Vector{<:Complex{<:Float64}}, dt::Float64)

  dtScaling = 1.0
  dtScalingMax = 1.0
  dtScalingMin = 0.0
  while dtScalingMax - dtScalingMin > 1e-12
    dtScaling = 0.5 * (dtScalingMax + dtScalingMin)
    if CheckStability(ForwardEulerWeight, a, b1, b2, EigVals, dt * dtScaling) - 1.0 >= eps(Float64)
    #if CheckStability(ForwardEulerWeight, a, b1, b2, EigVals, dt * dtScaling) - 1.0 > eps(Float64)
      dtScalingMax = dtScaling
    else
      dtScalingMin = dtScaling
    end
  end

  println("Scaling to get the exact minimum count of stable eigenvalues is: ", dtScalingMin)
  return dtScalingMin
end

### Based on file "methods_2N.jl", use this as a template for P-ERK RK methods

"""
    FE2S()

The following structures and methods provide a minimal implementation of
the 'ForwardEulerTwoStep' temporal integrator.

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{54, 43} methods.
"""

mutable struct FE2S
  # Reference = minimum number of stages
  StagesMin::Int
  # Determines how often one doubles the number of stages 
  # Number of methods = NumDoublings +1 
  NumDoublings::Int
  # Timestep corresponding to lowest number of stages
  dtOptMin::Real

  ForwardEulerWeights::Vector{Float64}
  a::Matrix{Float64}
  b1::Matrix{Float64}
  b2::Matrix{Float64}
  c::Vector{Float64}

  # Constructor for previously computed A Coeffs
  function FE2S(StagesMin_::Int, NumDoublings_::Int, dtOptMin_::Real, PathPseudoExtrema_::AbstractString, 
                EigVals_::Vector{<:ComplexF64}, 
                # TODO: A: Only for testing
                A_::Matrix{Float64})
    newFE2S = new(StagesMin_, NumDoublings_, dtOptMin_)

    newFE2S.ForwardEulerWeights, newFE2S.a, newFE2S.b1, newFE2S.b2, newFE2S.c = 
      ComputeFE2S_Coefficients(StagesMin_, NumDoublings_, PathPseudoExtrema_)

    #println("Stage 1 stability test")
    #CheckStability(newFE2S.ForwardEulerWeights[1], newFE2S.a[:, 1], newFE2S.b1[:, 1], newFE2S.b2[:, 1], EigVals_, dtOptMin_)
    if NumDoublings_ > 0
      #println("Stage 2 stability test")
      #CheckStability(newFE2S.ForwardEulerWeights[2], newFE2S.a[:, 2], newFE2S.b1[:, 2], newFE2S.b2[:, 2], EigVals_, dtOptMin_)

      #=
      newFE2S.dtOptMin *= FindMaxdtScaling(newFE2S.ForwardEulerWeights[2], newFE2S.a[:, 2], newFE2S.b1[:, 2], newFE2S.b2[:, 2], 
                                          EigVals_, dtOptMin_)
      println("Stable timestep is: ", newFE2S.dtOptMin)
      =#                                     

      println("Check joint stability, maximum possible timestep is:")
      println(CheckStabilityJointMethod(dtOptMin_, A_))

      println("Check joint stability, maximum possible timestep is:")
      println(CheckStabilityJointMethod(StagesMin_, NumDoublings_, PathPseudoExtrema_, A_, dtOptMin_))
    end

    return newFE2S
  end

end # struct FE2S


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct FE2S_IntegratorOptions{Callback}
  callback::Callback # callbacks; used in Trixi
  adaptive::Bool # whether the algorithm is adaptive; ignored
  dtmax::Float64 # ignored
  maxiters::Int # maximal numer of time steps
  tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function FE2S_IntegratorOptions(callback, tspan; maxiters=typemax(Int), kwargs...)
  FE2S_IntegratorOptions{typeof(callback)}(callback, false, Inf, maxiters, [last(tspan)])
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct FE2S_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, FE2S_IntegratorOptions}
  u::uType
  du::uType
  u_tmp::uType
  t::RealT
  dt::RealT # current time step
  dtcache::RealT # ignored
  iter::Int # current number of time steps (iteration)
  p::Params # will be the semidiscretization from Trixi
  sol::Sol # faked
  f::F # This is the ODE function from ODEProblem; within Trixi, this amounts to "rhs!"
  alg::Alg # This is our own class written above; Abbreviation for ALGorithm
  opts::FE2S_IntegratorOptions
  finalstep::Bool # added for convenience
end

# Forward integrator.destats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::FE2S_Integrator, field::Symbol)
  if field === :destats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::FE2S;
               dt::Real, callback=nothing, kwargs...)
  u0 = copy(ode.u0) # Initial value
  du = similar(u0)
  u_tmp = similar(u0)
  t0 = first(ode.tspan) # Initial time
  iter = 0 # We are in first iteration
  integrator = FE2S_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, 
                 ode.p, # the semidiscretization
                 (prob=ode,), # Not really sure whats going on here
                 ode.f, # the right-hand-side of the ODE u' = f(u, p, t)
                 alg, # The ODE integration algorithm/method
                 FE2S_IntegratorOptions(callback, ode.tspan; kwargs...), false)

  # initialize callbacks
  if callback isa CallbackSet
    for cb in callback.continuous_callbacks
      error("unsupported")
    end
    for cb in callback.discrete_callbacks
      cb.initialize(cb, integrator.u, integrator.t, integrator)
    end
  elseif !isnothing(callback)
    error("unsupported")
  end

  solve!(integrator)
end

function solve!(integrator::FE2S_Integrator)
  @unpack prob = integrator.sol
  @unpack alg = integrator
  t_end = last(prob.tspan) # Final time
  callbacks = integrator.opts.callback

  println("Start time integration.")

  integrator.finalstep = false
  @trixi_timeit timer() "main loop" while !integrator.finalstep
    if isnan(integrator.dt)
      error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end || isapprox(integrator.t + integrator.dt, t_end)
      integrator.dt = t_end - integrator.t
      terminate!(integrator)
    end

    # TODO: Multi-threaded execution as implemented for other integrators instead of vectorized operations
    @trixi_timeit timer() "Forward Euler Two Stage ODE integration step" begin
      t_stages = integrator.t .* ones(alg.NumDoublings + 1) # TODO: Make member variable?

      # Perform first Forward Euler step on finest grid
      integrator.f(integrator.du, integrator.u, prob.p, t_stages[1], 1) # du = k1

      # TODO: Update only relevant entries of u (that of the corresponding level)
      integrator.u_tmp .= integrator.u .+ alg.ForwardEulerWeights[1] .* integrator.dt .* integrator.du
      t_stages[1] += alg.ForwardEulerWeights[1] * integrator.dt

      # Intermediate "two-step" sub methods on finest level
      for step = 1:2^alg.NumDoublings - 1

        # Low-storage implementation (only one k = du):
        integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stages[1], 1) # du = k1
        integrator.u_tmp .+= integrator.dt .* alg.b1[step, 1] .* integrator.du

        # FIXME Not sure if this is the right way to do the time steps
        t_stages[1] = integrator.t + alg.a[step, 1] * integrator.dt
        integrator.f(integrator.du, integrator.u_tmp .+ integrator.dt .*(alg.a[step, 1] - alg.b1[step, 1]) .* integrator.du, 
                     prob.p, t_stages[1], 1) # du = k2
        integrator.u_tmp .+= integrator.dt .* alg.b2[step, 1] .* integrator.du

        for coarse_level = 2:alg.NumDoublings+1
          # Check if we can take a timestep on a coarser level
          if step % (2^(coarse_level - 1)) == 1
            if step == 2^(coarse_level - 1) - 1 # Do Forward Euler step for coarser levels
              integrator.f(integrator.du, integrator.u, prob.p, t_stages[coarse_level], coarse_level) # du = k1
              integrator.u_tmp .+= alg.ForwardEulerWeights[coarse_level] .* integrator.dt .* integrator.du

              # FIXME Not sure if this is the right way to do the time steps
              t_stages[coarse_level] = integrator.t + alg.ForwardEulerWeights[coarse_level] * integrator.dt
            else # Do intermediate "two-steps" for coarser levels (except for the coarsest, where we do only Forward Euler)
              println("This is erronous!")
              # TODO: This will currently not work, we evolve from wrong u_tmp!
              pos = Int(step / 2^(coarse_level - 1))
              integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stages[coarse_level], coarse_level) # du = k1
              integrator.u_tmp .+= integrator.dt .* alg.b1[pos, coarse_level] .* integrator.du

              # NOTE: Not sure if this is the right way to do the time steps
              t_stages[coarse_level] = integrator.t + alg.a[pos, coarse_level] * integrator.dt

              integrator.f(integrator.du, 
                integrator.u_tmp .+ integrator.dt .*(alg.a[pos, coarse_level] - alg.b1[pos, coarse_level]) .* integrator.du, 
                prob.p, t_stages[coarse_level], coarse_level) # du = k2

              integrator.u_tmp .+= integrator.dt .* alg.b2[pos, coarse_level] .* integrator.du
            end
          end
        end
      end

      u_prev_timestep = 42 # Just declare this

      # Intermediate "two-step" sub methods
      for step = 2^alg.NumDoublings:length(alg.a[:, 1])
        # TODO: This works currently only for two levels, need probably to store as many solutions (in fact only neighboring cells)
        # As we have refinement levels
        if step % 2 == 0 # Save u_tmp every second timestep
          u_prev_timestep = copy(integrator.u_tmp)
        end

        # Low-storage implementation (only one k = du):
        integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stages[1], 1) # du = k1

        integrator.u_tmp .+= integrator.dt .* alg.b1[step, 1] .* integrator.du

        t_stages[1] = integrator.t + alg.a[step, 1] * integrator.dt
        integrator.f(integrator.du, integrator.u_tmp .+ integrator.dt .*(alg.a[step, 1] - alg.b1[step, 1]) .* integrator.du, 
                     prob.p, t_stages[1], 1) # du = k2
        integrator.u_tmp .+= integrator.dt .* alg.b2[step, 1] .* integrator.du

        for coarse_level = 2:alg.NumDoublings+1
          # Check if we can take a timestep on a coarser level
          if step % (2^(coarse_level - 1)) == 1 # Do only intermediate "two-steps" (no Forward Euler) from here on
            pos = Int((step - 1) / 2^(coarse_level - 1))
            #integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stages[coarse_level], coarse_level) # du = k1
            #integrator.u_tmp .+= integrator.dt .* alg.b1[pos, coarse_level] .* integrator.du

            integrator.f(integrator.du, u_prev_timestep, prob.p, t_stages[coarse_level], coarse_level) # du = k1
            u_prev_timestep .+= integrator.dt .* alg.b1[pos, coarse_level] .* integrator.du

            # NOTE: Not sure if this is the right way to do the time steps
            t_stages[coarse_level] = integrator.t + alg.a[pos, coarse_level] * integrator.dt

            #=
            integrator.f(integrator.du, 
              integrator.u_tmp .+ integrator.dt .*(alg.a[pos, coarse_level] - alg.b1[pos, coarse_level]) .* integrator.du, 
              prob.p, t_stages[coarse_level], coarse_level) # du = k2

            integrator.u_tmp .+= integrator.dt .* alg.b2[pos, coarse_level] .* integrator.du
            =#

            integrator.f(integrator.du, 
              u_prev_timestep .+ integrator.dt .*(alg.a[pos, coarse_level] - alg.b1[pos, coarse_level]) .* integrator.du, 
              prob.p, t_stages[coarse_level], coarse_level) # du = k2

            integrator.u_tmp[1:32] = u_prev_timestep[1:32] + integrator.dt .* alg.b2[pos, coarse_level] .* integrator.du[1:32]
          end
        end
      end
      # Final Euler step with step length of dt
      # TODO: Remove assert later (performance)
      @assert all(y->y == t_stages[1], t_stages) # Expect that every stage is at the same time by now ...
      # ... Then, we can use this time to develop from
      integrator.f(integrator.du, integrator.u_tmp, prob.p, t_stages[1]) # k1
      integrator.u .+= integrator.dt .* integrator.du
    end # FE2S step

    integrator.iter += 1
    integrator.t += integrator.dt

    # handle callbacks
    if callbacks isa CallbackSet
      for cb in callbacks.discrete_callbacks
        if cb.condition(integrator.u, integrator.t, integrator)
          cb.affect!(integrator)
        end
      end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
      @warn "Interrupted. Larger maxiters is needed."
      terminate!(integrator)
    end
  end

  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u),
                                integrator.sol.prob)
end

# get a cache where the RHS can be stored
get_du(integrator::FE2S_Integrator) = integrator.du
get_tmp_cache(integrator::FE2S_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::FE2S_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::FE2S_Integrator, dt)
  integrator.dt = dt
end

# stop the time integration
function terminate!(integrator::FE2S_Integrator)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::FE2S_Integrator, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.u_tmp, new_size)
end

end # @muladd