module Interpolation

using GaussQuadrature: legendre, both, neither

export interpolate_nodes
export calc_dhat
export calc_dsplit
export polynomial_derivative_matrix
export polynomial_interpolation_matrix
export barycentric_weights
export calc_lhat
export lagrange_interpolating_polynomials
export gauss_lobatto_nodes_weights
export vandermonde_legendre
export modal2nodal


# Interpolate data using the given Vandermonde matrix and return interpolated values (1D version).
function interpolate_nodes(data_in::AbstractArray{T, 2},
                           vandermonde::AbstractArray{T, 2}, n_vars::Integer) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in = size(vandermonde, 2)
  data_out = zeros(eltype(data_in), n_vars, n_nodes_out)

  for i = 1:n_nodes_out
    for ii = 1:n_nodes_in
      for v = 1:n_vars
        data_out[v, i] += vandermonde[i, ii] * data_in[v, ii]
      end
    end
  end

  return data_out
end


# Interpolate data using the given Vandermonde matrix and return interpolated values (2D version).
function interpolate_nodes(data_in::AbstractArray{T, 3},
                           vandermonde::AbstractArray{T, 2}, n_vars::Integer) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in = size(vandermonde, 2)

  # Interpolate in x-direction
  temp = zeros(eltype(data_in), n_vars, n_nodes_out, n_nodes_in)
  for i = 1:n_nodes_out
    for j = 1:n_nodes_in
      for ii = 1:n_nodes_in
        for v = 1:n_vars
          temp[v, i, j] += vandermonde[i, ii] * data_in[v, ii, j]
        end
      end
    end
  end

  # Interpolate in y-direction
  data_out = zeros(eltype(data_in), n_vars, n_nodes_out, n_nodes_out)
  for i = 1:n_nodes_out
    for j = 1:n_nodes_out
      for jj = 1:n_nodes_in
        for v = 1:n_vars
          data_out[v, i, j] += vandermonde[j, jj] * temp[v, i, jj]
        end
      end
    end
  end

  return data_out
end


# Calculate the Dhat matrix
function calc_dhat(nodes, weights)
  n_nodes = length(nodes)
  dhat = polynomial_derivative_matrix(nodes)
  dhat = transpose(dhat)

  for n = 1:n_nodes, j = 1:n_nodes
    dhat[j, n] *= -weights[n] / weights[j]
  end

  return dhat
end


# Calculate the Dsplit matrix for split-form differentiation: dplit = 2D - M⁻¹B
function calc_dsplit(nodes, weights)
  # Start with 2 x the normal D matrix
  dsplit = polynomial_derivative_matrix(nodes)
  dsplit = 2 .* dsplit

  # Modify to account for 
  dsplit[1, 1] += 1/weights[1]
  dsplit[end, end] -= 1/weights[end]

  return dsplit
end


# Calculate the polynomial derivative matrix D
function polynomial_derivative_matrix(nodes)
  n_nodes = length(nodes)
  d = zeros(n_nodes, n_nodes)
  wbary = barycentric_weights(nodes)

  for i = 1:n_nodes, j = 1:n_nodes
    if j != i
      d[i, j] = wbary[j] / wbary[i] * 1 / (nodes[i] - nodes[j])
      d[i, i] -= d[i, j]
    end
  end

  return d
end


# Calculate and interpolation matrix (Vandermonde matrix) between two given sets of nodes
function polynomial_interpolation_matrix(nodes_in, nodes_out)
  n_nodes_in = length(nodes_in)
  n_nodes_out = length(nodes_out)
  wbary_in = barycentric_weights(nodes_in)
  vdm = zeros(n_nodes_out, n_nodes_in)

  for k = 1:n_nodes_out
    match = false
    for j = 1:n_nodes_in
      if isapprox(nodes_out[k], nodes_in[j], rtol=eps())
        match = true
        vdm[k, j] = 1
      end
    end

    if match == false
      s = 0.0
      for j = 1:n_nodes_in
        t = wbary_in[j] / (nodes_out[k] - nodes_in[j])
        vdm[k, j] = t
        s += t
      end
      for j = 1:n_nodes_in
        vdm[k, j] = vdm[k, j] / s
      end
    end
  end

  return vdm
end


# Calculate the barycentric weights for a given node distribution.
function barycentric_weights(nodes)
  n_nodes = length(nodes)
  weights = ones(n_nodes)

  for j = 2:n_nodes, k = 1:(j-1)
    weights[k] *= nodes[k] - nodes[j]
    weights[j] *= nodes[j] - nodes[k]
  end

  for j = 1:n_nodes
    weights[j] = 1 / weights[j]
  end

  return weights
end


# Calculate Lhat.
function calc_lhat(x::Float64, nodes, weights)
  n_nodes = length(nodes)
  wbary = barycentric_weights(nodes)

  lhat = lagrange_interpolating_polynomials(x, nodes, wbary)

  for i = 1:n_nodes
    lhat[i] /= weights[i]
  end

  return lhat
end


# Calculate Lagrange polynomials for a given node distribution.
function lagrange_interpolating_polynomials(x::Float64, nodes, wbary)
  n_nodes = length(nodes)
  polynomials = zeros(n_nodes)

  for i = 1:n_nodes
    if isapprox(x, nodes[i], rtol=eps(x))
      polynomials[i] = 1
      return polynomials
    end
  end

  for i = 1:n_nodes
    polynomials[i] = wbary[i] / (x - nodes[i])
  end
  total = sum(polynomials)

  for i = 1:n_nodes
    polynomials[i] /= total
  end

  return polynomials
end


# Calculate nodes and weights for Legendre-Gauss-Lobatto quadratue.
function gauss_lobatto_nodes_weights(n_nodes::Integer)
  return legendre(n_nodes, both)
end


# Calculate nodes and weights for Legendre-Gauss quadratue.
function gauss_nodes_weights(n_nodes::Integer)
  return legendre(n_nodes, neither)
end


# From FLUXO (but really from blue book by Kopriva)
function legendre_polynomial_and_derivative(N::Int, x::Real)
  if N == 0
    poly = 1.0
    deriv = 0.0
  elseif N == 1
    poly = convert(Float64, x)
    deriv = 1.0
  else
    poly_Nm2 = 1.0
    poly_Nm1 = convert(Float64, x)
    deriv_Nm2 = 0.0
    deriv_Nm1 = 1.0

    poly = 0.0
    deriv = 0.0
    for i in 2:N
      poly = ((2*i-1) * x * poly_Nm1 - (i-1) * poly_Nm2) / i
      deriv=deriv_Nm2 + (2*i-1)*poly_Nm1
      poly_Nm2=poly_Nm1
      poly_Nm1=poly
      deriv_Nm2=deriv_Nm1
      deriv_Nm1=deriv
    end
  end

  # Normalize
  poly = poly * sqrt(N+0.5)
  deriv = deriv * sqrt(N+0.5)

  return poly, deriv
end


# Calculate Legendre vandermonde matrix and its inverse
function vandermonde_legendre(nodes, N)
  n_nodes = length(nodes)
  n_modes = N + 1
  vandermonde = zeros(n_nodes, n_modes)

  for i in 1:n_nodes
    for m in 1:n_modes
      vandermonde[i, m], _ = legendre_polynomial_and_derivative(m-1, nodes[i])
    end
  end
  # for very high polynomial degree, this is not well conditioned    
  inverse_vandermonde = inv(vandermonde)
  return vandermonde, inverse_vandermonde
end
vandermonde_legendre(nodes) = vandermonde_legendre(nodes, length(nodes) - 1)


# Convert nodal to modal representation
function nodal2modal(data_in::AbstractArray{Float64, 3},
                     vandermonde::AbstractArray{Float64, 2})
  interpolate_nodes(data_in, vandermonde, 1)
end


end
