# Author: Arpit Babbar https://gist.github.com/Arpit-Babbar/a5ecb5435fa063066217654b561d3212#file-my_analysis-jl-L209

using DelimitedFiles

abstract type SurfaceQuantitiyViscous end

struct SaveSurfacePrimitives{Indices}
    indices::Indices
end

struct AnalysisSurfaceIntegral{Indices, Variable}
    indices::Indices
    variable::Variable
end

struct AnalysisSurfaceFrictionCoefficient{Indices, FreeStreamVariables} <:
       SurfaceQuantitiyViscous
    indices::Indices
    free_stream_variables::FreeStreamVariables
end

struct AnalysisSurfaceIntegralViscous{Indices, Variable} <: SurfaceQuantitiyViscous
    indices::Indices
    variable::Variable
end

# WARNING - This must be done before AnalysisSurfaceIntegralViscous as
# AnalysisSurfaceIntegralViscous will overwrite the gradient!
struct AnalysisSurfaceIntegralViscousCorrectedGrad{Indices, Variable} <:
       SurfaceQuantitiyViscous
    indices::Indices
    variable::Variable
end

struct ForceState{RealT <: Real}
    Ψl::Tuple{RealT, RealT}
    rhoinf::RealT
    uinf::RealT
    linf::RealT
end

# TODO - This should be a struct in ForceState
struct FreeStreamVariables{RealT <: Real}
    rhoinf::RealT
    uinf::RealT
    linf::RealT
end

struct LiftForcePressure{RealT <: Real}
    force_state::ForceState{RealT}
end

struct LiftForceViscous{RealT <: Real}
    force_state::ForceState{RealT}
end

struct DragForcePressure{RealT <: Real}
    force_state::ForceState{RealT}
end

struct DragForceViscous{RealT <: Real}
    force_state::ForceState{RealT}
end

function LiftForcePressure(aoa::Real, rhoinf::Real, uinf::Real, linf::Real)
    # Ψl is the normal unit vector to the freestream direction
    Ψl = (-sin(aoa), cos(aoa))
    force_state = ForceState(Ψl, rhoinf, uinf, linf)
    return LiftForcePressure(force_state)
end

function DragForcePressure(aoa::Real, rhoinf::Real, uinf::Real, linf::Real)
    # Ψd is the unit vector parallel to the freestream direction
    Ψd = (cos(aoa), sin(aoa))
    return DragForcePressure(ForceState(Ψd, rhoinf, uinf, linf))
end

function LiftForceViscous(aoa::Real, rhoinf::Real, uinf::Real, linf::Real)
    # Ψl is the normal unit vector to the freestream direction
    Ψl = (-sin(aoa), cos(aoa))
    force_state = ForceState(Ψl, rhoinf, uinf, linf)
    return LiftForceViscous(force_state)
end

function DragForceViscous(aoa::Real, rhoinf::Real, uinf::Real, linf::Real)
    # Ψd is the unit vector parallel to the freestream direction
    Ψd = (cos(aoa), sin(aoa))
    return DragForceViscous(ForceState(Ψd, rhoinf, uinf, linf))
end

# TODO: Outdated?
#=
function lift_force(u, normal_direction, equations::CompressibleEulerEquations2D)
    p = pressure(u, equations)
    return p * normal_direction[2] / norm(normal_direction)
end
=#

function (lift_force::LiftForcePressure)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack Ψl, rhoinf, uinf, linf = lift_force.force_state
    n = dot(normal_direction, Ψl) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
end

# TODO - Have only one function. Don't name it lift/drag. Varying the alpha allows you
# to choose between lift or drag in the elixir file.
function (lift_force_viscous::LiftForceViscous)(u, gradients, normal_direction, equations)
    @unpack Ψl, rhoinf, uinf, linf = lift_force_viscous.force_state
    @unpack mu = equations

    _, dv1dx, dv2dx, _ = convert_derivative_to_primitive(u, gradients[1], equations)
    _, dv1dy, dv2dy, _ = convert_derivative_to_primitive(u, gradients[2], equations)

    # Components of viscous stress tensor

    # (4/3 * (v1)_x - 2/3 * (v2)_y)
    tau_11 = 4.0 / 3.0 * dv1dx - 2.0 / 3.0 * dv2dy
    # ((v1)_y + (v2)_x)
    # stress tensor is symmetric
    tau_12 = dv1dy + dv2dx # = tau_21
    tau_21 = tau_12 # For readability
    # (4/3 * (v2)_y - 2/3 * (v1)_x)
    tau_22 = 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * dv1dx

    n = normal_direction / norm(normal_direction)
    force = tau_11 * n[1] * Ψl[1] + tau_12 * n[2] * Ψl[1] + tau_21 * n[1] * Ψl[2] +
            tau_22 * n[2] * Ψl[2]
    force *= mu
    factor = 0.5 * rhoinf * uinf^2 * linf
    return force / factor
end

function surface_skin_friction(u, gradients, normal_direction, equations,
                               free_stream_variables)
    @unpack rhoinf, uinf, linf = free_stream_variables
    @unpack mu = equations

    _, dv1dx, dv2dx, _ = convert_derivative_to_primitive(u, gradients[1], equations)
    _, dv1dy, dv2dy, _ = convert_derivative_to_primitive(u, gradients[2], equations)

    # Components of viscous stress tensor

    # (4/3 * (v1)_x - 2/3 * (v2)_y)
    tau_11 = 4.0 / 3.0 * dv1dx - 2.0 / 3.0 * dv2dy
    # ((v1)_y + (v2)_x)
    # stress tensor is symmetric
    tau_12 = dv1dy + dv2dx # = tau_21
    tau_21 = tau_12 # For readability
    # (4/3 * (v2)_y - 2/3 * (v1)_x)
    tau_22 = 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * dv1dx

    n = normal_direction / norm(normal_direction)
    n_perp = (-n[2], n[1])
    Cf = (tau_11 * n[1] * n_perp[1] + tau_12 * n[2] * n_perp[1]
          + tau_21 * n[1] * n_perp[2] + tau_22 * n[2] * n_perp[2])
    Cf *= mu
    factor = 0.5 * rhoinf * uinf^2 * linf
    return Cf / factor
end

#=
# TODO: Outdated?
function drag_force(u, normal_direction, equations)
    p = pressure(u, equations)
    return p * normal_direction[1] / norm(normal_direction)
end
=#

function (drag_force_viscous::DragForceViscous)(u, gradients, normal_direction, equations)
    @unpack Ψl, rhoinf, uinf, linf = drag_force_viscous.force_state
    mu = equations.mu

    _, dv1dx, dv2dx, _ = convert_derivative_to_primitive(u, gradients[1], equations)
    _, dv1dy, dv2dy, _ = convert_derivative_to_primitive(u, gradients[2], equations)

    # Components of viscous stress tensor

    # (4/3 * (v1)_x - 2/3 * (v2)_y)
    tau_11 = 4.0 / 3.0 * dv1dx - 2.0 / 3.0 * dv2dy
    # ((v1)_y + (v2)_x)
    # stress tensor is symmetric
    tau_12 = dv1dy + dv2dx # = tau_21
    tau_21 = tau_12 # Symmetric, and rewritten for readability
    # (4/3 * (v2)_y - 2/3 * (v1)_x)
    tau_22 = 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * dv1dx

    n = normal_direction / norm(normal_direction)
    force = tau_11 * n[1] * Ψl[1] + tau_12 * n[2] * Ψl[1] + tau_21 * n[1] * Ψl[2] +
            tau_22 * n[2] * Ψl[2]
    force *= mu # The tau had a factor of mu in Ray 2017, but it is not present in the
    # above expressions taken from Trixi.jl and thus it is included here
    factor = 0.5 * rhoinf * uinf^2 * linf
    return force / factor
end

function (drag_force::DragForcePressure)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack Ψl, rhoinf, uinf, linf = drag_force.force_state
    n = dot(normal_direction, Ψl) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
end

function analyze(quantity::SurfaceQuantitiyViscous,
                 du, u, t, semi::SemidiscretizationHyperbolicParabolic)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    equations_parabolic = semi.equations_parabolic
    cache_parabolic = semi.cache_parabolic
    analyze(quantity, du, u, t, mesh, equations, equations_parabolic, solver, cache,
            cache_parabolic)
end

function analyze(surface_variable::AnalysisSurfaceIntegral, du, u, t,
                 mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                 equations, dg::DGSEM, cache)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack indices, variable = surface_variable
    # TODO - Use initialize callbacks to move boundary_conditions to cache
    indices_ = indices(cache)

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    for local_index in eachindex(indices_)
        # Use the local index to get the global boundary index from the pre-sorted list
        boundary = indices_[local_index]

        # Get information on the adjacent element, compute the surface fluxes,
        # and store them
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in eachnode(dg)
            u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg, node_index,
                                         boundary)
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node,
                                                    element)

            # L2 norm of normal direction is the surface element
            # 0.5 factor is NOT needed, the norm(normal_direction) is all the factor needed
            dS = weights[node_index] * norm(normal_direction)
            surface_integral += variable(u_node, normal_direction, equations) * dS

            i_node += i_node_step
            j_node += j_node_step
        end
    end
    return surface_integral
end

function analyze(surface_variable::AnalysisSurfaceFrictionCoefficient,
                 du, u, t, mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                 equations,
                 equations_parabolic, dg::DGSEM, cache, cache_parabolic)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack indices, free_stream_variables = surface_variable
    # TODO - Use initialize callbacks to move boundary_conditions to cache
    indices_ = indices(cache)
    @unpack viscous_container = cache_parabolic
    @unpack gradients = viscous_container
    gradients_x, gradients_y = gradients

    dim = 2 # TODO Generalize!
    n_nodes = nnodes(dg)
    n_elements = length(indices_)
    avg_array = zeros(n_elements, dim + 1)
    soln_array = zeros(n_elements * n_nodes, dim + 1)

    it = 1
    element_it = 1

    index_range = eachnode(dg)
    for local_index in eachindex(indices_)
        # Use the local index to get the global boundary index from the pre-sorted list
        boundary = indices_[local_index]

        # Get information on the adjacent element, compute the surface fluxes,
        # and store them
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in eachnode(dg)
            x = get_node_coords(node_coordinates, equations, dg, i_node, j_node, element)
            u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg, node_index,
                                         boundary)
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node,
                                                    element)
            ux = Trixi.get_node_vars(gradients_x, equations, dg, i_node, j_node, element)
            uy = Trixi.get_node_vars(gradients_y, equations, dg, i_node, j_node, element)

            Cf = surface_skin_friction(u_node, (ux, uy), normal_direction,
                                       equations_parabolic, free_stream_variables)

            soln_array[it, 1:2] .= x
            soln_array[it, 3] = Cf
            avg_array[element_it, 1:2] .+= x * weights[node_index] / 2.0
            avg_array[element_it, 3] += Cf * weights[node_index] / 2.0

            i_node += i_node_step
            j_node += j_node_step

            it += 1
        end
        element_it += 1
    end
    mkpath("out")
    writedlm(joinpath("out", "Cf_t$t.txt"), soln_array)
    writedlm(joinpath("out", "Cf_avg_t$t.txt"), avg_array)

    return 0.0 # TODO: Required?
end

function analyze(surface_variable::AnalysisSurfaceIntegralViscousCorrectedGrad,
                 du, u, t, mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                 equations,
                 equations_parabolic, dg::DGSEM, cache, cache_parabolic)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack indices, variable = surface_variable
    # TODO - Use initialize callbacks to move boundary_conditions to cache
    indices_ = indices(cache)
    @unpack viscous_container = cache_parabolic
    @unpack gradients = viscous_container
    gradients_x, gradients_y = gradients

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    for local_index in eachindex(indices_)
        # Use the local index to get the global boundary index from the pre-sorted list
        boundary = indices_[local_index]

        # Get information on the adjacent element, compute the surface fluxes,
        # and store them
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in eachnode(dg)
            u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg, node_index,
                                         boundary)
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node,
                                                    element)
            ux = Trixi.get_node_vars(gradients_x, equations, dg, i_node, j_node, element)
            uy = Trixi.get_node_vars(gradients_y, equations, dg, i_node, j_node, element)

            # L2 norm of normal direction is the surface
            # 0.5 factor is NOT needed, the norm(normal_direction) is all the factor needed
            dS = weights[node_index] * norm(normal_direction)
            surface_integral += variable(u_node, (ux, uy), normal_direction,
                                         equations_parabolic) * dS

            i_node += i_node_step
            j_node += j_node_step
        end
    end
    return surface_integral
end

function analyze(surface_variable::AnalysisSurfaceIntegralViscous, du, u, t,
                 mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                 equations,
                 equations_parabolic, dg::DGSEM, cache, cache_parabolic)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack indices, variable = surface_variable
    # TODO - Use initialize callbacks to move boundary_conditions to cache
    indices_ = indices(cache)
    @unpack viscous_container = cache_parabolic
    @unpack gradients, u_transformed = viscous_container
    gradients_x, gradients_y = gradients

    reset_du!(gradients_x, dg, cache)
    reset_du!(gradients_y, dg, cache)

    @unpack derivative_matrix = dg.basis
    @threaded for element in eachelement(dg, cache)

        # Calculate volume terms in one element
        for j in eachnode(dg), i in eachnode(dg)
            # In Trixi, this is u_transformed instead of u. Does that have side-effects?
            # It shouldn't because we compute gradients in conservative variables
            u_node = get_node_vars(u_transformed, equations_parabolic, dg, i, j, element)

            for ii in eachnode(dg)
                multiply_add_to_node_vars!(gradients_x, derivative_matrix[ii, i],
                                           u_node, equations_parabolic, dg, ii, j,
                                           element)
            end

            for jj in eachnode(dg)
                multiply_add_to_node_vars!(gradients_y, derivative_matrix[jj, j],
                                           u_node, equations_parabolic, dg, i, jj,
                                           element)
            end
        end

        for j in eachnode(dg), i in eachnode(dg)
            Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j,
                                                  element)
            Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j,
                                                  element)

            gradients_reference_1 = get_node_vars(gradients_x, equations_parabolic, dg,
                                                  i, j, element)
            gradients_reference_2 = get_node_vars(gradients_y, equations_parabolic, dg,
                                                  i, j, element)

            # note that the contravariant vectors are transposed compared with computations of flux
            # divergences in `calc_volume_integral!`. See
            # https://github.com/trixi-framework/Trixi.jl/pull/1490#discussion_r1213345190
            # for a more detailed discussion.
            gradient_x_node = Ja11 * gradients_reference_1 + Ja21 * gradients_reference_2
            gradient_y_node = Ja12 * gradients_reference_1 + Ja22 * gradients_reference_2

            set_node_vars!(gradients_x, gradient_x_node, equations_parabolic, dg, i, j,
                           element)
            set_node_vars!(gradients_y, gradient_y_node, equations_parabolic, dg, i, j,
                           element)
        end
    end

    # apply_jacobian_parabolic! is needed because we don't want to flip the signs
    apply_jacobian_parabolic!(gradients_x, mesh, equations_parabolic, dg, cache_parabolic)
    apply_jacobian_parabolic!(gradients_y, mesh, equations_parabolic, dg, cache_parabolic)

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    for local_index in eachindex(indices_)
        # Use the local index to get the global boundary index from the pre-sorted list
        boundary = indices_[local_index]

        # Get information on the adjacent element, compute the surface fluxes,
        # and store them
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in eachnode(dg)
            u_node = Trixi.get_node_vars(boundaries.u, equations, dg, node_index, boundary)
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node,
                                                    element)
            ux = Trixi.get_node_vars(gradients_x, equations, dg, i_node, j_node, element)
            uy = Trixi.get_node_vars(gradients_y, equations, dg, i_node, j_node, element)

            # L2 norm of normal direction is the surface
            # 0.5 factor is NOT needed, the norm(normal_direction) is all the factor needed
            dS = weights[node_index] * norm(normal_direction)
            surface_integral += variable(u_node, (ux, uy), normal_direction,
                                         equations_parabolic) * dS

            i_node += i_node_step
            j_node += j_node_step
        end
    end
    return surface_integral
end

function analyze(surface_variable::SaveSurfacePrimitives, du, u, t,
                 mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                 equations, dg::DGSEM, cache)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack indices = surface_variable
    # TODO - Use initialize callbacks to move boundary_conditions to cache
    indices_ = indices(cache)
    dim = 2 # Generalize!
    nvar = nvariables(equations)
    n_nodes = nnodes(dg)
    n_elements = length(indices_)
    avg_array = zeros(n_elements, dim + nvar)
    soln_array = zeros(n_elements * n_nodes, dim + nvar)

    it = 1
    element_it = 1
    index_range = eachnode(dg)
    for local_index in eachindex(indices_)
        # Use the local index to get the global boundary index from the pre-sorted list
        boundary = indices_[local_index]

        # Get information on the adjacent element, compute the surface fluxes,
        # and store them
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in eachnode(dg)
            u_node = Trixi.get_node_vars(boundaries.u, equations, dg, node_index, boundary)
            x = get_node_coords(node_coordinates, equations, dg, i_node, j_node, element)
            prim = cons2prim(u_node, equations)

            soln_array[it, 1:2] .= x
            soln_array[it, 3:end] .= prim
            avg_array[element_it, 1:2] .+= x * weights[node_index] / 2.0
            avg_array[element_it, 3:end] .+= prim * weights[node_index] / 2.0
            i_node += i_node_step
            j_node += j_node_step
            it += 1
        end
        element_it += 1
    end
    mkpath("out")
    writedlm(joinpath("out", "soln_t$t.txt"), soln_array)
    writedlm(joinpath("out", "avg_t$t.txt"), avg_array)

    return 0.0 # TODO: Required?
end

pretty_form_ascii(::SaveSurfacePrimitives{<:Any}) = "Dummy value"
pretty_form_utf(::SaveSurfacePrimitives{<:Any}) = "Dummy value"

pretty_form_ascii(::AnalysisSurfaceFrictionCoefficient{<:Any}) = "Dummy value"
pretty_form_utf(::AnalysisSurfaceFrictionCoefficient{<:Any}) = "Dummy value"

# TODO: Outdated / no longer needed?
#pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, typeof(lift_force)}) = "Lift"
#pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, typeof(lift_force)}) = "Lift"
#pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, typeof(drag_force)}) = "Drag"
#pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, typeof(drag_force)}) = "Drag"

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, <:LiftForcePressure{<:Any}})
    "Pressure_lift"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, <:LiftForcePressure{<:Any}})
    "Pressure_lift"
end
function pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, <:DragForcePressure{<:Any}})
    "Pressure_drag"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, <:DragForcePressure{<:Any}})
    "Pressure_drag"
end

function pretty_form_ascii(::AnalysisSurfaceIntegralViscous{<:Any,
                                                            <:LiftForceViscous{<:Any}})
    "Viscous_lift"
end
function pretty_form_utf(::AnalysisSurfaceIntegralViscous{<:Any, <:LiftForceViscous{<:Any}})
    "Viscous_lift"
end
function pretty_form_ascii(::AnalysisSurfaceIntegralViscous{<:Any,
                                                            <:DragForceViscous{<:Any}})
    "Viscous_drag"
end
function pretty_form_utf(::AnalysisSurfaceIntegralViscous{<:Any, <:DragForceViscous{<:Any}})
    "Viscous_drag"
end

function pretty_form_ascii(::AnalysisSurfaceIntegralViscousCorrectedGrad{<:Any,
                                                                         <:LiftForceViscous{<:Any}})
    "Viscous_lift_corr"
end
function pretty_form_utf(::AnalysisSurfaceIntegralViscousCorrectedGrad{<:Any,
                                                                       <:LiftForceViscous{<:Any}})
    "Viscous_lift_corr"
end
function pretty_form_ascii(::AnalysisSurfaceIntegralViscousCorrectedGrad{<:Any,
                                                                         <:DragForceViscous{<:Any}})
    "Viscous_drag_corr"
end
function pretty_form_utf(::AnalysisSurfaceIntegralViscousCorrectedGrad{<:Any,
                                                                       <:DragForceViscous{<:Any}})
    "Viscous_drag_corr"
end
