# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# Custom implementation for PERK integrator
function (amr_callback::AMRCallback)(integrator::PERK_Integrator; kwargs...)
  u_ode = integrator.u
  semi = integrator.p

  @trixi_timeit timer() "AMR" begin
  has_changed = amr_callback(u_ode, semi,
                             integrator.t, integrator.iter; kwargs...)
    if has_changed
      resize!(integrator, length(u_ode))
      u_modified!(integrator, true)

      println("AMR Call")
      mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
      @unpack elements, interfaces, boundaries = cache

      n_elements   = length(elements.cell_ids)
      n_interfaces = length(interfaces.orientations)
      n_boundaries = length(boundaries.orientations) # TODO Not sure if adequate
  
      # TODO: Not sure if this still returns the correct number of ACTIVE Levels
      min_level = minimum_level(mesh.tree)
      max_level = maximum_level(mesh.tree)
      n_levels = max_level - min_level + 1
  
      # Initialize storage for level-wise information
      # Set-like datastructures more suited then vectors (Especially for interfaces)
      level_info_elements     = [Vector{Int}() for _ in 1:n_levels]
      integrator.level_info_elements_acc = [Vector{Int}() for _ in 1:n_levels]
  
      # Determine level for each element
      for element_id in 1:n_elements
        # Determine level
        level = mesh.tree.levels[elements.cell_ids[element_id]]
        # Convert to level id
        level_id = max_level + 1 - level
  
        push!(level_info_elements[level_id], element_id)
        # Add to accumulated container
        for l in level_id:n_levels
          push!(integrator.level_info_elements_acc[l], element_id)
        end
      end
  
      # Use sets first to avoid double storage of interfaces
      level_info_interfaces_set_acc = [Set{Int}() for _ in 1:n_levels]
      # Determine level for each interface
      for interface_id in 1:n_interfaces
        # Get element ids
        element_id_left  = interfaces.neighbor_ids[1, interface_id]
        element_id_right = interfaces.neighbor_ids[2, interface_id]
  
        # Determine level
        level_left  = mesh.tree.levels[elements.cell_ids[element_id_left]]
        level_right = mesh.tree.levels[elements.cell_ids[element_id_right]]
  
        # Convert to level id
        level_id_left  = max_level + 1 - level_left
        level_id_right = max_level + 1 - level_right
  
        # Add to accumulated container
        for l in level_id_left:n_levels
          push!(level_info_interfaces_set_acc[l], interface_id)
        end
        for l in level_id_right:n_levels
          push!(level_info_interfaces_set_acc[l], interface_id)
        end
      end
  
      # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
      integrator.level_info_interfaces_acc = [Vector{Int}() for _ in 1:n_levels]
      for level in 1:n_levels
        integrator.level_info_interfaces_acc[level] = sort(collect(level_info_interfaces_set_acc[level]))
      end
  
      # Use sets first to avoid double storage of boundaries
      level_info_boundaries_set_acc = [Set{Int}() for _ in 1:n_levels]
      # Determine level for each boundary
      for boundary_id in 1:n_boundaries
        # Get element ids
        element_id_left  = boundaries.neighbor_ids[1, boundary_id]
        element_id_right = interfaces.neighbor_ids[2, interface_id]
  
        # Determine level
        level_left  = mesh.tree.levels[elements.cell_ids[element_id_left]]
        level_right = mesh.tree.levels[elements.cell_ids[element_id_right]]
  
        # Convert to level id
        level_id_left  = max_level + 1 - level_left
        level_id_right = max_level + 1 - level_right
  
        # Add to accumulated container
        for l in level_id_left:n_levels
          push!(level_info_boundaries_set_acc[l], surface_id)
        end
        for l in level_id_right:n_levels
          push!(level_info_boundaries_set_acc[l], surface_id)
        end
      end
  
      # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
      integrator.level_info_boundaries_acc = [Vector{Int}() for _ in 1:n_levels]
      for level in 1:n_levels
        integrator.level_info_boundaries_acc[level] = sort(collect(level_info_boundaries_set_acc[level]))
      end
  
      u = wrap_array(u_ode, mesh, equations, solver, cache)
  
      integrator.level_u_indices_elements = [Vector{Int}() for _ in 1:n_levels]
      for level in 1:n_levels
        for element_id in level_info_elements[level]
          indices = vec(transpose(LinearIndices(u)[:, :, element_id]))
          append!(integrator.level_u_indices_elements[level], indices)
        end
      end
    end
  end

  return has_changed
end


# Custom implementation for PERK integrator
@inline function (amr_callback::AMRCallback)(u_ode::AbstractVector,
                                             semi::SemidiscretizationHyperbolic,
                                             t, iter; kwargs...)                                            
  # Note that we don't `wrap_array` the vector `u_ode` to be able to `resize!`
  # it when doing AMR while still dispatching on the `mesh` etc.
  amr_callback(u_ode, mesh_equations_solver_cache(semi)..., semi, t, iter; kwargs...)
end

end # @muladd
