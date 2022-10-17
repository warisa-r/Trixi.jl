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

      ### PERK addition ###
      @trixi_timeit timer() "PERK stage identifiers update" begin
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
        level_info_elements                = [Vector{Int}() for _ in 1:n_levels]
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
        @assert length(integrator.level_info_elements_acc[end]) == 
            n_elements "highest level should contain all elements"

    
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
        @assert length(integrator.level_info_interfaces_acc[end]) == 
            n_interfaces "highest level should contain all interfaces"

    
        # Use sets first to avoid double storage of boundaries
        level_info_boundaries_set_acc = [Set{Int}() for _ in 1:n_levels]
        # Determine level for each boundary
        for boundary_id in 1:n_boundaries
          # Get element ids
          element_id_left  = boundaries.neighbor_ids[1, boundary_id]
          element_id_right = boundaries.neighbor_ids[2, boundary_id]
    
          # Determine level
          level_left  = mesh.tree.levels[elements.cell_ids[element_id_left]]
          level_right = mesh.tree.levels[elements.cell_ids[element_id_right]]
    
          # Convert to level id
          level_id_left  = max_level + 1 - level_left
          level_id_right = max_level + 1 - level_right
    
          # Add to accumulated container
          for l in level_id_left:n_levels
            push!(level_info_boundaries_set_acc[l], boundary_id)
          end
          for l in level_id_right:n_levels
            push!(level_info_boundaries_set_acc[l], boundary_id)
          end
        end
    
        # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
        integrator.level_info_boundaries_acc = [Vector{Int}() for _ in 1:n_levels]
        for level in 1:n_levels
          integrator.level_info_boundaries_acc[level] = sort(collect(level_info_boundaries_set_acc[level]))
        end
        @assert length(integrator.level_info_boundaries_acc[end]) == 
            n_boundaries "highest level should contain all boundaries"


        dimensions = ndims(mesh.tree) # Spatial dimension
        integrator.level_info_mortars_acc = [Vector{Int}() for _ in 1:n_levels]

        if dimensions > 1
          # Determine level for each mortar
          # Since mortars belong by definition to two levels, theoretically we have to
          # add them twice: Once for each level of its neighboring elements. However,
          # as we store the accumulated mortar ids, we only need to consider the one of
          # the small neighbors (here: the lower one), is it has the higher level and
          # thus the lower level id.
    
          @unpack mortars = cache
          n_mortars = length(mortars.orientations)
    
          # TODO: Mortars need probably to be reconsidered! (sets, level-assignment, ...)
          for mortar_id in 1:n_mortars
            # Get element ids
            element_id_lower = mortars.neighbor_ids[1, mortar_id]
    
            # Determine level
            level_lower = mesh.tree.levels[elements.cell_ids[element_id_lower]]
    
            # Convert to level id
            level_id_lower = max_level + 1 - level_lower
    
            # Add to accumulated container
            for l in level_id_lower:n_levels
              push!(integrator.level_info_mortars_acc[l], mortar_id)
            end
          end
          @assert length(integrator.level_info_mortars_acc[end]) == 
            n_mortars "highest level should contain all mortars"
        end
        

        #=
        # Initialize storage for level-wise information
        # Set-like datastructures more suited then vectors (Especially for interfaces)
        level_info_elements_set     = [Set{Int}() for _ in 1:n_levels]
        level_info_elements_set_acc = [Set{Int}() for _ in 1:n_levels]
        # Loop over interfaces to have access to its neighbors
        for interface_id in 1:n_interfaces
          # Get element ids
          element_id_left  = interfaces.neighbor_ids[1, interface_id]
          element_id_right = interfaces.neighbor_ids[2, interface_id]

          # Determine level
          level_left  = mesh.tree.levels[elements.cell_ids[element_id_left]]
          level_right = mesh.tree.levels[elements.cell_ids[element_id_right]]

          # Neighbors of finer cells should be integrated with same integrator
          ode_level = max(level_left, level_right)

          # Convert to level id
          level_id = max_level + 1 - ode_level

          # Assign elements according to their neighbors
          push!(level_info_elements_set[level_id], element_id_left)
          push!(level_info_elements_set[level_id], element_id_right)
          # Add to accumulated container
          for l in level_id:n_levels
            push!(level_info_elements_set_acc[l], element_id_left)
            push!(level_info_elements_set_acc[l], element_id_right)
          end
        end
        # Turn sets into sorted vectors to have (hopefully) faster accesses due to contiguous storage
        level_info_elements = [Vector{Int}() for _ in 1:n_levels]
        for level in 1:n_levels
          # Make sure elements are only stored once: In the finest level
          for fine_level in 1:level-1
            level_info_elements_set[level] = setdiff(level_info_elements_set[level], level_info_elements_set[fine_level])
          end

          level_info_elements[level] = sort(collect(level_info_elements_set[level]))
        end

        # Set up dictionary to set later ODE level for interfaces
        element_ODE_level_dict = Dict{Int, Int}()
        for level in 1:n_levels
          for element_id in level_info_elements[level]
            push!(element_ODE_level_dict, element_id=>level)
          end
        end

        integrator.level_info_elements_acc = [Vector{Int}() for _ in 1:n_levels]
        for level in 1:n_levels
          integrator.level_info_elements_acc[level] = sort(collect(level_info_elements_set_acc[level]))
        end
        @assert length(integrator.level_info_elements_acc[end]) == 
          n_elements "highest level should contain all elements"

        # Use sets first to avoid double storage of interfaces
        level_info_interfaces_set_acc = [Set{Int}() for _ in 1:n_levels]
        # Determine ODE level for each interface
        for interface_id in 1:n_interfaces
          # Get element ids
          element_id_left  = interfaces.neighbor_ids[1, interface_id]
          element_id_right = interfaces.neighbor_ids[2, interface_id]

          # Neighbors of finer cells should be integrated with same integrator
          ode_level = min(get(element_ODE_level_dict, element_id_left, -1), 
                          get(element_ODE_level_dict, element_id_right, -1))

          @assert ode_level != -1 "Errors in datastructures for ODE level assignment"           

          # Add to accumulated container
          for l in ode_level:n_levels
            push!(level_info_interfaces_set_acc[l], interface_id)
          end
        end
        # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
        integrator.level_info_interfaces_acc = [Vector{Int}() for _ in 1:n_levels]
        for level in 1:n_levels
          integrator.level_info_interfaces_acc[level] = sort(collect(level_info_interfaces_set_acc[level]))
        end
        @assert length(integrator.level_info_interfaces_acc[end]) == 
          n_interfaces "highest level should contain all interfaces"


        # Use sets first to avoid double storage of boundaries
        level_info_boundaries_set_acc = [Set{Int}() for _ in 1:n_levels]
        # Determine level for each boundary
        for boundary_id in 1:n_boundaries
          # Get element ids
          element_id_left  = boundaries.neighbor_ids[1, boundary_id]
          element_id_right = boundaries.neighbor_ids[2, boundary_id]

          # Determine level
          level_left  = mesh.tree.levels[elements.cell_ids[element_id_left]]
          level_right = mesh.tree.levels[elements.cell_ids[element_id_right]]

          # Convert to level id
          level_id_left  = max_level + 1 - level_left
          level_id_right = max_level + 1 - level_right

          # Add to accumulated container
          for l in level_id_left:n_levels
            push!(level_info_boundaries_set_acc[l], boundary_id)
          end
          for l in level_id_right:n_levels
            push!(level_info_boundaries_set_acc[l], boundary_id)
          end
        end

        # Turn set into sorted vectors to have (hopefully) faster accesses due to contiguous storage
        integrator.level_info_boundaries_acc = [Vector{Int}() for _ in 1:n_levels]
        for level in 1:n_levels
          integrator.level_info_boundaries_acc[level] = sort(collect(level_info_boundaries_set_acc[level]))
        end
        @assert length(integrator.level_info_boundaries_acc[end]) == 
          n_boundaries "highest level should contain all boundaries"

        # TODO: Mortars need probably to be reconsidered! (sets, level-assignment, ...)
        integrator.level_info_mortars_acc = [Vector{Int}() for _ in 1:n_levels]
        dimensions = ndims(mesh.tree) # Spatial dimension
        if dimensions > 1
          # Determine level for each mortar
          # Since mortars belong by definition to two levels, theoretically we have to
          # add them twice: Once for each level of its neighboring elements. However,
          # as we store the accumulated mortar ids, we only need to consider the one of
          # the small neighbors (here: the lower one), is it has the higher level and
          # thus the lower level id.

          @unpack mortars = cache
          n_mortars = length(mortars.orientations)

          for mortar_id in 1:n_mortars
            # Get element ids
            element_id_lower = mortars.neighbor_ids[1, mortar_id]

            # Determine level
            level_lower = mesh.tree.levels[elements.cell_ids[element_id_lower]]

            # Convert to level id
            level_id_lower = max_level + 1 - level_lower

            # Add to accumulated container
            for l in level_id_lower:n_levels
              push!(integrator.level_info_mortars_acc[l], mortar_id)
            end
          end
          @assert length(integrator.level_info_mortars_acc[end]) == 
            n_mortars "highest level should contain all mortars"
        end
        =#
    
        u = wrap_array(u_ode, mesh, equations, solver, cache)
    
        integrator.level_u_indices_elements = [Vector{Int}() for _ in 1:n_levels]

        # Have if outside for performance reasons
        if dimensions == 1
          for level in 1:n_levels
            for element_id in level_info_elements[level]
              indices = vec(transpose(LinearIndices(u)[:, :, element_id]))
              append!(integrator.level_u_indices_elements[level], indices)
            end
          end
        elseif dimensions == 2
          for level in 1:n_levels
            for element_id in level_info_elements[level]
              indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, element_id]))
              append!(integrator.level_u_indices_elements[level], indices)
            end
          end
        end
        
      end # "PERK stage identifiers update" timing
    end # if has changed
  end # "AMR" timing

  return has_changed
end

end # @muladd