# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# Custom implementation for PERK integrator
function (amr_callback::AMRCallback)(integrator::Union{PERK_Multi_Integrator, PERK3_Multi_Integrator}; kwargs...)
  u_ode = integrator.u
  semi = integrator.p

  @trixi_timeit timer() "AMR" begin
    has_changed = amr_callback(u_ode, semi,
                               integrator.t, integrator.iter; kwargs...)

    if has_changed
      resize!(integrator, length(u_ode))
      u_modified!(integrator, true)

      ### PERK addition ###
      # TODO: Need to make this much less allocating!
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

        integrator.min_lvl = min_level
        integrator.max_lvl = max_level
        integrator.n_levels = n_levels

        n_dims = ndims(mesh.tree) # Spatial dimension

        # Next to fine NOT integrated with fine scheme

        # Initialize storage for level-wise information
        if n_levels != length(integrator.level_info_elements_acc)
          integrator.level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
          integrator.level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]
          integrator.level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]
          integrator.level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]
          # For efficient treatment of boundaries we need additional datastructures
          integrator.level_info_boundaries_orientation_acc = [[Vector{Int64}() for _ in 1:2*n_dims] for _ in 1:n_levels]
          integrator.level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]
          integrator.level_u_indices_elements = [Vector{Int64}() for _ in 1:n_levels]
          #resize!(integrator.level_info_elements_acc, n_levels) # TODO: Does unfortunately not work
        else # Just empty datastructures
          for level in 1:n_levels
            empty!(integrator.level_info_elements[level])
            empty!(integrator.level_info_elements_acc[level])
            empty!(integrator.level_info_interfaces_acc[level])
            empty!(integrator.level_info_boundaries_acc[level])
            for dim in 1:2*n_dims
              empty!(integrator.level_info_boundaries_orientation_acc[level][dim])
            end
            empty!(integrator.level_info_mortars_acc[level])
            empty!(integrator.level_u_indices_elements[level])
          end
        end

        # Determine level for each element
        for element_id in 1:n_elements
          # Determine level
          level = mesh.tree.levels[elements.cell_ids[element_id]]
          # Convert to level id
          level_id = max_level + 1 - level

          push!(integrator.level_info_elements[level_id], element_id)
          # Add to accumulated container
          for l in level_id:n_levels
            push!(integrator.level_info_elements_acc[l], element_id)
          end
        end
        @assert length(integrator.level_info_elements_acc[end]) == 
          n_elements "highest level should contain all elements"


        # Determine level for each interface
        for interface_id in 1:n_interfaces
          # Get element ids
          element_id_left  = interfaces.neighbor_ids[1, interface_id]
          element_id_right = interfaces.neighbor_ids[2, interface_id]

          # Determine level
          level_left  = mesh.tree.levels[elements.cell_ids[element_id_left]]
          level_right = mesh.tree.levels[elements.cell_ids[element_id_right]]

          # Higher element's level determines this interfaces' level
          level_id = max_level + 1 - max(level_left, level_right)
          for l in level_id:n_levels
            push!(integrator.level_info_interfaces_acc[l], interface_id)
          end
        end
        @assert length(integrator.level_info_interfaces_acc[end]) == 
          n_interfaces "highest level should contain all interfaces"


        # Determine level for each boundary
        for boundary_id in 1:n_boundaries
          # Get element id (boundaries have only one unique associated element)
          element_id = boundaries.neighbor_ids[boundary_id]

          # Determine level
          level = mesh.tree.levels[elements.cell_ids[element_id]]

          # Convert to level id
          level_id = max_level + 1 - level

          # Add to accumulated container
          for l in level_id:n_levels
            push!(integrator.level_info_boundaries_acc[l], boundary_id)
          end

          # For orientation-side wise specific treatment
          if boundaries.orientations[boundary_id] == 1 # x Boundary
            if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
              for l in level_id:n_levels
                push!(integrator.level_info_boundaries_orientation_acc[l][2], boundary_id)
              end
            else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
              for l in level_id:n_levels
                push!(integrator.level_info_boundaries_orientation_acc[l][1], boundary_id)
              end
            end
          elseif boundaries.orientations[boundary_id] == 2 # y Boundary
            if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
              for l in level_id:n_levels
                push!(integrator.level_info_boundaries_orientation_acc[l][4], boundary_id)
              end
            else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
              for l in level_id:n_levels
                push!(integrator.level_info_boundaries_orientation_acc[l][3], boundary_id)
              end
            end
          elseif boundaries.orientations[boundary_id] == 3 # z Boundary
            if boundaries.neighbor_sides[boundary_id] == 1 # Boundary on negative coordinate side
              for l in level_id:n_levels
                push!(integrator.level_info_boundaries_orientation_acc[l][6], boundary_id)
              end
            else # boundaries.neighbor_sides[boundary_id] == 2 Boundary on positive coordinate side
              for l in level_id:n_levels
                push!(integrator.level_info_boundaries_orientation_acc[l][5], boundary_id)
              end
            end 
          end
        end
        @assert length(integrator.level_info_boundaries_acc[end]) == 
          n_boundaries "highest level should contain all boundaries"


        dimensions = ndims(mesh.tree) # Spatial dimension
        if dimensions > 1
          @unpack mortars = cache
          n_mortars = length(mortars.orientations)

          for mortar_id in 1:n_mortars
            # Get element ids
            element_id_lower  = mortars.neighbor_ids[1, mortar_id]
            element_id_higher = mortars.neighbor_ids[2, mortar_id]

            # Determine level
            level_lower  = mesh.tree.levels[elements.cell_ids[element_id_lower]]
            level_higher = mesh.tree.levels[elements.cell_ids[element_id_higher]]

            # Higher element's level determines this mortars' level
            level_id = max_level + 1 - max(level_lower, level_higher)
            # Add to accumulated container
            for l in level_id:n_levels
              push!(integrator.level_info_mortars_acc[l], mortar_id)
            end
          end
          @assert length(integrator.level_info_mortars_acc[end]) == 
            n_mortars "highest level should contain all mortars"
        end
    
        u = wrap_array(u_ode, mesh, equations, solver, cache)

        if dimensions == 1
          for level in 1:n_levels
            for element_id in integrator.level_info_elements[level]
              # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
              indices = vec(transpose(LinearIndices(u)[:, :, element_id]))
              append!(integrator.level_u_indices_elements[level], indices)
            end
          end
        elseif dimensions == 2
          for level in 1:n_levels
            for element_id in integrator.level_info_elements[level]
              # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
              indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, element_id]))
              append!(integrator.level_u_indices_elements[level], indices)
            end
          end
        elseif n_dims == 3
          for level in 1:n_levels
            for element_id in integrator.level_info_elements[level]
              # First dimension of u: nvariables, following: nnodes (per dim) last: nelements
              indices = collect(Iterators.flatten(LinearIndices(u)[:, :, :, :, element_id]))
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