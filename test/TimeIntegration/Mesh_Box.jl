using HOHQMesh, GLMakie

# Create a new HOHQMesh model project. The project name
# "box" will be the name of the mesh file
# saved in the directory "out".
box_flow = newProject("box", "out")

# Reset polynomial order of the mesh model curves and output format.
# The "ABAQUS" mesh file format is needed for the adaptive mesh
# capability of Trixi.jl.
setPolynomialOrder!(box_flow, 3)
setMeshFileFormat!(box_flow, "ABAQUS")

# A background grid is required for the mesh generation. In this example we lay a
# background grid of Cartesian boxes with size 0.2.
addBackgroundGrid!(box_flow, [0.125, 0.125, 0.0])

# Add outer boundary curves in counter-clockwise order.
# Note, the curve names are those that will be present in the mesh file.
left = newEndPointsLineCurve("Left", [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0])

bottom = newEndPointsLineCurve("Bottom", [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0])

right = newEndPointsLineCurve("Right", [1.0, -1.0, 0.0], [1.0, 1.0, 0.0])

top = newEndPointsLineCurve("Top", [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0])

# Outer boundary curve chain is created to have counter-clockwise
# orientation, as required by HOHQMesh generator
addCurveToOuterBoundary!(box_flow, bottom)
addCurveToOuterBoundary!(box_flow, right)
addCurveToOuterBoundary!(box_flow, top)
addCurveToOuterBoundary!(box_flow, left)


# Add a refinement line for the wake region.
ref_1 = newRefinementLine("ref_1", "smooth", [-0.5,0.0,0.0], [0.5,0.0,0.0], 0.0625, 0.5)
ref_2 = newRefinementLine("ref_2", "smooth", [-0.25,0.0,0.0], [0.25,0.0,0.0], 0.03125, 0.25)

addRefinementRegion!(box_flow, ref_1)
addRefinementRegion!(box_flow, ref_2)

# Visualize the model, refinement region and background grid
# prior to meshing.
plotProject!(box_flow, MODEL+REFINEMENTS+GRID)

@info "Press enter to generate the mesh and update the plot."
readline()

# Generate the mesh. Saves the mesh file to the directory "out".
generate_mesh(box_flow)