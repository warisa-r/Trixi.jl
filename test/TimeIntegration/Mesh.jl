using HOHQMesh, GLMakie

# Create a new HOHQMesh model project. The project name
# "cylinder" will be the name of the mesh file
# saved in the directory "out".
cylinder_flow = newProject("cylinder", "out")

# Reset polynomial order of the mesh model curves and output format.
# The "ABAQUS" mesh file format is needed for the adaptive mesh
# capability of Trixi.jl.
setPolynomialOrder!(cylinder_flow, 3)
setMeshFileFormat!(cylinder_flow, "ABAQUS")

# A background grid is required for the mesh generation. In this example we lay a
# background grid of Cartesian boxes
base_size = 0.5
addBackgroundGrid!(cylinder_flow, [base_size, base_size, 0.0])

#=
lower_left = [0.0, -10, 0.0]
spacing = [base_size, base_size, 0.0]
Nx = Int(40 / base_size)
Ny = Int(20 / base_size)
num_intervals = [Nx, Ny, 0]
addBackgroundGrid!(cylinder_flow, lower_left, spacing, num_intervals)
=#

#=
bounds = [10.0, 0.0, -10.0, 40.0]
N = [Int(40/base_size), Int(20/base_size), 0]
addBackgroundGrid!(cylinder_flow, bounds, N)
=#


# Add outer boundary curves in counter-clockwise order.
# Note, the curve names are those that will be present in the mesh file.
left = newEndPointsLineCurve("Left", [0.0, 10.0, 0.0], [0.0, -10.0, 0.0])

bottom = newEndPointsLineCurve("Bottom", [0.0, -10, 0.0], [40.0, -10, 0.0])

right = newEndPointsLineCurve("Right", [40.0, -10, 0.0], [40, 10, 0.0])

top = newEndPointsLineCurve("Top", [40.0, 10, 0.0], [0.0, 10, 0.0])


# Outer boundary curve chain is created to have counter-clockwise
# orientation, as required by HOHQMesh generator
addCurveToOuterBoundary!(cylinder_flow, bottom)
addCurveToOuterBoundary!(cylinder_flow, right)
addCurveToOuterBoundary!(cylinder_flow, top)
addCurveToOuterBoundary!(cylinder_flow, left)

# Add inner boundary curve
cylinder = newCircularArcCurve("Circle",        # curve name
                               [10, 0.0, 0.0], # circle center
                               0.5,            # circle radius
                               0.0,             # start angle
                               360.0,           # end angle
                               "degrees")       # angle units

addCurveToInnerBoundary!(cylinder_flow, cylinder, "inner1")


# Add a refinement line for the wake region.
wake_region = newRefinementLine("wake_region", "smooth", [10,0.0,0.0], [35,0.0,0.0], base_size/2, 5.0)
cylinder = newRefinementCenter("cylinder", "smooth", [10.0,0.0,0.0],  base_size/2, 3.0)

addRefinementRegion!(cylinder_flow, wake_region)
addRefinementRegion!(cylinder_flow, cylinder)

# Visualize the model, refinement region and background grid
# prior to meshing.
plotProject!(cylinder_flow, MODEL+REFINEMENTS+GRID)

@info "Press enter to generate the mesh and update the plot."
readline()

# Generate the mesh. Saves the mesh file to the directory "out".
generate_mesh(cylinder_flow)