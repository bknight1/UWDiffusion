from enum import Enum
import gmsh
import underworld3 as uw
import os

def generate_2D_mesh_from_points(points, cell_size, qdegree=2, output_path="./.meshes", model_name="point_mesh"):
    """
    Generates a 2D mesh using Gmsh based on user-defined points.

    Parameters:
    - points: list of tuple
        List of (x, y, z) tuples defining the vertices of the outer boundary.
    - cell_size: float
        Mesh cell size for the elements.
    - output_path: str
        Path to save the generated mesh file.
    - model_name: str, optional
        Name of the Gmsh model (default: "mesh_model").

    Returns:
    - None: Writes the mesh to the specified output file.
    """
    # Dynamically create the Boundaries2D enum class
    class Boundaries2D(Enum):
        """
        Enum class for defining boundary tags in a 2D mesh.
        Dynamically generated based on the number of points.
        """
        def _generate_next_value_(name, start, count, last_values):
            # Automatically generate the boundary tag values starting from 11
            return 11 + count

    for i in range(len(points)):
        setattr(Boundaries2D, f"Boundary{i}", 11 + i)

    model_name = f"{model_name}_2D_csize={cell_size}_degree={qdegree}"

    if uw.mpi.rank == 0:
        os.makedirs(output_path, exist_ok=True)
    
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add(model_name)

    # Create points
    point_ids = [gmsh.model.geo.addPoint(x, y, z, meshSize=cell_size) for x, y, z in points]

    # Create lines by connecting consecutive points and closing the loop
    line_ids = [gmsh.model.geo.addLine(point_ids[i], point_ids[(i + 1) % len(point_ids)]) for i in range(len(point_ids))]

    # Create a curve loop and a plane surface
    curve_loop = gmsh.model.geo.addCurveLoop(line_ids)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    # Synchronize the Gmsh model
    gmsh.model.geo.synchronize()

    # Add physical groups for lines and assign boundary tags
    for i, line_id in enumerate(line_ids):
        boundary_tag = getattr(Boundaries2D, f"Boundary{i}", 11 + i)  # Default to 11 + i
        gmsh.model.addPhysicalGroup(1, [line_id], tag=boundary_tag)
        gmsh.model.setPhysicalName(1, boundary_tag, f"Boundary{i}")

    # Add a physical group for the surface
    surface_tag = 99999
    gmsh.model.addPhysicalGroup(2, [surface], surface_tag)
    gmsh.model.setPhysicalName(2, surface_tag, "Elements")

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    # Write the mesh to the specified output path
    gmsh.write(f"{output_path}/{model_name}.msh")

    # Finalize Gmsh
    gmsh.finalize()

    Boundaries2D = Enum(
        "Boundaries2D",
        {f"Boundary{i}": 11 + i for i in range(len(points))}
    )

    mesh = uw.discretisation.Mesh(
        f'{output_path}/{model_name}.msh',
        degree=1,
        qdegree=qdegree,
        boundaries=Boundaries2D,
        boundary_normals=None,
        coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        refinement=None,
        refinement_callback=None,
    )

    return mesh