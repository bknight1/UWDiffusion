from enum import Enum
import gmsh
import underworld3 as uw
import os

def generate_2D_mesh_from_points(points, 
                                 cell_size, 
                                 qdegree=2, 
                                 output_path="./.meshes", 
                                 model_name="point_mesh"):
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


def generate_garnet_annulus_mesh(
    radii_nd,
    mesh_degree,
    total_shell_count,
    growth_shell_count,
    output_path="./.meshes",
    mesh_name="garnet_mesh",
):
    """Generate annulus mesh with embedded shell boundaries for garnet-growth models.

    Parameters
    ----------
    radii_nd : array-like
        Non-dimensional radial coordinates used to build concentric circular shells.
    mesh_degree : int
        Quadrature degree for the returned Underworld mesh.
    output_path : str
        Directory where the gmsh `.msh` file will be written.
    total_shell_count : int
        Total number of radial shell samples in the reference profile (e.g. `Rr.shape[0]`).
    growth_shell_count : int
        Number of shells used in growth evolution (e.g. `radial_growth.shape[0]`).
    mesh_name : str, optional
        Output mesh filename stem (default: ``garnet_mesh``).

    Returns
    -------
    underworld3.discretisation.Mesh
        Underworld mesh with boundaries ``Centre`` and ``shell_0..shell_n``.
    """
    boundary_labels = ["Centre"]
    boundary_tags = [100]

    if uw.mpi.rank == 0:
        os.makedirs(output_path, exist_ok=True)
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", False)
        gmsh.model.add("Annulus_garnet_mesh")

        def generate_curve(radius, cellsize, p0, label):
            p1 = gmsh.model.geo.add_point(radius, 0.0, 0.0, meshSize=cellsize)
            p2 = gmsh.model.geo.add_point(-radius, 0.0, 0.0, meshSize=cellsize)
            c0 = gmsh.model.geo.add_circle_arc(p1, p0, p2)
            c1 = gmsh.model.geo.add_circle_arc(p2, p0, p1)
            cl = gmsh.model.geo.add_curve_loop([c0, c1], tag=label)
            return [c0, c1], cl

        internal_curves = []
        cls = []

        cellsize = float(radii_nd[1] - radii_nd[0])
        p0 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=cellsize)

        tag = 101
        final_loop = None
        for radius in radii_nd[1:]:
            internal_curve, cl = generate_curve(radius, cellsize, p0, tag)
            cls.append(cl)
            internal_curves.append(internal_curve)
            final_loop = cl
            tag += 1

        if final_loop is None:
            raise ValueError("At least two radial points are required to build annulus mesh.")

        s = gmsh.model.geo.add_plane_surface([final_loop])
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.embed(0, [p0], 2, s)
        for curve in internal_curves:
            gmsh.model.mesh.embed(1, curve, 2, s)
        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(0, [p0], tag=boundary_tags[0], name=boundary_labels[0])

        shell = 0
        start_shell = int(total_shell_count) - int(growth_shell_count) - 1
        for curve in internal_curves[start_shell:]:
            boundary_label = f"shell_{shell}"
            boundary_labels.append(boundary_label)
            boundary_tags.append(cls[shell])
            gmsh.model.addPhysicalGroup(1, curve, cls[shell], name=boundary_label)
            shell += 1

        gmsh.model.addPhysicalGroup(2, [s], 666666, "Elements")
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(f"{output_path}/{mesh_name}.msh")
        gmsh.finalize()

    members = dict(zip(boundary_labels, boundary_tags))
    boundaries = Enum("boundaries", members)

    return uw.discretisation.Mesh(
        f"{output_path}/{mesh_name}.msh",
        degree=1,
        qdegree=mesh_degree,
        boundaries=boundaries,
        boundary_normals=None,
        coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
    )