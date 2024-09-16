import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio


def evol_mesh(mesh_factor = 0.011):
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0

    # Initialize the gmsh module
    gmsh.initialize()

    # Load the .geo file
    gmsh.merge('../EVOL_geom.geo')
    gmsh.model.geo.synchronize()

    # Set algorithm (adaptive = 1, Frontal-Delaunay = 6)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_factor)
    gdim = 2

    # Linear Finite Element
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.optimize("Netgen")

    # Import into dolfinx
    model_rank = 0
    domain, ct, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, model_rank, gdim = gdim )

    domain.topology.create_connectivity(gdim, gdim)
    domain.topology.create_connectivity(gdim-1, gdim)

    # Finalize the gmsh module
    gmsh.finalize()
    
    return domain, ct, ft