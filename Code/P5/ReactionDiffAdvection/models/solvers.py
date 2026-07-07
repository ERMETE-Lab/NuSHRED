import dolfinx
import numpy as np
from dolfinx import fem
import dolfinx.fem.petsc
from dolfinx.fem import (Function, functionspace, assemble_scalar, form, 
                        locate_dofs_topological, dirichletbc)
from ufl import grad, inner, dot, div
import ufl
from petsc4py import PETSc
import basix.ufl
from mpi4py import MPI
from scipy.integrate import solve_ivp

def build_quadratic_reaction_tensor(num_species: int, reactions: list):
    """Build a (num_species, num_species, num_species) quadratic reaction tensor
    Q[i, j, k] from a list of elementary reactions, so that the RHS contribution
    to species i is -sum_{j,k} Q[i,j,k] * c_j * c_k.

    Each reaction is a dict:
        {"reactants": [j, k], "products": [i, ...], "rate": coeff}
    meaning c_j + c_k -react at rate `coeff`-> sum of listed products (mass-action
    kinetics: reaction rate = coeff * c_j * c_k). Reactants may repeat the same
    index (e.g. [0, 0] for a 2*c_0 -> ... self-reaction).

    This only guarantees that the loss/gain bookkeeping for THIS reaction is
    self-consistent (each reactant loses at the stated rate, each product gains
    at the stated rate) -- it does NOT guarantee sum_i c_i is conserved in
    general. For A + B -> C, dC/dt = +rate*c_A*c_B while dA/dt = dB/dt =
    -rate*c_A*c_B, so d(sum c_i)/dt = -rate*c_A*c_B != 0: mole count isn't
    conserved unless you assign matching stoichiometric mass weights to
    reactants/products yourself.
    """
    Q = np.zeros((num_species, num_species, num_species))
    for reaction in reactions:
        j, k = reaction["reactants"]
        rate = reaction["rate"]
        # Loss for each reactant species (2*rate for a self-reaction j == k,
        # since both "slots" of c_j * c_k are the same species)
        if j == k:
            Q[j, j, k] += 2 * rate
        else:
            Q[j, j, k] += rate
            Q[k, j, k] += rate
        # Gain for each product species
        for i in reaction["products"]:
            Q[i, j, k] -= rate
    return Q

class AdvReactDiff_HF():
    def __init__(self, N: int, physical_params: dict,
                 dirichlet: float = None):

        self.domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
        metadata = {"quadrature_degree": 4}
        self.dx = ufl.Measure('dx', domain=self.domain, metadata=metadata)

        # Create function space for the species
        self.num_species = physical_params["num_species"]
        self.P1 = basix.ufl.element("Lagrange", self.domain.topology.cell_name(), 1)
        spaces = list()
        for _ in range(self.num_species):
            spaces.append(self.P1)
        self.V = functionspace(self.domain, basix.ufl.mixed_element(spaces))

        # Create space for the velocity field
        self.Q = functionspace(self.domain, ("Lagrange", 1, (2,)))
        self.velocity = Function(self.Q)

        # Create source term function space
        self.Si = [Function(self.V.sub(i).collapse()[0]) for i in range(self.num_species)] # spatial profile
        self.Si_time = [fem.Constant(self.domain, PETSc.ScalarType(0.0)) for _ in range(self.num_species)] # time-dependent amplitude

        # Create old solution function
        self.u_old = Function(self.V)
        self.solution = Function(self.V)

        # Create Trial and Test functions
        self.u = ufl.TrialFunctions(self.V)
        self.v = ufl.TestFunctions(self.V)

        # Set physical parameters
        ## Diffusion coefficients
        if isinstance(physical_params["diffusion"], (float, int)):
            self.diff_coeffs = [physical_params["diffusion"]] * self.num_species
        elif isinstance(physical_params["diffusion"], list):
            self.diff_coeffs = physical_params["diffusion"] # list of diffusion coefficients for each species
            assert len(self.diff_coeffs) == self.num_species, "Length of diffusion coefficients must match number of species"
        elif isinstance(physical_params["diffusion"], np.ndarray):
            self.diff_coeffs = physical_params["diffusion"].tolist() # convert numpy array to list
            assert len(self.diff_coeffs) == self.num_species, "Length of diffusion coefficients must match number of species"
        else:
            raise ValueError("Diffusion coefficients must be a float, list, or numpy array")
        
        ## Linear reaction coefficients
        if isinstance(physical_params["linear_reaction"], np.ndarray):
            self.reaction_coeffs = np.atleast_2d(physical_params["linear_reaction"])
        else:
            raise ValueError("Linear reaction coefficients must be a numpy array")
        
        assert self.reaction_coeffs.shape == (self.num_species, self.num_species), "Linear reaction coefficients must be a square matrix of size num_species x num_species"

        ## Quadratic reaction coefficients
        if "quadratic_reaction" in physical_params:
            if isinstance(physical_params["quadratic_reaction"], np.ndarray):
                self.quadratic_reaction_coeffs = np.atleast_3d(physical_params["quadratic_reaction"])
            else:
                raise ValueError("Quadratic reaction coefficients must be a numpy array")
            assert self.quadratic_reaction_coeffs.shape == (self.num_species, self.num_species, self.num_species), "Quadratic reaction coefficients must be a 3D array of size num_species x num_species x num_species"

    
        if dirichlet is not None and isinstance(dirichlet, float):

            fdim = self.domain.topology.dim - 1
            boundary_facets = dolfinx.mesh.locate_entities_boundary(
                self.domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
            )
            ft = fem.locate_dofs_topological(self.V, fdim, boundary_facets)

            # Apply Dirichlet boundary conditions to all species
            self.bcs = []
            for i in range(self.num_species):
                self.bcs.append(
                    dirichletbc(PETSc.ScalarType(dirichlet), ft, self.V.sub(i))
                )
        else:
            self.bcs = None

    def assign_initial_conditions(self, initial_conditions: list):
        assert len(initial_conditions) == self.num_species, "Length of initial conditions must match number of species"
        for i in range(self.num_species):
            self.u_old.sub(i).interpolate(initial_conditions[i])

    def assign_velocity(self, scale = 1.0):
        def velocity_expression(x):
            # Double gyre: two counter-rotating cells, zero on all four edges (no-slip)
            return (
                scale * -np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
                scale * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
            )
        self.velocity.interpolate(velocity_expression)

    def assign_gaussian_source_terms(self, source_terms: list):
        # Each element of the list source_terms should have 3 elements: [sigma, x0, y0]
        for i, params in enumerate(source_terms):
            sigma, x0, y0 = params
            def gaussian_expression(x):
                return np.exp(-((x[0] - x0)**2 + (x[1] - y0)**2) / (2 * sigma**2))
            self.Si[i].interpolate(gaussian_expression)
    def assign_sinusoidal_source_terms(self, source_terms: list):
        # Each element of the list source_terms should have 3 elements: [frequency, x0, y0]
        for i, params in enumerate(source_terms):
            frequency, x0, y0 = params
            def sinusoidal_expression(x):
                return np.abs(np.sin(frequency * (x[0] - x0)) * np.sin(frequency * (x[1] - y0)))
            self.Si[i].interpolate(sinusoidal_expression)

    def assemble_form(self, dt, direct_solver=False):

        self.dt = dt
        
        # Left-Hand Side - Time derivative
        self.left_size = 1 / dt * sum([inner(self.u[i], 
                                             self.v[i]) * self.dx for i in range(self.num_species)])

        # Left-Hand Side - Diffusion
        self.left_size += sum([self.diff_coeffs[i] * inner(grad(self.u[i]), 
                                                           grad(self.v[i])) * self.dx for i in range(self.num_species)])

        # Left-Hand Side - Advection
        self.left_size += sum([inner(div(self.velocity * self.u[i]), 
                                     self.v[i]) * self.dx for i in range(self.num_species)])

        # Left-Hand Side - Linear Reaction
        self.left_size -= sum([sum([self.reaction_coeffs[i, j] * inner(self.u[j], 
                                                                       self.v[i]) * self.dx for j in range(self.num_species)]) for i in range(self.num_species)])

        # Left-Hand Side - Quadratic Reaction
        if hasattr(self, 'quadratic_reaction_coeffs'):
            self.left_size += sum([sum([sum([self.quadratic_reaction_coeffs[i, j, k] * inner(self.u[j] * self.u_old[k], 
                                                                                             self.v[i]) * self.dx for k in range(self.num_species)]) for j in range(self.num_species)]) for i in range(self.num_species)])

        # Right-Hand Side - Time derivative
        self.right_size = 1 / dt * sum([inner(self.u_old[i], 
                                              self.v[i]) * self.dx for i in range(self.num_species)])

        # Right-Hand Side - Source terms
        self.right_size += sum([self.Si_time[i] * inner(self.Si[i], 
                                                        self.v[i]) * self.dx for i in range(self.num_species)])

        # Assemble forms
        self.a = form(self.left_size)
        self.L = form(self.right_size)

        # Create matrix and vector
        self.A = dolfinx.fem.petsc.create_matrix(self.a)
        self.b = dolfinx.fem.petsc.create_vector(dolfinx.fem.extract_function_spaces(self.L))

        # Assign solver
        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.A)

        if direct_solver:
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)
        else:
            self.solver.setType(PETSc.KSP.Type.GMRES)
            self.solver.getPC().setType(PETSc.PC.Type.ILU)

    def advance(self, t: float, St_lambda: list):
        # Update source term amplitudes
        for i in range(self.num_species):
            self.Si_time[i].value = St_lambda[i](t)

        # Assemble the system
        self.A.zeroEntries()
        if self.bcs is not None:
            fem.petsc.assemble_matrix(self.A, self.a, bcs=self.bcs)
        else:
            fem.petsc.assemble_matrix(self.A, self.a)
        self.A.assemble()

        # Assemble the right-hand side vector
        self.b.zeroEntries()
        fem.petsc.assemble_vector(self.b, self.L)
        if self.bcs is not None:
            fem.petsc.apply_lifting(self.b, [self.a], bcs=[self.bcs])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(self.b, self.bcs)

        # Solve the linear system
        self.solver.solve(self.b, self.solution.x.petsc_vec)
        self.solution.x.scatter_forward()

        # Update the old solution for the next time step
        self.u_old.x.array[:] = self.solution.x.array[:]

        return [self.solution.sub(i).collapse() for i in range(self.num_species)]

class AdvReactDiff_ODE():
    def __init__(self, physical_params: dict, Si_avg: list):
        self.num_species = physical_params["num_species"]
        self.reaction_coeffs = physical_params["linear_reaction"]
        self.quadratic_reaction_coeffs = physical_params.get("quadratic_reaction", None)
        # Spatial average of each source profile, S_i = (1/|Omega|) * int_Omega Si(x) dx,
        # so that the lumped source S_i(t) = St_lambda[i](t) * Si_avg[i] matches the
        # HF model's source term averaged over the domain (see markdown derivation above).
        self.Si_avg = np.asarray(Si_avg)

    def rhs(self, t, c, St_lambda):
        # c is a 1D array of length num_species
        rhs = np.zeros_like(c)
        
        # Linear reactions
        rhs += self.reaction_coeffs @ c
        
        # Quadratic reactions
        if self.quadratic_reaction_coeffs is not None:
            for i in range(self.num_species):
                for j in range(self.num_species):
                    for k in range(self.num_species):
                        rhs[i] -= self.quadratic_reaction_coeffs[i, j, k] * c[j] * c[k]
        
        # Source terms
        for i in range(self.num_species):
            rhs[i] += St_lambda[i](t) * self.Si_avg[i]
        
        return rhs
    
    def solve(self, t_span, c0, St_lambda):
        sol = solve_ivp(self.rhs, t_span, c0, args=(St_lambda,), dense_output=True)
        return sol