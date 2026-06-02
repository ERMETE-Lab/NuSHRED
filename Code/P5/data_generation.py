from dolfinx.io import gmshio
import gmsh
from mpi4py import MPI
import copy
import numpy as np
import ufl
from IPython.display import clear_output

import sys
sys.path.append('models')

from neutr_diff import steady_neutron_diff, transient_neutron_diff

class DiffusionModel():

    def __init__(self, mesh_path='./',
                 mesh_factor=1.5):
        self.mesh_path = mesh_path

        # Generate mesh
        gdim = 2
        model_rank = 0
        mesh_comm = MPI.COMM_WORLD

        # Initialize the gmsh module
        gmsh.initialize()

        # Load the .geo file
        gmsh.merge(mesh_path+'LRA2D.geo')
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_factor)

        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.optimize("Netgen")

        # gmsh.write('../../NuSHRED_Datasets/D5/LRA2D.vtk')

        clear_output()

        # Domain
        domain, ct, ft = gmshio.model_to_mesh(gmsh.model, comm = mesh_comm, rank = model_rank, gdim = gdim )
        gmsh.finalize()

        self.domain1_marker = 10
        self.domain2_marker = 20
        self.domain3_marker = 30
        self.domain3b_marker = 35
        self.domain4_marker = 40
        self.domain5_marker = 50

        self.boundary_marker = 1

        tdim = domain.topology.dim
        fdim = tdim - 1

        self.ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
        self.dx = ufl.Measure("dx", domain=domain)

        domain.topology.create_connectivity(fdim, tdim)

        # Store domain
        self.domain = domain
        self.ct = ct
        self.ft = ft
        self.tdim = tdim
        self.fdim = fdim

    def extract_nodes(self):
        nodes = list()
        
        for ii in range(self.trans_pb.V.num_sub_spaces):
            nodes.append(
                self.trans_pb.V.sub(ii).collapse()[0].tabulate_dof_coordinates()
            )
        return nodes

    def set_parameters(self, path_mgxs = None, nu_value = 2.43, Ef = 1, reactor_power = 1):
        self.regions = [self.domain1_marker, self.domain2_marker, self.domain3_marker,
                        self.domain3b_marker, self.domain4_marker, self.domain5_marker]

        neutronics_param = dict()
        self.path_mgxs = path_mgxs

        if path_mgxs is None:
            neutronics_param['Energy Groups'] = 2

            neutronics_param['D'] = [np.array([1.255, 1.268,  1.259,  1.259,  1.259,  1.257]),
                                     np.array([0.211, 0.1902, 0.2091, 0.2091, 0.2091, 0.1592])]
            neutronics_param['xs_a'] = [np.array([0.008252, 0.007181, 0.008002, 0.008002, 0.008002, 0.0006034]),
                                        np.array([0.1003,   0.07047,  0.08344,  0.08344,  0.073324, 0.01911])]
            neutronics_param['nu_xs_f'] = [np.array([0.004602, 0.004609, 0.004663, 0.004663, 0.004663, 0.]),
                                           np.array([0.1091,   0.08675,  0.1021,   0.1021,   0.1021,   0.])]
            neutronics_param['xs_f'] = [neutronics_param['nu_xs_f'][i]/nu_value for i in range(neutronics_param['Energy Groups'])]
            neutronics_param['xs_s'] = [[np.array([[0] * len(self.regions)]).flatten(), np.array([0.02533, 0.02767, 0.02617, 0.02617, 0.02617, 0.04754])],
                                        [np.array([[0] * len(self.regions)]).flatten(), np.array([0] * len(self.regions)).flatten()]]
            neutronics_param['B2z'] = [np.array([[1e-4] * len(self.regions)]).flatten(),
                                       np.array([[1e-4] * len(self.regions)]).flatten()]
            neutronics_param['chi'] = [np.array([[1.0] * len(self.regions)]).flatten(),
                                       np.array([[0.0] * len(self.regions)]).flatten()]

            # Kinetic parameters
            neutronics_param['v'] = [3e7, 3e5] #cm/s
            neutronics_param['beta_l'] =  np.array([    [0.0054,   0.0054,   0.0054,   0.0054,   0.0054,   0.],
                                                        [0.001087, 0.001087, 0.001087, 0.001087, 0.001087, 0.]])
            neutronics_param['lambda_p_l'] =  np.array([ [0.0654, 0.0654, 0.0654, 0.0654, 0.0654, 0.],
                                                        [1.35,   1.35,   1.35,   1.35,   1.35,   0.]]) # 1/s


        else:

            ## How are the cross sections: 0 is fast or thermal?
            xs_lib_mc = np.load(path_mgxs+'xs_extracted.npy', allow_pickle=True).item()

            neutronics_param['Energy Groups'] = xs_lib_mc['diffusion-coefficient'].shape[1]

            neutronics_param['D'] = [xs_lib_mc['diffusion-coefficient'][:, i] for i in range(neutronics_param['Energy Groups'])]
            neutronics_param['xs_a'] = [xs_lib_mc['absorption'][:, i] for i in range(neutronics_param['Energy Groups'])]
            neutronics_param['nu_xs_f'] = [xs_lib_mc['nu-fission'][:, i] for i in range(neutronics_param['Energy Groups'])]
            neutronics_param['xs_f'] = [xs_lib_mc['fission'][:, i] for i in range(neutronics_param['Energy Groups'])]
            neutronics_param['xs_s'] = [[xs_lib_mc['scatter matrix'][:, g, gp] for gp in range(neutronics_param['Energy Groups'])] for g in range(neutronics_param['Energy Groups'])]
            neutronics_param['B2z'] = [np.array([[0.] * len(self.regions)]).flatten() for _ in range(neutronics_param['Energy Groups'])]
            neutronics_param['chi'] = [xs_lib_mc['chi'][:, i] for i in range(neutronics_param['Energy Groups'])]

            # Kinetic parameters
            neutronics_param['v'] = (1/xs_lib_mc['inverse-velocity']).mean(axis=0) #cm/s

            delay_lib_mc = np.load(path_mgxs+'prec_extracted.npy', allow_pickle=True).item()

            neutronics_param['beta_l'] = [delay_lib_mc['beta_mat'][:, i] for i in range(delay_lib_mc['beta_mat'].shape[1])]
            neutronics_param['lambda_p_l'] = [delay_lib_mc['decay-rate_mat'][:, i] for i in range(delay_lib_mc['decay-rate_mat'].shape[1])]

        # Store parameters
        self.neutronics_param = neutronics_param
        self.nu_value = nu_value
        self.Ef = Ef
        self.reactor_power = reactor_power

    def criticality_calculation(self, albedo = None, LL = 10, maxIter = 500, verbose=True):

        self.ss_pb = steady_neutron_diff(self.domain, self.ct, self.ft, self.neutronics_param, self.regions, self.boundary_marker,
                                         albedo=albedo)

        # Assemble problem
        self.ss_pb.assembleForm()

        # Solve Eigenvalue problem
        phi_ss, k_eff = self.ss_pb.solve(power = self.reactor_power, nu=self.nu_value, Ef = self.Ef,
                                         LL=LL, maxIter=maxIter, verbose=verbose)

        # Make reactor critical
        self.neutronics_param['k_eff_0'] = k_eff
        self.neutronics_param['nu_xs_f'] = [self.neutronics_param['nu_xs_f'][0] / k_eff,
                                            self.neutronics_param['nu_xs_f'][1] / k_eff]
        self.critical_flux = phi_ss

        if verbose:
            print(f'Criticality calculation: k_eff = {k_eff:.6f}')


    def calculate_perturbation(self, _reduction_xsa_10):

        _perturbed_neutr_param = copy.deepcopy(self.neutronics_param) # self.neutronics_param.copy()
        _perturbed_neutr_param['xs_a'][0][3] *= _reduction_xsa_10

        _pertub_ss_diff = steady_neutron_diff(self.domain, self.ct, self.ft, _perturbed_neutr_param, self.regions, self.boundary_marker)
        _pertub_ss_diff.assembleForm()

        _keff = _pertub_ss_diff.solve(power = self.reactor_power, Ef=self.Ef, nu = self.nu_value,
                                    LL = 10, maxIter = 500, verbose=False)[1]
        _rho = 1 - (1 / _keff)

        return _keff, _rho
    
    def define_transient_pb(self, dt = 1e-3, albedo = None):

        self.trans_pb = transient_neutron_diff(self.domain, self.ct, self.ft, self.neutronics_param, self.regions, self.boundary_marker,
                                               albedo=albedo)
        self.dt = dt

    def solve_transient(self, change_xsa_10: callable, t_end = 0.5,
                        save_every = 1,
                        verbose=True):
        
        # Set critical flux as initial condition
        self.trans_pb.assembleForm(self.critical_flux, self.dt, nu = self.nu_value, Ef = self.Ef)

        # Initialize time
        t = 0.0

        # Define perturbed XS
        xs_a1_trans = lambda t: np.array([  self.neutronics_param['xs_a'][0][0], 
                                            self.neutronics_param['xs_a'][0][1],
                                            self.neutronics_param['xs_a'][0][2],
                                            self.neutronics_param['xs_a'][0][3] * change_xsa_10(t),
                                            self.neutronics_param['xs_a'][0][4],
                                            self.neutronics_param['xs_a'][0][5]])
        
        xs_a2_trans = lambda t: np.array([  self.neutronics_param['xs_a'][1][0],
                                            self.neutronics_param['xs_a'][1][1],
                                            self.neutronics_param['xs_a'][1][2],
                                            self.neutronics_param['xs_a'][1][3],
                                            self.neutronics_param['xs_a'][1][4],
                                            self.neutronics_param['xs_a'][1][5]])
        xs_a_trans = [xs_a1_trans, xs_a2_trans]

        # Prepare output variables
        snapshots = {
            'phi1': list(),
            'phi2': list(),
            'c1': list(),
            'c2': list(),
        }
        integral_qties = list()
        times = list()

        # Store initial condition
        times.append(t)
        integral_qties.append(
                np.concatenate((np.array([1.0]), 
                               np.array([self.trans_pb.old.sub(ii).collapse().x.array[:].mean() 
                                         for ii in range(self.trans_pb.G + self.trans_pb.prec_groups)]) )),
                               )

        num_steps = int(t_end / self.dt)

        for step in range(num_steps):
            t += self.dt

            _res = self.trans_pb.advance(t, xs_a_trans, return_prec = True)


            if step % save_every == 0:

                # Store snapshots
                for ii, field in enumerate(snapshots.keys()):
                    snapshots[field].append(_res[1][ii].x.array[:])

                # Store integral quantities
                times.append(t)
                integral_qties.append(
                    np.concatenate((np.array([_res[0]]), 
                                np.array([np.mean(snapshots[field][-1]) for field in snapshots.keys()])))
                                )

            if verbose:
                print(f'Time step {step+1}/{num_steps} completed. Time: {t:.4f} s', end='\r')

        times = np.array(times)
        integral_qties = np.array(integral_qties)
        snapshots = {
                        field: np.array(snapshots[field]) for field in snapshots.keys()
                    }
        
        return snapshots, integral_qties, times


###################################################################################################

from scipy.integrate import solve_ivp

# Point kinetics equations
def PK(t, y, rho_, betas: list, lambdas_prec: list, Lambda):
    rho = rho_(t)
    beta = sum(betas)
    lambdas = np.array(lambdas_prec)

    # Power equation
    power_eqn = (rho - beta) / Lambda * y[0] + np.sum(lambdas * y[1:])

    # Precursor equations
    prec_eqn = [
        betas[i]/Lambda * y[0] - lambdas[i] * y[i+1]
        for i in range(len(betas))
    ]

    return [power_eqn] + prec_eqn


class PointKinetics():
    def __init__(self, rtol = 1e-8, atol = 1e-10, method = 'RK45',
                 path_mgxs = None, Lambda: float = None):
        # Integrator options
        self.integrator_keywords = {
            'rtol': rtol,
            'atol': atol,
            'method': method
        }

        # Default parameters from diffusion model
        if path_mgxs is None:
            self.betas = np.array([0.0054, 0.001087])
            self.lambdas_prec = np.array([0.0654, 1.35])
            self.Lambda = 1e-6 # s
   
        else:
            dec_mgxs = np.load('MGXS/prec_extracted.npy', allow_pickle=True).item()
            self.betas = dec_mgxs['beta_glob']
            self.lambdas_prec = dec_mgxs['lambda_glob']
            self.Lambda = dec_mgxs['inv_vel_glob'][0] if Lambda is None else Lambda

    def solve(self, rho_t, t_eval):

        # Initial conditions
        ic = np.array([1.0] + [self.betas[i]/(self.lambdas_prec[i]*self.Lambda) for i in range(len(self.betas))])

        # Solve ODEs
        t_end = t_eval[-1]
        pk_model = solve_ivp(
                                PK, (0, t_end), ic,
                                args=(rho_t, self.betas, self.lambdas_prec, self.Lambda),
                                dense_output=True,
                                **self.integrator_keywords
                            )
        
        return pk_model.sol(t_eval).T