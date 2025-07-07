import numpy as np
import os
from sklearn.utils.extmath import randomized_svd
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from tqdm import tqdm
import pickle

def list_npz_files(directory):
    return [file for file in os.listdir(directory) if file.endswith('.npz')]

def import_data(npz_files, field):

    u = list()
    s = list()
    vh = list()

    for ii in tqdm(range(len(npz_files)), 'Importing '+field):

        filename = npz_files[ii]

        data = np.load(filename, allow_pickle=True)['arr_0'].item()
        
        u.append(data['u'][field])
        s.append(data['s'][field])
        vh.append(data['vh'][field])

        if ii == 0:
            fom_times = data['time']

        del data
            
    return u, s, vh, fom_times

class SVDLargeDataset():
    def __init__(self):
        pass

    def randomized(self, data: dict, rank: int, 
                   input_svd = True, scaler = None):

        if input_svd:
            Np = data['U'].shape[0]
            assert Np == data['S'].shape[0]
            assert Np == data['Vh'].shape[0]

            Nh = data['U'].shape[1]
            Nt = data['Vh'].shape[2]

            _data = list()
            for ii in range(Np):
                _data.append( (data['U'][ii] @ np.diag(data['S'][ii]) @ data['Vh'][ii]).T )
            _data = np.asarray(_data).reshape(-1, Nh).T

            assert _data.shape[0] == Nh
            assert _data.shape[1] == Nt * Np

        else:
            _data = data['snaps']

        if scaler is not None:
            self.scaler = scaler(_data)
            _data = self.scaler.transform(_data)

        U, S, Vh = randomized_svd(_data, n_components=rank, n_iter='auto')
        return U, S, Vh

    def project(self, data_to_project: dict, U: np.ndarray, S: np.ndarray, 
                input_svd = True, compute_errors = True):
        
        if input_svd:
            Np = data_to_project['U'].shape[0]
            assert Np == data_to_project['S'].shape[0]
            assert Np == data_to_project['Vh'].shape[0]

            Nh = data_to_project['U'].shape[1]
            Nt = data_to_project['Vh'].shape[2]

        else:
            Np = data_to_project['snaps'].shape[0]
            Nt = data_to_project['snaps'].shape[1]
            Nh = data_to_project['snaps'].shape[2]

        rank = U.shape[1]
        assert S.shape[0] == rank
        vh_pod = np.zeros((Np, Nt, rank))

        if compute_errors:
            errors = np.zeros((Np, Nt))
        else:
            errors = None

        for pp in range(Np):
            if input_svd:
                _data = (data_to_project['U'][pp] @ np.diag(data_to_project['S'][pp]) @ data_to_project['Vh'][pp])
            else:
                _data = data_to_project['snaps'][pp]

            assert _data.shape[0] == Nh, f"Expected {Nh}, got {_data.shape[0]}"
            assert _data.shape[1] == Nt, f"Expected {Nt}, got {_data.shape[1]}"

            if self.scaler is not None:
                _data = self.scaler.transform(_data)

            vh_pod[pp] = (np.linalg.inv(np.diag(S)) @ U.T @ _data).T

            if compute_errors:
                error = np.linalg.norm(_data - self.reconstruct(vh_pod[pp].T, U, S), axis=0) / np.linalg.norm(_data, axis=0)
                errors[pp] = error

        return vh_pod, errors

    def reconstruct(self, vh: np.ndarray, U: np.ndarray, S: np.ndarray):
        assert vh.shape[0] == U.shape[1]
        assert vh.shape[0] == S.shape[0]

        return U @ np.diag(S) @ vh

class LagrangianTransport_msfr():
    def __init__(self, nodes, directory_path: str, times: np.ndarray, scaler):
        
        self.nodes = np.stack([nodes[:,0], nodes[:,1]]).T
        self.path = directory_path
        self.times = times
        self.scaler = scaler

        self.Nt = len(times)

    def load_u_snapshots(self, mu_idx):

        u_data  = pickle.load(open(self.path + f'CompressedDataset/pod_basis_U.svd', 'rb'))
        s_data  = pickle.load(open(self.path + f'CompressedDataset/sing_vals_U.svd', 'rb'))
        vh_data = pickle.load(open(self.path + f'CompressedDataset/v_POD_all_fields.svd', 'rb'))['U'][mu_idx]

        assert vh_data.shape[0] == len(self.times)

        # Load the scaler

        _snap = self.scaler.inverse_transform(u_data @ s_data @ vh_data.T).T
        _snap = np.asarray([_fom.reshape(-1, 3) for _fom in _snap])
        
        return np.transpose(_snap[:, :, [0,2]], [2, 1, 0])  # x and z components only

    def interpolate_velocity(self,  position, velocity, time_idx,
                                    sampling_mesh = 1, method = 'linear'):

        if method == 'nearest':
            interpolator = NearestNDInterpolator(self.nodes[::sampling_mesh], velocity[:, ::sampling_mesh, time_idx].T)
        else:
            interpolator = LinearNDInterpolator(self.nodes[::sampling_mesh], velocity[:, ::sampling_mesh, time_idx].T)

        return interpolator(position)

    def calculate_trajectory(self, x0, velocity, 
                             sampling_mesh = 1,
                             method = 'linear', verbose = True):

            trajectory = np.zeros((self.Nt, 2))
            trajectory[0] = x0

            if verbose:
                bar = tqdm(range(1, self.Nt), desc='Calculating trajectory')

            for i in range(1, self.Nt):
                interpolated_velocity = self.interpolate_velocity(trajectory[i-1], velocity, i-1, 
                                                                  sampling_mesh = sampling_mesh, method = method)

                trajectory[i] = trajectory[i-1] + interpolated_velocity * (self.times[i] - self.times[i-1])
    
                if verbose: 
                    bar.update(1)


            if verbose:
                 bar.close()
            return trajectory



class IncrementalSVD():
    def __init__(self, initial_dataset: dict, rank: int, input_svd = True, scaler = None):
        if input_svd:
            _inital_dataset = np.linalg.multi_dot([ initial_dataset['U'],
                                                    np.diag(initial_dataset['S']),
                                                    initial_dataset['Vh']])
        else:
            _inital_dataset = initial_dataset['snaps']

        if scaler is not None:
            self.scaler = scaler(_inital_dataset)
            _inital_dataset = self.scaler.transform(_inital_dataset)
        else:
            self.scaler = None

        self.num_snapshots = _inital_dataset.shape[1]
        self.Nh = _inital_dataset.shape[0]

        self.U, _S, self.Vh = randomized_svd(_inital_dataset, n_components=rank)
        self.S = np.diag(_S)
        self.rank = rank

    def eigen_decon(self, new_data):
        L = self.U.T @ new_data
        num_new_snaps = new_data.shape[1]
        
        assert L.shape[0] == self.rank
        assert L.shape[1] == num_new_snaps

        return L, num_new_snaps
    
    def qr_decomp(self, L, new_data):
        H = new_data - self.U @ L

        J, K = np.linalg.qr(H)

        return J,K

    def compute_new_svd(self, L, K):

        Q = np.vstack([ np.hstack([self.S, L]), 
                        np.hstack([np.zeros((K.shape[0], self.S.shape[1])), K])])

        Uplus, Splus, Vhplus = randomized_svd(Q, n_components=self.rank)

        return Uplus, Splus, Vhplus

    def update_svd_basis(self, J, Uplus, Splus, Vhplus):
        self.U = np.hstack([self.U, J]) @ Uplus
        self.S = np.diag(Splus)
        
        tmp_V = np.vstack([ np.hstack([self.Vh.T, np.zeros((self.Vh.shape[1], J.shape[1]))]), 
                            np.hstack([np.zeros((J.shape[1], self.Vh.shape[0])), np.eye(J.shape[1])])])

        self.Vh = (tmp_V @ Vhplus.T).T

    def check_svd(self):
        assert self.U.shape[0] == self.Nh
        assert self.U.shape[1] == self.rank
        assert self.S.shape[0] == self.rank
        assert self.S.shape[1] == self.rank
        assert self.Vh.shape[0] == self.rank
        assert self.Vh.shape[1] == self.num_snapshots

    def update(self, new_data: dict, input_svd = True):
        
        if input_svd:
            _new_data = np.linalg.multi_dot([new_data['U'], 
                                             np.diag(new_data['S']), 
                                             new_data['Vh']])
        else:
            _new_data = new_data['snaps']

        if self.scaler is not None:
            _new_data = self.scaler.transform(_new_data)

        # Step 1: eigen-decom (projection onto the existing POD modes)
        L, num_new_snaps = self.eigen_decon(_new_data)

        # Step 2: non-orthogonal components over the existing POD modes
        J, K = self.qr_decomp(L, _new_data)

        # Step 3: new SVD
        Uplus, Splus, Vhplus = self.compute_new_svd(L, K)

        # Step 4: update the SVD
        self.update_svd_basis(J, Uplus, Splus, Vhplus)

        # Check SVD
        self.num_snapshots += num_new_snaps
        self.check_svd()

    def project(self, data: dict, input_svd=True, compute_errors = True):

        if input_svd:
            _data = np.linalg.multi_dot([data['U'], 
                                             np.diag(data['S']), 
                                             data['Vh']])
        else:
            _data = data['snaps']

        if self.scaler is not None:
            scaled_data = self.scaler.transform(_data)

        # Compute the projection of the data onto the POD modes
        v_pod = np.linalg.inv(self.S) @ self.U.T @ scaled_data
       
        if compute_errors:
            recon = self.U @ self.S @ v_pod
            error = np.linalg.norm(scaled_data - recon, axis=0) / np.linalg.norm(scaled_data, axis=0)

            return v_pod.T, error.max(), error.mean()
        else:
            return v_pod.T
        
class HierarchicalSVD():
    def __init__(self, rank):
        self.rank = rank
        
        self.U = None
        self.S = None
    def update(self, new_data: dict, input_svd = True):

        if input_svd:
            Uplus = new_data['U']
            Splus = new_data['S'] # suppose to be a vector
        else:
            Uplus, Splus, _ = randomized_svd(new_data['snaps'], n_components=self.rank, n_iter='auto')

        if self.U is None:
            self.U, self.S = Uplus[:, :self.rank], Splus[:self.rank]
            self.Nh = self.U.shape[0]
        else:
            _A = np.hstack([self.U @ np.diag(self.S), Uplus @ np.diag(Splus)])
            self.U, self.S, _ = randomized_svd(_A, n_components=self.rank, n_iter='auto')

    def project(self, data: dict, input_svd=True, compute_errors = True):

        if input_svd:
            _data = np.linalg.multi_dot([data['U'], 
                                             np.diag(data['S']), 
                                             data['Vh']])
        else:
            _data = data['snaps']

        # Compute the projection of the data onto the POD modes
        v_pod = np.linalg.inv(np.diag(self.S)) @ self.U.T @ _data
       
        if compute_errors:
            recon = self.U @ np.diag(self.S) @ v_pod
            error = np.linalg.norm(_data - recon, axis=0) / np.linalg.norm(_data, axis=0)

            return v_pod.T, error.max(), error.mean()
        else:
            return v_pod.T