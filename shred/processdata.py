import torch

mae = lambda datatrue, datapred: (datatrue - datapred).abs().mean()
mse = lambda datatrue, datapred: (datatrue - datapred).pow(2).sum(axis = -1).mean()
mre = lambda datatrue, datapred: ((datatrue - datapred).pow(2).sum(axis = -1).sqrt() / ((datatrue).pow(2).sum(axis = -1).sqrt() + 1e-8)).mean()
num2p = lambda prob : ("%.2f" % (100*prob)) + "%"

class TimeSeriesDataset(torch.utils.data.Dataset):
    '''
    Input: sequence of input measurements with shape (ntrajectories, ntimes, ninput) and corresponding measurements of high-dimensional state with shape (ntrajectories, ntimes, noutput)
    Output: Torch dataset
    '''

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len


def Padding(data, lag):
    '''
    Extract time-series of lenght equal to lag from longer time series in data, whose dimension is (number of time series, sequence length, data shape)
    '''
    
    data_out = torch.zeros(data.shape[0] * data.shape[1], lag, data.shape[2])

    for i in range(data.shape[0]):
        for j in range(1, data.shape[1] + 1):
            if j < lag:
                data_out[i * data.shape[1] + j - 1, -j:] = data[i, :j]
            else:
                data_out[i * data.shape[1] + j - 1] = data[i, j - lag : j]

    return data_out

def weighted_mse(datatrue, datapred, weights=None):
    """
    Compute MSE using scaling factor.

    Input:
        datatrue: true data, shape (nsamples, nfeatures)
        datapred: predicted data, shape (nsamples, nfeatures)
        weights: scaling factor, shape (nfeatures,)
    """
    
    if weights is None:
        weights = torch.ones(datatrue.shape[1], device=datatrue.device)

    diff = datatrue - datapred                     # single allocation
    return (diff.square() * weights).sum(dim=-1).mean()

class SmartTimeSeriesDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset that generates time-lagged windows on the fly instead of storing a massive padded tensor in memory.
    """
    def __init__(self, X, Y, lag):
        # X: (n_trajectories, n_timesteps, n_sensors)
        # Y: (n_trajectories, n_timesteps, n_outputs) where n_outputs can be high-dimensional or low-rank POD coeffs
        self.X = X
        self.Y = Y
        self.lag = lag
        self.n_traj, self.n_time, _ = X.shape
        assert self.n_traj == Y.shape[0] and self.n_time == Y.shape[1], "Mismatch in number of trajectories or time steps between X and Y."

        self.len = self.n_traj * self.n_time # Total number of samples

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # Map flat index to trajectory and time indices
        traj_idx = index // self.n_time
        time_idx = index % self.n_time

        # Calculate start and end for the window
        # We want the window ending at time_idx
        start_idx = time_idx - self.lag + 1
        end_idx = time_idx + 1

        if start_idx < 0:
            # If the window goes out of bounds (start of trajectory), pad with zeros
            # This replicates the zero-padding behavior of the original code
            valid_data = self.X[traj_idx, :end_idx]
            pad_size = self.lag - valid_data.shape[0]
            # Create zeros of the same type and device as X
            padding = torch.zeros((pad_size, *valid_data.shape[1:]), dtype=self.X.dtype, device=self.X.device)
            x_window = torch.cat([padding, valid_data], dim=0)
        else:
            x_window = self.X[traj_idx, start_idx:end_idx]

        y_val = self.Y[traj_idx, time_idx]

        return x_window, y_val