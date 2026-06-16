import torch

# Metric helpers — all operate on tensors of arbitrary batch dimensions.
# The last axis is treated as the feature/state dimension.
mae     = lambda datatrue, datapred: (datatrue - datapred).abs().mean()
# Mean Squared Error: averaged over the batch after summing squared residuals per sample.
mse     = lambda datatrue, datapred: (datatrue - datapred).pow(2).sum(axis=-1).mean()
# Mean Relative Error: L2 norm of residuals divided by L2 norm of targets (+eps for stability).
mre     = lambda datatrue, datapred: ((datatrue - datapred).pow(2).sum(axis=-1).sqrt() / (datatrue.pow(2).sum(axis=-1).sqrt() + 1e-8)).mean()
# Format a scalar in [0,1] as a percentage string with two decimal places.
num2p   = lambda prob: ("%.2f" % (100 * prob)) + "%"

class TimeSeriesDataset(torch.utils.data.Dataset):
    """Thin wrapper around a pre-computed (X, Y) pair for use with DataLoader.

    Expects inputs that have already been padded/windowed (e.g. via :func:`Padding`),
    so that each sample in ``X`` is one complete lag window.

    Args:
        X (torch.Tensor): Input tensor of shape ``(nsamples, lag, ninput)``.
        Y (torch.Tensor): Target tensor of shape ``(nsamples, noutput)``.
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len


def Padding(data, lag):
    """Slide a fixed-length window over each trajectory and zero-pad the leading edge.

    For each trajectory and each time step ``t``, extracts the window
    ``data[traj, max(0, t-lag+1):t+1]`` and left-pads with zeros when
    ``t < lag - 1``.  The result has one row per (trajectory, time) pair.

    Args:
        data (torch.Tensor): Input of shape
            ``(n_trajectories, sequence_length, n_features)``.
        lag (int): Length of each output window.

    Returns:
        torch.Tensor: Windowed tensor of shape
        ``(n_trajectories * sequence_length, lag, n_features)``.
    """
    
    data_out = torch.zeros(data.shape[0] * data.shape[1], lag, data.shape[2])

    for i in range(data.shape[0]):
        for j in range(1, data.shape[1] + 1):
            if j < lag:
                data_out[i * data.shape[1] + j - 1, -j:] = data[i, :j]
            else:
                data_out[i * data.shape[1] + j - 1] = data[i, j - lag : j]

    return data_out

def weighted_mse(datatrue: torch.Tensor, datapred: torch.Tensor, weights: torch.Tensor = None):
    """Compute a feature-weighted Mean Squared Error.

    Each feature's squared residual is multiplied by its corresponding weight
    before summing, allowing the loss to emphasise specific output dimensions.

    Args:
        datatrue (torch.Tensor): Ground-truth values, shape ``(nsamples, nfeatures)``.
        datapred (torch.Tensor): Model predictions, shape ``(nsamples, nfeatures)``.
        weights (torch.Tensor | None): Per-feature scaling factors, shape
            ``(nfeatures,)``.  Defaults to uniform weights (all ones) when None.

    Returns:
        torch.Tensor: Scalar weighted MSE.
    """
    
    if weights is None:
        weights = torch.ones(datatrue.shape[1], device=datatrue.device)

    diff = datatrue - datapred                     # single allocation
    return (diff.square() * weights).sum(dim=-1).mean()

class SmartTimeSeriesDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that builds lag windows on-the-fly from raw trajectories.

    Rather than materialising the full ``(n_traj * n_time, lag, n_features)`` tensor
    upfront (as :func:`Padding` does), windows are sliced from the stored trajectories
    at ``__getitem__`` time.  This trades a small per-sample CPU overhead for a large
    reduction in peak memory, which matters when ``lag`` or ``n_features`` is large.

    The windowing behaviour matches :func:`Padding`: windows that extend before the
    start of a trajectory are left-padded with zeros.

    Args:
        X (torch.Tensor): Sensor trajectories, shape
            ``(n_trajectories, n_timesteps, n_sensors)``.
        Y (torch.Tensor): Target trajectories, shape
            ``(n_trajectories, n_timesteps, n_outputs)``.  ``n_outputs`` may be
            the full-order state dimension or low-rank POD coefficients.
        lag (int): Length of the input window fed to the model.
    """
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, lag: int):
        self.X = X
        self.Y = Y
        self.lag = lag
        self.n_traj, self.n_time, _ = X.shape
        assert self.n_traj == Y.shape[0] and self.n_time == Y.shape[1], "Mismatch in number of trajectories or time steps between X and Y."

        self.len = self.n_traj * self.n_time

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # Unravel the flat sample index into (trajectory, time-step) coordinates.
        traj_idx = index // self.n_time
        time_idx = index % self.n_time

        start_idx = time_idx - self.lag + 1
        end_idx = time_idx + 1

        if start_idx < 0:
            # Window reaches before the start of the trajectory — zero-pad the left.
            valid_data = self.X[traj_idx, :end_idx]
            pad_size = self.lag - valid_data.shape[0]
            padding = torch.zeros((pad_size, *valid_data.shape[1:]), dtype=self.X.dtype, device=self.X.device)
            x_window = torch.cat([padding, valid_data], dim=0)
        else:
            x_window = self.X[traj_idx, start_idx:end_idx]

        y_val = self.Y[traj_idx, time_idx]

        return x_window, y_val