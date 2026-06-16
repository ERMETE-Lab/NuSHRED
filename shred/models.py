import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from IPython.display import clear_output as clc
from .processdata import mse, mre, num2p, weighted_mse

class SHRED(torch.nn.Module):
    """SHallow REcurrent Decoder (SHRED) network.

    Combines an LSTM encoder that processes sensor time-series with a
    fully-connected decoder that reconstructs the high-dimensional state.

    Architecture:
        LSTM  →  last hidden state  →  Linear(+Dropout+ReLU) × n  →  output
    """

    def __init__(self, input_size, output_size, hidden_size = 64, hidden_layers = 2, decoder_sizes = [350, 400], dropout = 0.0):
        """
        Args:
            input_size (int): Number of input features per time step (e.g. number of sensors).
            output_size (int): Dimension of the reconstructed state (e.g. full-order field size).
            hidden_size (int): Number of units in each LSTM hidden layer. Default: 64.
            hidden_layers (int): Number of stacked LSTM layers. Default: 2.
            decoder_sizes (list[int]): Widths of the intermediate decoder layers.
                The list is automatically prepended with ``hidden_size`` and
                appended with ``output_size``. Default: [350, 400].
            dropout (float): Dropout probability applied between decoder layers. Default: 0.0.
        """
            
        super(SHRED,self).__init__()

        self.lstm = torch.nn.LSTM(input_size = input_size,
                                  hidden_size = hidden_size,
                                  num_layers = hidden_layers,
                                  batch_first=True)
        
        self.decoder = torch.nn.ModuleList()
        decoder_sizes.insert(0, hidden_size)
        decoder_sizes.append(output_size)

        for i in range(len(decoder_sizes)-1):
            self.decoder.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
            if i != len(decoder_sizes)-2:
                self.decoder.append(torch.nn.Dropout(dropout))
                self.decoder.append(torch.nn.ReLU())

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        """Run a forward pass through the SHRED model.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch, lag, input_size)``.

        Returns:
            torch.Tensor: Reconstructed state of shape ``(batch, output_size)``.
        """
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float).to(x.device)
        c_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float).to(x.device)
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        _, (output, _) = self.lstm(x, (h_0, c_0))
        output = output[-1].view(-1, self.hidden_size)

        for layer in self.decoder:
            output = layer(output)

        return output

    def freeze(self):
        """Set the model to eval mode and disable gradient computation for all parameters."""
        self.eval()

        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Set the model to train mode and re-enable gradient computation for all parameters."""
        self.train()

        for param in self.parameters():
            param.requires_grad = True

def fit(model: torch.nn.Module, train_dataset: torch.utils.data.Dataset, valid_dataset: torch.utils.data.Dataset,
        batch_size: int = 64, epochs: int = 4000, optim: type = torch.optim.Adam, lr: float = 1e-3,
        loss_function: callable = mse, scaling_factor: torch.Tensor = None,
        verbose: bool = False, patience: int = 5):
    """Train a model with early stopping based on validation loss.

    Uses a closure-based optimizer step (compatible with second-order optimizers
    such as L-BFGS) and loads the best-seen parameters before returning.

    Args:
        model (torch.nn.Module): The network to train (modified in-place).
        train_dataset (torch.utils.data.Dataset): Training set.
        valid_dataset (torch.utils.data.Dataset): Validation set used for early stopping.
        batch_size (int): Mini-batch size. Default: 64.
        epochs (int): Maximum number of training epochs. Default: 4000.
        optim: Optimizer class (not an instance). Default: ``torch.optim.Adam``.
        lr (float): Learning rate. Default: 1e-3.
        loss_function: Callable ``(pred, target) -> scalar`` used during training.
            Default: ``mse``.
        scaling_factor: Feature-wise weights passed to ``weighted_mse``.
            Ignored unless ``loss_function`` is ``weighted_mse``. Default: None.
        verbose (bool): Print per-epoch progress to stdout. Default: False.
        patience (int): Number of epochs without validation improvement before
            stopping. Default: 5.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of per-epoch training and
        validation MRE, respectively.
    """

    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    optimizer = optim(model.parameters(), lr = lr)

    train_error_list = []
    valid_error_list = []
    patience_counter = 0
    best_params = model.state_dict()

    for epoch in range(1, epochs + 1):
        
        for k, data in enumerate(train_loader):
            model.train()
            def closure():
                outputs = model(data[0])
                optimizer.zero_grad()
                if scaling_factor is not None and loss_function == weighted_mse:
                    loss = loss_function(outputs, data[1], weights = scaling_factor)
                else:
                    loss = loss_function(outputs, data[1])
                loss.backward()
                return loss
            optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            train_error = mre(train_dataset.Y, model(train_dataset.X))
            valid_error = mre(valid_dataset.Y, model(valid_dataset.X))
            train_error_list.append(train_error)
            valid_error_list.append(valid_error)

        if verbose == True:
            print("Epoch "+ str(epoch) + ": Training loss = " + num2p(train_error_list[-1]) + " \t Validation loss = " + num2p(valid_error_list[-1]) + " "*10, end = "\r")

        if valid_error == torch.min(torch.tensor(valid_error_list)):
            patience_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter == patience:
            model.load_state_dict(best_params)
            train_error = mre(train_dataset.Y, model(train_dataset.X)).detach()
            valid_error = mre(valid_dataset.Y, model(valid_dataset.X)).detach()
            
            if verbose == True:
                print("Training done: Training loss = " + num2p(train_error) + " \t Validation loss = " + num2p(valid_error))
         
            return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()
    
    model.load_state_dict(best_params)
    train_error = mre(train_dataset.Y, model(train_dataset.X)).detach().item()
    valid_error = mre(valid_dataset.Y, model(valid_dataset.X)).detach().item()
    
    if verbose == True:
        print("Training done: Training loss = " + num2p(train_error) + " \t Validation loss = " + num2p(valid_error))

    return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()


def fit_memory_efficient(model: torch.nn.Module, train_dataset: torch.utils.data.Dataset, valid_dataset: torch.utils.data.Dataset,
        batch_size: int = 64, epochs: int = 4000, optim: type = torch.optim.Adam, lr: float = 1e-3,
        loss_function: callable = mse, scaling_factor: torch.Tensor = None,
        verbose: bool = False, patience: int = 5):
    """Train a model with early stopping, computing errors batch-by-batch to reduce peak memory.

    Unlike :func:`fit`, this variant never passes the entire dataset through the
    model at once, making it suitable for large datasets or GPU-constrained environments.
    Errors reported each epoch are averages of per-batch MRE values.

    Args:
        model (torch.nn.Module): The network to train (modified in-place).
        train_dataset (torch.utils.data.Dataset): Training set.
        valid_dataset (torch.utils.data.Dataset): Validation set used for early stopping.
        batch_size (int): Mini-batch size. Default: 64.
        epochs (int): Maximum number of training epochs. Default: 4000.
        optim: Optimizer class (not an instance). Default: ``torch.optim.Adam``.
        lr (float): Learning rate. Default: 1e-3.
        loss_function: Callable ``(pred, target) -> scalar`` used during training.
            Default: ``mse``.
        scaling_factor: Feature-wise weights passed to ``weighted_mse``.
            Ignored unless ``loss_function`` is ``weighted_mse``. Default: None.
        verbose (bool): Print per-epoch progress and early-stopping messages. Default: False.
        patience (int): Number of epochs without validation improvement before
            stopping. Default: 5.

    Returns:
        tuple[list[float], list[float]]: Per-epoch training and validation MRE histories.
    """
    # Data Loaders
    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle = False, batch_size = batch_size)

    # Optimizer
    optimizer = optim(model.parameters(), lr = lr)

    # Check device
    device = next(model.parameters()).device # model already on device

    # Error history
    train_error_history = []
    valid_error_history = []

    # Initialization for early stopping
    patience_counter = 0
    best_params = deepcopy(model.state_dict())
    best_valid_error = float('inf')

    for epoch in range(epochs):

        model.train()
        batch_train_errors = []

        # Training loop
        for x_batch, y_batch in train_loader:

            optimizer.zero_grad()

            # Compute outputs and loss
            outputs = model(x_batch.to(device))
            if scaling_factor is not None and loss_function == weighted_mse:
                loss = loss_function(outputs, y_batch.to(device), weights = scaling_factor)
            else:
                loss = loss_function(outputs, y_batch.to(device))
            loss.backward()
            optimizer.step()

            # Store batch training error
            batch_train_errors.append(mre(y_batch.to(device), outputs).item())

        # Average training error for the epoch
        epoch_train_error = sum(batch_train_errors) / len(batch_train_errors)
        train_error_history.append(epoch_train_error)

        # Validation loop
        model.eval()
        batch_valid_errors = []
        with torch.no_grad():

            for x_batch, y_batch in valid_loader:
                outputs = model(x_batch.to(device))
                batch_valid_errors.append(mre(y_batch.to(device), outputs).item())

        # Average validation error for the epoch
        epoch_valid_error = sum(batch_valid_errors) / len(batch_valid_errors)
        valid_error_history.append(epoch_valid_error)

        if verbose:
            print(f"Epoch {epoch+1}: Train Loss {epoch_train_error:.6f}, Valid Loss {epoch_valid_error:.6f}", end="\r")

        # Early stopping check
        if epoch_valid_error < best_valid_error:
            best_valid_error = epoch_valid_error
            best_params = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}. Best Valid Loss: {best_valid_error:.6f}")
                break

    # Load best model parameters
    model.load_state_dict(best_params)
    return train_error_history, valid_error_history
            
def forecast(forecaster: torch.nn.Module, input_data: torch.Tensor, steps: int, nsensors: int):
    """Autoregressively forecast sensor trajectories over time.

    At each step the model predicts the next sensor values, which are then
    fed back as the last row of the rolling input window for the following step.
    Parameter columns (indices ``nsensors:`` in the last dimension) remain fixed
    throughout the rollout.

    Args:
        forecaster (torch.nn.Module): Trained model that maps
            ``(batch, lag, nsensors+nparams)`` → ``(batch, nsensors)``.
        input_data (torch.Tensor): Seed window of shape
            ``(ntrajectories, lag, nsensors+nparams)``.  Modified in-place.
        steps (int): Number of future time steps to predict.
        nsensors (int): Number of sensor columns (leading columns in the last
            dimension); used to slice predictions back into the window.

    Returns:
        torch.Tensor: Predicted sensor values stacked along the time axis,
        shape ``(ntrajectories, steps, nsensors)``.
    """

    forecast = []
    for i in range(steps):
        forecast.append(forecaster(input_data))
        temp = input_data.clone()
        input_data[:,:-1] = temp[:,1:]
        input_data[:,-1, :nsensors] = forecast[i]

    return torch.stack(forecast, 1)

def predict_in_batches(model: torch.nn.Module, dataset: torch.utils.data.Dataset, batch_size=64):
    """Run inference on a dataset in mini-batches and return a single concatenated tensor.

    Keeps peak GPU memory bounded regardless of dataset size by processing one
    batch at a time and immediately moving each result back to CPU.

    Args:
        model (torch.nn.Module): Trained model in eval-compatible state.
        dataset (torch.utils.data.Dataset): Dataset to run inference on.
        batch_size (int): Number of samples per inference batch. Default: 64.

    Returns:
        torch.Tensor: Predictions for the full dataset, shape
        ``(len(dataset), output_size)``, always on CPU.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = next(model.parameters()).device
    
    model.eval() # Set model to evaluation mode
    predictions = []
    
    with torch.no_grad():
        for x_batch, _ in loader:
            # Move batch to the same device as the model (GPU)
            x_batch = x_batch.to(device)
            
            # Predict
            out = model(x_batch)
            
            # Move result back to CPU immediately to free up GPU memory
            predictions.append(out.cpu())
            
    # Stitch all batches back together into one tensor
    return torch.cat(predictions, dim=0)