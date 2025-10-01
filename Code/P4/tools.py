import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd

class MyScaler():
    def __init__(self, scaler_type: str = 'minmax') -> None:
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError('Scaler type not supported.')

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the scaler to the data. The shape of the data is assumed to be (Nh, Ns).
        """

        self.Nh = data.shape[0]
        self.scaler.fit(data.T)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the data. The shape of the data is assumed to be (Nh, Ns).
        """

        return self.scaler.transform(data.T).T

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data. The shape of the data is assumed to be (Nh, Ns).
        """

        return self.scaler.inverse_transform(data.T).T

    def inverse_std_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data. The shape of the data is assumed to be (Nh, Ns).
        """

        if isinstance(self.scaler, StandardScaler):
            return (self.scaler.inverse_transform(data.T) - self.scaler.mean_).T
        else:
            return (self.scaler.inverse_transform(data.T) - self.scaler.data_min_).T