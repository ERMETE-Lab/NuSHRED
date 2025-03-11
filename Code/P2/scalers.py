import numpy as np
import torch

class ICMinMaxScaler():
    def __init__(self, u: np.ndarray):
        
        self.Nh = u.shape[0]
        
        self._min = u[:,0].min()
        self._max = u[:,0].max()
        
    def transform(self, X):
        
        assert X.shape[0] == self.Nh
        
        scaled_X = (X - self._min) / (self._max - self._min)
        return scaled_X
    
    def inverse_transform(self, X):
        
        assert X.shape[0] == self.Nh
        
        unscaled_X = X * (self._max - self._min) + self._min
        return unscaled_X
    
    def inverse_std_transform(self, std):
        
        assert std.shape[0] == self.Nh
        
        unscaled_std = std * (self._max - self._min)
        return unscaled_std


class ICStdScaler():
    def __init__(self, u: np.ndarray):
        
        self.Nh = u.shape[0]
        
        self._mean = np.mean(u[:,0])
        self._std  = np.std(u[:,0])
        
    def transform(self, X):
        
        assert X.shape[0] == self.Nh
        
        scaled_X = (X - self._mean) / self._std
        return scaled_X
    
    def inverse_transform(self, X):
        
        assert X.shape[0] == self.Nh
        
        unscaled_X = X * self._std + self._mean
        return unscaled_X
    
    def inverse_std_transform(self, std):
        
        assert std.shape[0] == self.Nh
        
        unscaled_std = std * self._std
        return unscaled_std