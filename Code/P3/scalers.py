from sklearn.preprocessing import MinMaxScaler

class MyScaler(MinMaxScaler):
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        super().__init__(feature_range=feature_range, copy=copy, clip=clip)

    def inverse_std_transform(self, Xstd):
        """Undo the scaling of Xstd according to feature_range.

        Parameters
        ----------
        Xstd : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """

        Xtransformed = Xstd / self.scale_
        return Xtransformed