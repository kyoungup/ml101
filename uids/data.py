import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class TableData(pd.DataFrame):
    def __init__(self):
        super().__init__()
    

class ListData(list):
    def __init__(self):
        super().__init__()


class Data:
    def __init__(self, data=None, columns=None):
        if isinstance(data, pd.DataFrame):
            self._dataframe = data
        elif isinstance(data, np.ndarray):
            self._dataframe = pd.DataFrame(data, columns=columns)
        elif isinstance(data, Data):
            self.copy(data)
        else:
            self._dataframe = pd.DataFrame()

    def __array__(self, dtype=None) -> np.ndarray:
        # return np.asarray(self._dataframe._values, dtype=dtype)
        return self._dataframe.__array__(dtype)

    def copy(self, data):
        self._dataframe = data._dataframe.copy(deep=True)

    @property
    def dataframe(self):
        return self._dataframe

    @property
    def columns(self) -> list:
        return self._dataframe.columns.tolist()


class Datap(pd.DataFrame):
    def __init__(self, data=None):
        super().__init__(data)


class DataMixin(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        # Keeps the original columns
        if hasattr(self, 'columns_') is False or self.columns_ is None:
            if isinstance(X, Data) or isinstance(X, pd.DataFrame):
                self.columns_ = X.columns

        self.estimator.fit(X.dataframe)
        return self

    def transform(self, X):
        X = self.estimator.transform(X)
        return Data(X, columns=self.columns_)