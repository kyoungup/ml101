from ml101.preprocess import DataFilter
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import accumulate
from typing import Iterator
import pandas as pd
from ml101.data import Data


class Shift(BaseEstimator, TransformerMixin):
    def __init__(self, columns:list = None, move:int = 1, dropna=True, append=True, inplace=True):
        self.columns = columns
        self.move = move
        self.dropna = dropna
        self.append = append
        self.inplace = inplace

    def fit(self, X, y=None):
        # X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]
        # Return the transformer
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # X = super().transform(X)
        if self.columns is None:
            self.columns = X.columns

        prep = DataFilter(X)
        return prep.shift(self.columns, self.move, self.dropna, self.append, self.inplace)


class SetPeriod(BaseEstimator, TransformerMixin):
    def __init__(self, columns:list=None, outs:list=None, labelon=1, append=True, inplace=True):
        self.columns = columns
        self.outs = outs
        self.labelon = labelon
        self.append = append
        self.inplace = inplace

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        prep = DataFilter(X)
        outs = list()
        for col, out in zip(self.columns, self.outs):
            outs.append(pd.DataFrame(prep.period(X[col], self.labelon), columns=[out]))

        if self.append:
            result = pd.concat([X] + outs, axis=1)
        else:
            result = pd.concat(outs, axis=1)
        # result = Data(result)
        return result