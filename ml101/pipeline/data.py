import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from ml101.data import Data


class CropData(TransformerMixin, BaseEstimator):
    def __init__(self, rows=None, cols=None):
        self.rows = rows
        self.cols = cols

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X:pd.DataFrame, **transform_params) -> pd.DataFrame:
        return Data(X).crop(self.rows, self.cols).dataframe


class DataMixin(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        # Keeps the original columns because sklearn convert it to numpy, which loses metadata like columns.
        if hasattr(self, 'columns_') is False or self.columns_ is None:
            if isinstance(X, Data) or isinstance(X, pd.DataFrame):
                self.comumns_ = X.columns
        if isinstance(self.estimator, TransformerMixin):
            self.estimator.fit(X.dataframe, y, **fit_params)
        return self

    def transform(self, X, **tranform_params):
        if isinstance(self.estimator, TransformerMixin):
            X = self.estimator.transform(X)
        elif callable(self.estimator):
            X = self.estimator(X)
        return Data(X, columns=self.comumns_)


class ArrayMixin(BaseEstimator, TransformerMixin):
    def __init__(self, estimator_func=None, columns=None):
        self.estimator_func = estimator_func

    def fit(self, X, y=None, **fit_params):
        # Keeps the original columns because sklearn convert it to numpy, which loses metadata like columns.
        if hasattr(self, 'columns_') is False or self.columns_ is None:
            if isinstance(X, Data) or isinstance(X, pd.DataFrame):
                self.comumns_ = X.columns
        if isinstance(self.estimator, TransformerMixin):
            self.estimator.fit(X.dataframe, y, **fit_params)
        return self

    def transform(self, X, **tranform_params):
        if isinstance(self.estimator, TransformerMixin):
            X = self.estimator.transform(X)
        elif callable(self.estimator):
            X = self.estimator(X)
        return Data(X, columns=self.comumns_)