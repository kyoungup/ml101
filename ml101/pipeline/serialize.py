from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
from ml101.data import Data
from ml101.serialize import Stream


class ReadStream(BaseEstimator, TransformerMixin):
    def __init__(self, path2file=None, header=None, sheet_name=None):
        self.path2file = path2file
        # BaseEstimator does not allow **kwargs for set_params, which means it does not work with CVs
        # Explicitly declare all candidate params and filter only used ones by __kwargs().
        self.header = header
        self.sheet_name = sheet_name

    def __kwargs(self):
        kwargs = dict()
        if self.header: kwargs['header'] = self.header
        if self.sheet_name: kwargs['sheet_name'] = self.sheet_name

        return kwargs

    def fit(self, X=None, y=None):
        return self

    def transform(self, X:pd.DataFrame=None, **kwargs) -> pd.DataFrame:
        stream = Stream(self.path2file)
        data = stream.read(**self.__kwargs())
        return data.dataframe


class WriteStream(BaseEstimator, TransformerMixin):
    def __init__(self, path2file=None, append=None, header=None, sheet_name=None):
        self.path2file = path2file
        # BaseEstimator does not allow **kwargs for set_params, which means it does not work with CVs
        # Explicitly declare all candidate params and filter only used ones by __kwargs().
        self.append = append
        self.header = header
        self.sheet_name = sheet_name

    def __kwargs(self):
        kwargs = dict()
        if self.append: kwargs['append'] = self.append
        if self.header: kwargs['header'] = self.header
        if self.sheet_name: kwargs['sheet_name'] = self.sheet_name

        return kwargs

    def fit(self, X=None, y=None):
        return self

    def transform(self, X:pd.DataFrame, **kwargs) -> pd.DataFrame:
        stream = Stream(self.path2file)
        stream.write(Data(X), **self.__kwargs())
        return X