import pandas as pd
import numpy as np
from functools import reduce
from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class TableData(pd.DataFrame):
    def __init__(self):
        super().__init__()
    

class ListData(list):
    def __init__(self):
        super().__init__()


class Data:
    """Fundamental object to share dataset and its meta information to use for training, evalutation, etc.
    This class is a common interface among modules for data analysis such as preparing, training, evaluation, and visualization.
    It takes pandas DataFrame as a main data type and contains additional meta attributes.
    It also opens common interfaces to simplify access to the core attributes for data processing.
    """
    def __init__(self, data=None, columns=None):
        if isinstance(data, pd.DataFrame):
            self._dataframe = data
        elif isinstance(data, np.ndarray):
            self._dataframe = pd.DataFrame(data, columns=columns)
        elif isinstance(data, Data):
            self.__deepcopy(data)
        elif data is None:
            self._dataframe = pd.DataFrame()
        else:
            # TDOD: check the data type and the shape in case of unsupported format.
            # raise ValueError(data)
            self._dataframe = pd.DataFrame(data)
        
        self.meta = dict()
        self.label = None
        self.labels = list()
        self.features = list()

    def __deepcopy(self, data:'Data'):
        # TODO: Need to update whenever Data updates
        self._dataframe = data._dataframe.copy(deep=True)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self._dataframe._values, dtype=dtype)

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataframe

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @property
    def shape(self) -> tuple:
        return self._dataframe.shape

    @property
    def columns(self) -> list:
        return self._dataframe.columns.tolist()

    def head(self, lines=5) -> pd.DataFrame:
        return self._dataframe.head(lines)

    def tail(self, lines=5) -> pd.DataFrame:
        return self._dataframe.tail(lines)

    def crop(self, rows:list, cols:list) -> 'Data':
        """Crops only the specified rows and cols
        It crops the data by various index types. The supported index types are: int, excel column index, Dataframe column names.
        Args:
            rows (list): a list of indices
            cols (list): a list of indices

        Returns:
            Data: new Data instance
        """
        # TODO: slice를 지원하도록 IndexList 확인 필요
        if rows is None:
            rows = [slice(None)]
        if cols is None:
            cols = [slice(None)]
        idx_rows = IndexList(rows, self._dataframe.shape[0])
        idx_cols = IndexList(cols, self._dataframe.shape[0])
        df = self._dataframe.iloc[idx_rows.list, idx_cols.list]
        return Data(df)

    def equals(self, data:'Data') -> bool:
        """Compares elementwisely

        Args:
            data (Data): Data to be compared

        Returns:
            bool: Equal or not
        """
        return self._dataframe.equals(data._dataframe)

    def na_ratio(self, axis='column'):
        """Get an array of NA ratio per axis

        Args:
            axis (str, optional): Axis(column or row) to get NA ratio. Defaults to 'column'.

        Returns:
            array: An array of NA ratio
        """
        axis = 0 if axis == 'column' else 1
        nNa = self._dataframe.isna().sum(axis)
        nTotal = self._dataframe.shape[axis]
        return (nNa / nTotal) * 100


class CropData(TransformerMixin, BaseEstimator):
    def __init__(self, rows=None, cols=None):
        self.rows = rows
        self.cols = cols

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X:pd.DataFrame, **transform_params) -> pd.DataFrame:
        return Data(X).crop(self.rows, self.cols).dataframe


class IndexList:
    # TODO: slice의 간단한 form을 지원하도록 검토
    # TODO: slice의 ':'을 지원하도록 검토
    # TODO: slice의 attribute가 mutable 하지않음. 검토필요

    len_alphabets = len("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(self, indices:list, refs:Union[int, list]):
        self._indices = indices
        self._elements = list()
        if isinstance(refs, int):
            self.maxlen = refs
            self.names = None
        else:
            self.maxlen = len(refs)
            self.names = refs

        self._elements = self.__convert(indices)

    @classmethod
    def isexcelcolumns(cls, cols) -> bool:
        return isinstance(cols, str) and len(cols) < 3 and cols.isupper()

    @classmethod
    def excel2idx(cls, colname:str) -> int:
        """
        A utility function to convert column names in excel to index. Index starts from zero.

        Args:
            colname (str): column name

        Returns:
            int: zero-based index
        """
        assert IndexList.isexcelcolumns(colname)
        return reduce(lambda acc, cur: acc * IndexList.len_alphabets + ord(cur) - ord('A') + 1, colname, 0) - 1

    def __convert(self, indices) -> list:
        elements = list()
        for elem in indices:
            if isinstance(elem, int):
                elements += [self.maxlen - 1] if elem == -1 else [elem]
            elif isinstance(elem, str):
                if self.isexcelcolumns(elem):
                    elements += [self.excel2idx(elem)]
                else:
                    elements += [self.names.index(elem)]
            elif isinstance(elem, slice):
                elements += self.__slice_to_list(elem)
            else:
                raise TypeError('Unsupported Element Types!')
        return list(dict.fromkeys(elements))

    def __slice_to_list(self, sli) -> list:
        start = self.excel2idx(sli.start) if self.isexcelcolumns(sli.start) else sli.start
        stop = self.excel2idx(sli.stop) if self.isexcelcolumns(sli.stop) else sli.stop
        indices = slice(start, stop, sli.step).indices(self.maxlen)
        return list(range[indices[0], indices[1], indices[2]])

    @property
    def list(self):
        return self._elements

    @list.setter
    def list(self, indices):
        self._indices = indices
        self._elements = self.__convert(indices)
        

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


class Datap(pd.DataFrame):
    def __init__(self, data=None):
        super().__init__(data)