import pandas as pd
import numpy as np
from functools import reduce
from typing import Union


class Data:
    """Fundamental object to share dataset and its meta information to use for training, evalutation, etc.
    This class is a common interface among modules for data analysis such as preparing, training, evaluation, and visualization.
    It takes pandas DataFrame as a main data type and contains additional meta attributes.
    It also opens common interfaces to simplify access to the core attributes for data processing.
    """
    options = dict(resolve_dtypes=True)
    
    def __init__(self, data:'TDATA'=None, columns=None):
        # TODO: support time-series - switch time index with flags
        # TODO: check_data should be used and copy meta data
        self._dataframe = Types.check_dataframe(data, columns)
        self.index_time = None
        self.meta = dict()
        self.label = None
        self.labels = list()
        self.features = list()

        if self.options['resolve_dtypes']:
            self.resolve_dtypes()

    @classmethod
    def deepcopy(cls, data:'Data') -> pd.DataFrame:
        # TODO: Need to update whenever Data updates
        return data._dataframe.copy(deep=True)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self._dataframe._values, dtype=dtype)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe
        
    @dataframe.setter
    def dataframe(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        self._dataframe = dataframe

    @property
    def shape(self) -> tuple:
        return self._dataframe.shape

    @property
    def columns(self) -> list:
        return self._dataframe.columns.tolist()

    @property
    def dtypes(self) -> dict:
        return self._dataframe.dtypes.to_dict()

    def head(self, lines=5) -> pd.DataFrame:
        return self._dataframe.head(lines)

    def tail(self, lines=5) -> pd.DataFrame:
        return self._dataframe.tail(lines)

    def crop(self, rows:list=None, cols:list=None) -> 'Data':
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
        idx_cols = IndexList(cols, self._dataframe.shape[1])
        df = self._dataframe.iloc[idx_rows.list, idx_cols.list]
        return Data(df)

    def equals(self, data:'Data', ndigits:int=None, ignore_index=True, ignore_types=True) -> bool:
        """Compares elementwisely

        Args:
            data (Data): Data to be compared
            ndigits (int): precision to compare
            ignore_index (bool, optional): check it except indices. Defaults to True.

        Returns:
            bool: Equal or not
        """
        # df.equals compares indices. Compare only contents and column names.
        if ignore_index:
            df_mine = self._dataframe.reset_index(drop=True)
            df_given = data._dataframe.reset_index(drop=True)
        else:
            df_mine = self._dataframe
            df_given = data._dataframe

        # df.equals fails different dtypes whose values are same. i.e. equals() is False with 10 and 10.0
        if ignore_types:
            columns = df_mine.columns[(df_mine.dtypes == np.int64) | (df_mine.dtypes == np.float)]
            df_mine[columns] = df_mine[columns].astype(np.float32)
            columns = df_given.columns[(df_given.dtypes == np.int64) | (df_given.dtypes == np.float)]
            df_given[columns] = df_given[columns].astype(np.float32)

        if ndigits:
            return df_mine.round(ndigits).equals(df_given.round(ndigits))
        else:
            return df_mine.equals(df_given)

    # TODO: implement sort()

    def time_series_on(self, colname:str):
        if self.index_time:
            self.time_series_off()
        if np.issubdtype(self._dataframe[colname].dtype, np.datetime64) is False:
            self._dataframe[colname] = pd.to_datetime(self._dataframe[colname], errors='coerce')
        self._dataframe.set_index(colname, inplace=True)
        self.index_time = colname
        return self

    def time_series_off(self):
        if self.index_time:
            self._dataframe.reset_index(inplace=True)
            self.index_time = None
        return self

    @property
    def numeric_cols(self) -> pd.Index:
        df = self._dataframe
        # variables = pd.Series(df.columns)
        cols = df.columns
        idx_numerics = []
        for idx, col in enumerate(cols):
            if (df[col].dtypes == 'int64' or df[col].dtype == 'float64'):
                idx_numerics.append(idx)
        cols = cols[idx_numerics]
        return cols

    def resolve_dtypes(self) -> pd.DataFrame:
        # TODO: check datetime after convert_dtypes()
        return self._dataframe.convert_dtypes().dtypes

    def convert_dtypes(self, dtypes:dict=None) -> Union[pd.DataFrame, 'Data']:
        # TODO: implement convert columns' dtypes. If dtypes is None, convert all with resolve_dtypes()
        return self

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

    @property
    def info(self):
        return self._dataframe.info()

    @property
    def description(self) -> pd.DataFrame:
        return self._dataframe.describe()

    # Inherits built-in functions here
    def __repr__(self):
        return repr(self._dataframe)


TDATA = Union[pd.DataFrame, np.ndarray, Data, dict]
TLT = Union[list, pd.Series, pd.Index, np.ndarray]
TAR = Union[pd.DataFrame, np.ndarray, Data]


class Types:
    @classmethod
    def check_dataframe(cls, data: TDATA, columns:list=None) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return_value = data
        elif isinstance(data, dict):
            return_value = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            return_value = pd.DataFrame(data, columns=columns)
        elif isinstance(data, Data):
            # return_value = self.deepcopy(data)
            return_value = data.dataframe
        elif data is None:
            return_value = pd.DataFrame()
        else:
            # TDOD: check the data type and the shape in case of unsupported format.
            # raise ValueError(data)
            return_value = pd.DataFrame(data)
        return return_value

    @classmethod
    def check_list(cls, data: TLT) -> list:
        if isinstance(data, list):
            return_value = data
        elif isinstance(data, pd.Series) or isinstance(data, pd.Index):
            return_value = data.to_list()
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            return_value = data.tolist()
        elif data is None:
            return_value = list()
        else:
            # check the data type and the shape in case of unsupported format.
            raise TypeError(data)
        return return_value

    @classmethod
    def check_data(cls, data: TDATA) -> Data:
        if isinstance(data, Data):
            return_value = data
        else:
            return_value = Data(cls.check_dataframe(data))
        return return_value

    @classmethod
    def check_array(cls, data: TAR) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return_value = data
        elif isinstance(data, Data):
            return_value = data.dataframe.to_numpy()
        elif isinstance(data, pd.DataFrame):
            return_value = data.to_numpy()
        else:
            raise TypeError(data)
        return return_value


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
        return list(range(indices[0], indices[1], indices[2]))

    @property
    def list(self):
        return self._elements

    @list.setter
    def list(self, indices):
        self._indices = indices
        self._elements = self.__convert(indices)
        

# Tests
class TableData(pd.DataFrame):
    def __init__(self):
        super().__init__()
    

class ListData(list):
    def __init__(self):
        super().__init__()


class Datap(pd.DataFrame):
    def __init__(self, data=None):
        super().__init__(data)