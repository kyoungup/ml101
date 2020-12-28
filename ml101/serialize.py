from abc import ABCMeta, abstractmethod, abstractproperty
from operator import attrgetter
from pathlib import Path
import pandas as pd
from functools import reduce
import openpyxl
from sklearn.base import TransformerMixin, BaseEstimator
from ml101.data import Data
import pickle


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
        stream = StreamFactory.open(self.path2file)
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
        stream = StreamFactory.open(self.path2file)
        stream.write(Data(X), **self.__kwargs())
        return X


class DataStream(metaclass=ABCMeta):
    """This is an abstract class to serialize Data class from/to various file formats.
    Subclasses should implment itself for specific file formats.
    """
    __default_options = {}

    def __init__(self, pathe2file):
        self.path2file = Path(pathe2file)

    @abstractmethod
    def read(self, **kwargs):
        pass

    @abstractmethod
    def write(self, Data, **kwargs):
        pass

    @property
    def default(self):
        return dict(self.__default_options)


class StreamExcel(DataStream):
    __default_options = {
        'sheet_name': 0,
        'header': 0
    }

    def __init__(self, path2file):
        super().__init__(path2file)
        self._options = self.default

    def __isexcelcolumns(self, cols) -> bool:
        return isinstance(cols, str) and cols.upper().isupper()

    def read(self, **kwargs) -> Data:
        """Refer to pandas' read_excel()

        Returns:
            Data:
        """
        df = pd.read_excel(self.path2file, engine='openpyxl', **kwargs)
        return Data(df)

    def write(self, data: Data, append=False, **kwargs):
        sheet_name = kwargs.get('sheet_name', 'Sheet1')

        if append & self.path2file.exists():
            mode = 'a'
            with pd.ExcelWriter(self.path2file, mode=mode) as writer:  #pylint: disable=abstract-class-instantiated
                # To overwrite an existing sheet
                book = openpyxl.load_workbook(self.path2file)
                writer.book = book
                writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                data.dataframe.to_excel(writer, sheet_name, index=False)
        else:
            mode = 'w'
            with pd.ExcelWriter(self.path2file, mode=mode) as writer:   #pylint: disable=abstract-class-instantiated
                data.dataframe.to_excel(writer, sheet_name, index=False)


class StreamCSV(DataStream):
    __default_options = {
        'sep': ',',
        'header': 0,
        'encoding': None
    }

    def __init__(self, path2file):
        super().__init__(path2file)

    def read(self, **kwargs) -> Data:
        df = pd.read_csv(self.path2file, **kwargs)
        return Data(df)

    def write(self, data, **kwargs):
        data.dataframe.to_csv(self.path2file, index=False, **kwargs)


class StreamTSV(StreamCSV):
    __default_options = {
        'header': 0,
        'encoding': None
    }

    def __init__(self, path2file):
        super().__init__(path2file)

    def read(self, **kwargs):
        df = pd.read_csv(self.path2file, sep='\t', **kwargs)
        return Data(df)

    def write(self, data, **kwargs):
        data.dataframe.to_csv(self.path2file, sep='\t', index=False, **kwargs)


class StreamPickle(DataStream):
    __default_options = {}

    def __init__(self, path2file):
        super().__init__(path2file)

    def read(self, **kwargs) -> Data:
        with self.path2file.open('rb') as f:
            data = pickle.load(f, **kwargs)
        return data

    def write(self, data, **kwargs):
        with self.path2file.open('wb') as f:
            pickle.dump(data, f, **kwargs)


class StreamFactory:
        """This is a factory class to detect file formats by file extention and make a proper instance for serialization

        Raises:
            ValueError: [description]

        Returns:
            DataStream: a file reader/writer for the specified file format
        """
        @classmethod
        def open(cls, path2file):
            path2file = Path(path2file)

            format = path2file.suffix
            stream: DataStream = None

            if format == '.csv':
                stream = StreamCSV(path2file)
            elif format == '.tsv':
                stream = StreamTSV(path2file)
            elif format in ['.xls', '.xlsx']:
                stream = StreamExcel(path2file)
            elif format in ['.pkl', '.zip', '.gz', '.bz']:
                stream = StreamPickle(path2file)
            else:
                raise ValueError(format.upper())
            return stream
