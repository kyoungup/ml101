from abc import ABCMeta, abstractmethod, abstractproperty
from pathlib import Path
import pandas as pd
import openpyxl
import pickle
from typing import Union
from ml101.data import Data


class BaseStream(metaclass=ABCMeta):
    """This is an abstract class to serialize Data class from/to various file formats.
    Subclasses should implment itself for specific file formats.
    """
    # TODO: implement better option structure
    # TODO: replace **{} parameters over Stream inherited classes
    __default_options = {}

    def __init__(self, pathe2file):
        self.path2file = Path(pathe2file)

    @abstractmethod
    def read(self, **kwargs):
        pass

    @abstractmethod
    def write(self, Data, **kwargs):
        # TODO: accept Data, pd.DataFrame, and more
        # TODO: if time-series, try to include index field
        pass

    @property
    def default(self):
        return dict(self.__default_options)


class StreamExcel(BaseStream):
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

    def write(self, data:Union[Data, pd.DataFrame], append=False, index=False, **kwargs):
        sheet_name = kwargs.get('sheet_name', 'Sheet1')
        df = Data.check_dataframe(data)
        if isinstance(data, Data) and data.index_time: index = True

        if append & self.path2file.exists():
            mode = 'a'
            with pd.ExcelWriter(self.path2file, mode=mode) as writer:  #pylint: disable=abstract-class-instantiated
                # To overwrite an existing sheet
                book = openpyxl.load_workbook(self.path2file)
                writer.book = book
                writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                df.to_excel(writer, sheet_name, index=index)
        else:
            mode = 'w'
            with pd.ExcelWriter(self.path2file, mode=mode) as writer:   #pylint: disable=abstract-class-instantiated
                df.to_excel(writer, sheet_name, index=index)


class StreamCSV(BaseStream):
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

    def write(self, data:Union[Data, pd.DataFrame], index=False, **kwargs):
        df = Data.check_dataframe(data)
        if isinstance(data, Data) and data.index_time: index = True
        df.to_csv(self.path2file, index=index, **kwargs)


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

    def write(self, data:Union[Data, pd.DataFrame], index=False, **kwargs):
        df = Data.check_dataframe(data)
        if isinstance(data, Data) and data.index_time: index = True
        df.to_csv(self.path2file, sep='\t', index=index, **kwargs)


class StreamPickle(BaseStream):
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


class Stream:
        """This is a factory class to detect file formats by file extention and make a proper instance for serialization

        Raises:
            ValueError: [description]

        Returns:
            DataStream: a file reader/writer for the specified file format
        """
        @classmethod
        # TODO: remove calling open()
        def open(cls, path2file):
            path2file = Path(path2file)

            format = path2file.suffix
            stream: BaseStream = None

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
