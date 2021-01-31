from abc import ABCMeta, abstractmethod, abstractproperty
from pathlib import Path
import pandas as pd
import openpyxl
import pickle
from typing import Union
from ml101.data import Data, Types, TDATA


class BaseStream(metaclass=ABCMeta):
    """This is an abstract class to serialize Data class from/to various file formats.
    Subclasses should implment itself for specific file formats.
    """
    def __init__(self, pathe2file):
        self.path2file = Path(pathe2file)

    @abstractmethod
    def read(self, pos_header:int=0, columns:Union[str,list,int]=None, **kwargs) -> Data:
        pass

    @abstractmethod
    def write(self, data:TDATA, index:bool=False, include_header=True, columns:list=None, **kwargs):
        pass


class StreamExcel(BaseStream):
    def __init__(self, path2file):
        super().__init__(path2file)

    def __isexcelcolumns(self, cols) -> bool:
        return isinstance(cols, str) and cols.upper().isupper()

    def read(self, sheet_name:Union[str,int,list]=0, pos_header:int=0, columns:Union[str,list,int]=None, **kwargs) -> Data:
        """Refer to pandas' read_excel()

        Returns:
            Data:
        """
        kwargs['sheet_name'] = sheet_name
        kwargs['header'] = pos_header
        kwargs['usecols'] = columns
        kwargs['engine'] = 'openpyxl'
        df = pd.read_excel(self.path2file, **kwargs)
        return Data(df)

    def write(self, data:TDATA, sheet_name:str='Sheet1', append=False, index=False, include_header:bool=True, columns:list=None, **kwargs):
        kwargs['sheet_name'] = sheet_name
        kwargs['index'] = index
        kwargs['header'] = include_header
        kwargs['columns'] = columns
        df = Types.check_dataframe(data)
        if isinstance(data, Data) and data.index_time: index = True

        if append & self.path2file.exists():
            mode = 'a'
            with pd.ExcelWriter(self.path2file, mode=mode) as writer:  #pylint: disable=abstract-class-instantiated
                # To overwrite an existing sheet
                book = openpyxl.load_workbook(self.path2file)
                writer.book = book
                writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                df.to_excel(writer, **kwargs)
        else:
            mode = 'w'
            with pd.ExcelWriter(self.path2file, mode=mode) as writer:   #pylint: disable=abstract-class-instantiated
                df.to_excel(writer, **kwargs)


class StreamCSV(BaseStream):
    def __init__(self, path2file):
        super().__init__(path2file)

    # TODO: support chunksize
    def read(self, pos_header:int=0, columns:Union[str,list,int]=None, **kwargs) -> Data:
        kwargs['header'] = pos_header
        kwargs['usecols'] = columns
        df = pd.read_csv(self.path2file, **kwargs)
        return Data(df)

    def write(self, data:TDATA, index:bool=False, include_header:bool=True, columns:list=None, **kwargs):
        kwargs['index'] = index
        kwargs['header'] = include_header
        kwargs['columns'] = columns
        df = Types.check_dataframe(data)
        if isinstance(data, Data) and data.index_time: index = True
        df.to_csv(self.path2file, **kwargs)


class StreamTSV(StreamCSV):
    def __init__(self, path2file):
        super().__init__(path2file)

    def read(self, pos_header:int=0, columns:Union[str,list,int]=None, **kwargs) -> Data:
        kwargs['sep'] = '\t'
        return super().read(pos_header=pos_header, columns=columns, **kwargs)

    def write(self, data:TDATA, index:bool=False, include_header:bool=True, columns:list=None, **kwargs):
        kwargs['sep'] = '\t'
        super().write(data, index=index, include_header=include_header, columns=columns, **kwargs)


class StreamPickle(BaseStream):
    def __init__(self, path2file):
        super().__init__(path2file)

    def read(self, **kwargs) -> Data:
        with self.path2file.open('rb') as f:
            data = pickle.load(f, **kwargs)
        return data

    def write(self, data, **kwargs):
        with self.path2file.open('wb') as f:
            pickle.dump(data, f, **kwargs)


class Stream(BaseStream):
        """This is a factory class to detect file formats by file extention and make a proper instance for serialization

        Raises:
            ValueError: [description]

        Returns:
            DataStream: a file reader/writer for the specified file format
        """
        def __new__(cls, path2file) -> BaseStream:
            assert path2file
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
